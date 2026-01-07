import dsmil as mil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, glob, copy
import pandas as pd
import numpy as np
from PIL import Image
from collections import OrderedDict
from sklearn.utils import shuffle
import glob
from pathlib import Path
import models_vit

def interpolate_pos_embed(model, checkpoint_model):
    """
    Interpolate position embeddings for different image sizes
    """
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

class BagDataset():
    def __init__(self, csv_file, transform=None):
        self.files_list = csv_file
        self.transform = transform
    def __len__(self):
        return len(self.files_list)
    def __getitem__(self, idx):
        temp_path = self.files_list[idx]
        img = os.path.join(temp_path)
        img = Image.open(img)
        sample = {'input': img}
        
        if self.transform:
            sample = self.transform(sample)
        return sample 

class ToTensor(object):
    def __call__(self, sample):
        img = sample['input']
        img = VF.to_tensor(img)
        return {'input': img} 
    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img
class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample['input']
        img = VF.resize(img, self.size)
        return {'input': img}

def bag_dataset(args, csv_file_path):
    transformed_dataset = BagDataset(csv_file=csv_file_path,
                                    transform=Compose([
                                        Resize((224, 224)),
                                        ToTensor()
                                    ]))
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)

def compute_feats(args, bags_list, model, save_path=None, magnification='single'):
    model.eval()
    num_bags = len(bags_list)
    
    for i in range(0, num_bags):
        feats_list = []
        if magnification=='OHTS':
            csv_file_path = glob.glob(os.path.join(bags_list[i], '*.jpg')) 
        if magnification=='GRAPE':
            csv_file_path = glob.glob(os.path.join(bags_list[i], '*.jpg')) 
        elif magnification=='single' or magnification=='low':
            csv_file_path = glob.glob(os.path.join(bags_list[i], '*.jpg')) + glob.glob(os.path.join(bags_list[i], '*.jpeg'))
        elif magnification=='high':
            csv_file_path = glob.glob(os.path.join(bags_list[i], '*'+os.sep+'*.jpg')) + glob.glob(os.path.join(bags_list[i], '*'+os.sep+'*.jpeg'))
        
        dataloader, bag_size = bag_dataset(args, csv_file_path)
        
        with torch.no_grad():
            for iteration, batch in enumerate(dataloader):
                patches = batch['input'].float().cuda()
                # 使用模型的encoder部分获取特征
                feats = model.forward_features(patches)  # 这里假设model有forward_features方法
                feats = feats.cpu().numpy()
                feats_list.extend(feats)
                sys.stdout.write('\r Computed: {}/{} -- {}/{}'.format(i+1, num_bags, iteration+1, len(dataloader)))
        
        if len(feats_list) == 0:
            print('No valid patch extracted from: ' + str(bags_list[i]))
        else:
            df = pd.DataFrame(feats_list)
            os.makedirs(os.path.join(args.dataset, str(bags_list[i]).split(os.path.sep)[-2]), exist_ok=True)
            df.to_csv(os.path.join(args.dataset, str(bags_list[i]).split(os.path.sep)[-2], 
                     str(bags_list[i]).split(os.path.sep)[-1]+'.csv'), index=False, float_format='%.4f')

def main(args):
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        
        # 删除原有的head
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # 插值位置编码
        interpolate_pos_embed(model, checkpoint_model)

        # 加载预训练模型
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        # 设置head为直接映射
        model.head = torch.nn.Identity()

    model.to(device)
    model.eval()

    # 设置路径
    if args.magnification == 'OHTS':
        bags_path = os.path.join('datasets', args.dataset, '*', '*')
        bags_list = glob.glob(bags_path)# 移除'case'前缀
        # 设置路径
    elif args.magnification == 'GRAPE':
        bags_path = os.path.join('datasets', args.dataset, '*')
        bags_list = glob.glob(bags_path)# 
    else:
        bags_path = os.path.join('WSI', args.dataset, 'single', '*', '*')
        bags_list = glob.glob(bags_path)
    
    feats_path = os.path.join('datasets', args.dataset)
    os.makedirs(feats_path, exist_ok=True)
    
    # 计算特征
    compute_feats(args, bags_list, model, feats_path, args.magnification)
    
    # 创建标签CSV
    n_classes = glob.glob(os.path.join('datasets', args.dataset, '*'+os.sep))
    n_classes = sorted(n_classes)
    all_df = []
    for i, item in enumerate(n_classes):
        bag_csvs = glob.glob(os.path.join(item, '*.csv'))
        bag_df = pd.DataFrame(bag_csvs)
        bag_df['label'] = i
        bag_df.to_csv(os.path.join('datasets', args.dataset, item.split(os.path.sep)[2]+'.csv'), index=False)
        all_df.append(bag_df)
    
    bags_path = pd.concat(all_df, axis=0, ignore_index=True)
    bags_path = shuffle(bags_path)
    bags_path.to_csv(os.path.join('datasets', args.dataset+'.csv'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('MAE feature extraction', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--model', default='vit_base_patch16', type=str)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--finetune', default='/home/lin01231/song0760/mae/checkpoint/OHTS/checkpoint-399.pth', help='finetune from checkpoint')
    parser.add_argument('--drop_path', type=float, default=0.0, help='Drop path rate')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
    parser.add_argument('--nb_classes', default=2, type=int)
    parser.add_argument('--dataset', default='OHTS_1', type=str)
    parser.add_argument('--magnification', default='OHTS', type=str)
    
    args = parser.parse_args()
    
        
    main(args)
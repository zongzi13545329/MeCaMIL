import sys
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import yaml
from pprint import pprint
import json

import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import pickle

from utils.utils import cosine_scheduler, Struct, set_seed, Wandb_Writer, ema_update, save_model
from modules import causalmil
from utils.utils import MetricLogger, SmoothedValue, adjust_learning_rate
from timm.utils import accuracy, AverageMeter
import wandb
import torchmetrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================================================================
# === 优化后的H5数据集加载器 ===
# =========================================================================

class H5_CausalMIL_Dataset(Dataset):
    """优化后的H5数据集加载器，返回结构化字典"""
    
    def __init__(self, h5_file_path, slide_ids, args, use_demo=True, min_patches=1):
        self.h5_file_path = h5_file_path
        self.args = args
        self.use_demo = use_demo
        self.min_patches = min_patches
        
        self.feats_size = args.feats_size
        self.num_classes = args.num_classes
        self.repetition = [30] * args.num_classes + [9, 9, 9, 9, 9, 9, 9, 8]
        self.u_dim = sum(self.repetition)
        self.demographic_dim = 8
        
        self.slide_ids = self._filter_slides_by_patch_count(slide_ids)
        
        print(f"CausalMIL Dataset: {len(self.slide_ids)}/{len(slide_ids)} slides (>= {min_patches} patches)")

    def _filter_slides_by_patch_count(self, slide_ids):
        filtered_slides = []
        try:
            with h5py.File(self.h5_file_path, 'r') as h5_file:
                for slide_id in slide_ids:
                    if slide_id not in h5_file:
                        continue
                    feats_shape = h5_file[slide_id]['feat'].shape
                    num_patches = feats_shape[0] if len(feats_shape) > 1 else 1
                    if num_patches >= self.min_patches:
                        filtered_slides.append(slide_id)
        except Exception as e:
            print(f"Error filtering: {e}")
            return slide_ids
        return filtered_slides

    def __len__(self):
        return len(self.slide_ids)
    
    def __getitem__(self, idx):
        slide_id = self.slide_ids[idx]
        try:
            with h5py.File(self.h5_file_path, 'r') as h5_file:
                if slide_id not in h5_file:
                    return None
                slide_grp = h5_file[slide_id]
                
                feats = np.array(slide_grp['feat'], dtype=np.float32)
                if feats.ndim == 1:
                    feats = feats.reshape(1, -1)
                
                label = np.array(slide_grp['label'], dtype=np.float32)
                if label.ndim == 0:
                    label = np.array([label])
                
                result = {
                    'feats': torch.tensor(feats, dtype=torch.float32),
                    'label': torch.tensor(label, dtype=torch.float32),
                    'slide_id': slide_id,
                    'num_patches': feats.shape[0]
                }
                
                if self.use_demo:
                    if 'u_vector' not in slide_grp or 'demographic' not in slide_grp:
                        return None
                    u_vector = np.array(slide_grp['u_vector'], dtype=np.float32)
                    demographic = np.array(slide_grp['demographic'], dtype=np.float32)
                    result['u_vector'] = torch.tensor(u_vector, dtype=torch.float32)
                    result['demographic'] = torch.tensor(demographic, dtype=torch.float32)
                else:
                    result['u_vector'] = None
                    result['demographic'] = None
                
                return result
        except Exception as e:
            print(f"Error loading {slide_id}: {e}")
            return None
    
    def get_slide_info(self, idx):
        slide_id = self.slide_ids[idx]
        with h5py.File(self.h5_file_path, 'r') as h5_file:
            if slide_id not in h5_file:
                return None
            slide_grp = h5_file[slide_id]
            return {
                'slide_id': slide_id,
                'feat_shape': slide_grp['feat'].shape,
                'has_demographic': 'demographic' in slide_grp,
                'has_u_vector': 'u_vector' in slide_grp,
            }


def safe_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    if len(batch) == 1:
        return batch[0]
    try:
        collated = {}
        for key in batch[0].keys():
            if key in ['slide_id', 'num_patches']:
                collated[key] = [item[key] for item in batch]
            elif batch[0][key] is not None:
                collated[key] = torch.stack([item[key] for item in batch], dim=0)
            else:
                collated[key] = None
        return collated
    except:
        return None


def create_h5_dataloaders(h5_file_path, args, train_slides, val_slides, test_slides, 
                         use_demo=True, batch_size=1, num_workers=4, min_patches=1):
    train_dataset = H5_CausalMIL_Dataset(h5_file_path, train_slides, args, use_demo, min_patches)
    val_dataset = H5_CausalMIL_Dataset(h5_file_path, val_slides, args, use_demo, min_patches)
    test_dataset = H5_CausalMIL_Dataset(h5_file_path, test_slides, args, use_demo, min_patches)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True, collate_fn=safe_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True, collate_fn=safe_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True, collate_fn=safe_collate_fn)
    
    return train_loader, val_loader, test_loader


def verify_data_consistency(h5_dataset, slide_idx=0):
    print("\n" + "="*60)
    print("=== H5数据格式验证 ===")
    print("="*60)
    
    if len(h5_dataset) == 0:
        print("Dataset is empty")
        return
    
    data_dict = h5_dataset[slide_idx]
    if data_dict is None:
        print("Failed to load data")
        return
    
    print(f"\nSlide ID: {data_dict['slide_id']}")
    print(f"Patches: {data_dict['num_patches']}")
    print(f"Feats: {tuple(data_dict['feats'].shape)}")
    print(f"Label: {tuple(data_dict['label'].shape)}")
    
    if h5_dataset.use_demo and data_dict['demographic'] is not None:
        print(f"Demographic: {tuple(data_dict['demographic'].shape)} (used as model input)")
        print(f"U_vector: {tuple(data_dict['u_vector'].shape)} (stored but not used)")
    
    print("✓ Verification complete\n" + "="*60)


def extract_data_from_h5_batch(batch, args, device):
    """提取数据，保持与原版模型输入接口一致"""
    if batch is None:
        return None, None, None, None

    bag_feats = batch['feats'].to(device)
    bag_label = batch['label'].unsqueeze(0).to(device)
    N_patches = bag_feats.size(0)
    
    if args.use_demographic:
        if batch['demographic'] is None:
            return None, None, None, None
        
        demographic_data = batch['demographic'].to(device)
        gender_label = demographic_data[:2]
        race_label = demographic_data[2:7]
        age_label = demographic_data[7].unsqueeze(0)
        
        demo_info = {
            'demographic_data': demographic_data,
            'gender_label': gender_label,
            'race_label': race_label,
            'age_label': age_label
        }
        
        u_for_model = demographic_data.unsqueeze(0).expand(N_patches, -1)
        return bag_feats, u_for_model, bag_label, demo_info
    else:
        return bag_feats, None, bag_label, None


def dropout_patches(feats, p):
    if feats is None or feats.numel() == 0:
        return feats
    num_rows = feats.size(0)
    num_rows_to_select = max(1, int(num_rows * (1 - p)))
    random_indices = torch.randperm(num_rows)[:num_rows_to_select]
    return feats[random_indices]


def train_one_epoch_with_h5(model, criterion, data_loader, optimizer, device, epoch, conf):
    if len(data_loader.dataset) == 0:
        return 0.0

    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('bag_loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('max_loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    l1_loss = torch.nn.L1Loss()
    total_loss = 0

    for data_it, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if batch is None:
            continue
        
        adjust_learning_rate(optimizer, epoch + data_it/len(data_loader), conf)
        optimizer.zero_grad()

        bag_feats, u, bag_label, demo_info = extract_data_from_h5_batch(batch, conf, device)
        
        if bag_feats is None or bag_feats.numel() == 0:
            continue
        
        bag_feats = dropout_patches(bag_feats, 1 - conf.dropout_patch)
        bag_feats = bag_feats.view(-1, conf.feats_size)
        
        if conf.use_demographic and u is not None:
            if u.size(0) != bag_feats.size(0):
                u = u[:1].expand(bag_feats.size(0), -1)

        ins_prediction, bag_prediction, result = model(bag_feats, u)
        
        max_prediction, _ = torch.max(ins_prediction, 0)
        max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
        bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
        
        if conf.use_demographic and demo_info is not None:
            if 'decoded_demographics' in result:
                su = torch.mean(result['decoded_demographics'][:, :2], 0)
                ru = torch.mean(result['decoded_demographics'][:, 2:7], 0)
                au = torch.mean(result['decoded_demographics'][:, 7], 0)
                
                SuLoss = criterion(su.unsqueeze(0), demo_info['gender_label'].unsqueeze(0))
                RuLoss = criterion(ru.unsqueeze(0), demo_info['race_label'].unsqueeze(0))
                AuLoss = l1_loss(au.unsqueeze(0), demo_info['age_label'].unsqueeze(0))
                
                loss = 0.5*bag_loss + 0.4*max_loss + 0.033*SuLoss + 0.033*RuLoss + 0.033*AuLoss
            else:
                loss = 0.5*bag_loss + 0.5*max_loss
        else:
            loss = 0.5*bag_loss + 0.5*max_loss
        # if conf.use_demographic and 'x_contrib' in result and 'u_contrib' in result:
        #     contrastive_loss = model.b_classifier.graph.contrastive_loss(
        #         result['x_contrib'], result['u_contrib']
        #     )
        #     loss = loss + 0.05 * contrastive_loss
        loss.backward()
        if conf.grad_clipping > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), conf.grad_clipping)
        optimizer.step()

        total_loss += loss.item()
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        metric_logger.update(bag_loss=bag_loss.item())
        metric_logger.update(max_loss=max_loss.item())
        
        if conf.wandb_mode != 'disabled':
            wandb.log({'train/loss': loss.item(), 'train/bag_loss': bag_loss.item(), 
                      'train/max_loss': max_loss.item()})
            
    return total_loss / len(data_loader)


@torch.no_grad()
def evaluate_with_h5(model, criterion, val_loader, device, conf, split):
    model.eval()
    all_bag_pred_probas = []
    all_bag_labels = []
    losses = AverageMeter()

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc=f"Evaluating {split}")):
            if batch is None:
                continue

            try:
                bag_feats, u, bag_label, _ = extract_data_from_h5_batch(batch, conf, device)
                if bag_feats is None:
                    continue
                
                _, bag_prediction, _ = model(bag_feats, u)
                loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
                losses.update(loss.item())

                pred_proba = F.softmax(bag_prediction, dim=1)
                all_bag_pred_probas.append(pred_proba.cpu())
                all_bag_labels.append(bag_label.cpu())
            except Exception as e:
                continue

    if not all_bag_pred_probas:
        return 0.0, 0.0, 0.0, 0.0

    all_bag_pred_probas = torch.cat(all_bag_pred_probas, dim=0)
    all_bag_labels = torch.cat(all_bag_labels, dim=0)
    
    if conf.num_classes == 2:
        try:
            val_auc = roc_auc_score(all_bag_labels[:, 1].numpy(), all_bag_pred_probas[:, 1].numpy())
        except:
            val_auc = 0.0
        bag_pred_classes = (all_bag_pred_probas[:, 1] > 0.5).long()
        bag_true_classes = all_bag_labels.argmax(dim=1).long()
        val_acc = accuracy_score(bag_true_classes.numpy(), bag_pred_classes.numpy())
        val_f1 = f1_score(bag_true_classes.numpy(), bag_pred_classes.numpy(), 
                         average='binary', zero_division=0)
    else:
        val_auc = roc_auc_score(all_bag_labels.numpy(), all_bag_pred_probas.numpy(), 
                               average='macro', multi_class='ovr')
        bag_pred_classes = all_bag_pred_probas.argmax(dim=1).long()
        bag_true_classes = all_bag_labels.argmax(dim=1).long()
        val_acc = accuracy_score(bag_true_classes.numpy(), bag_pred_classes.numpy())
        val_f1 = f1_score(bag_true_classes.numpy(), bag_pred_classes.numpy(), 
                         average='macro', zero_division=0)
    
    return val_auc, val_acc, val_f1, losses.avg


def save_best_state(best_state, save_path):
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        serializable_state = {}
        for key, value in best_state.items():
            if isinstance(value, (torch.Tensor, np.ndarray)):
                serializable_state[key] = float(value)
            else:
                serializable_state[key] = value
        with open(save_path, 'w') as f:
            json.dump(serializable_state, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving state: {e}")


def load_best_state(load_path):
    try:
        if os.path.exists(load_path):
            with open(load_path, 'r') as f:
                return json.load(f)
    except:
        pass
    return None


def save_interpretability_analysis(model, test_loader, device, conf, save_dir):
    model.eval()
    interpretability_data = {
        'attention_maps': [],
        'causal_attributions': [],
        'causal_graph_attention': [],
        'predictions': [],
        'labels': []
    }
    
    print("\nGenerating Interpretability Analysis...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Interpretability")):
            if batch is None or i >= 50:
                continue
                
            try:
                bag_feats, u, bag_label, _ = extract_data_from_h5_batch(batch, conf, device)
                if bag_feats is None:
                    continue
                
                interp_outputs = model.get_interpretability_outputs(bag_feats, u)
                
                if interp_outputs['attention_weights'] is not None:
                    interpretability_data['attention_maps'].append(
                        interp_outputs['attention_weights'].cpu())
                
                if interp_outputs['causal_attributions']:
                    attr_dict = {}
                    for key, value in interp_outputs['causal_attributions'].items():
                        if isinstance(value, torch.Tensor):
                            attr_dict[key] = value.cpu()
                    interpretability_data['causal_attributions'].append(attr_dict)
                
                if 'causal_graph_info' in interp_outputs:
                    graph_info = interp_outputs['causal_graph_info']
                    if 'attention' in graph_info:
                        interpretability_data['causal_graph_attention'].append(
                            graph_info['attention'].cpu())
                
                if interp_outputs['prediction_bag'] is not None:
                    interpretability_data['predictions'].append(interp_outputs['prediction_bag'].cpu())
                    interpretability_data['labels'].append(bag_label.cpu())
            except:
                continue
    
    demo_suffix = "with_demo" if conf.use_demographic else "no_demo"
    save_path = os.path.join(save_dir, f"interpretability_{conf.dataset}_{demo_suffix}.pth")
    torch.save(interpretability_data, save_path)
    print(f"Saved: {save_path}")
    return interpretability_data


@torch.no_grad()
def perform_sensitivity_analysis(model, test_loader, device, conf, save_dir):
    model.eval()
    sensitivity_results = {'baseline': [], 'without_u_to_z': [], 'do_u_zero': []}
    
    print("\nPerforming Sensitivity Analysis...")
    
    for i, batch in enumerate(tqdm(test_loader, desc="Sensitivity")):
        if batch is None or i >= 20:
            continue
            
        try:
            bag_feats, u, bag_label, _ = extract_data_from_h5_batch(batch, conf, device)
            if bag_feats is None or u is None:
                continue
            
            results = model.perform_sensitivity_analysis(bag_feats, u)
            if results is not None:
                for key in sensitivity_results:
                    if key in results:
                        sensitivity_results[key].append(results[key].cpu())
        except:
            continue
    
    if any(sensitivity_results.values()):
        save_path = os.path.join(save_dir, f"sensitivity_{conf.dataset}.pth")
        torch.save(sensitivity_results, save_path)
        print(f"Saved: {save_path}")
    
    return sensitivity_results


def get_arguments():
    parser = argparse.ArgumentParser('CausalMIL training')
    parser.add_argument('--config', default='config/tcga_natural_supervised_config.yml')
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument('--wandb_mode', default='disabled', choices=['offline', 'online', 'disabled'])
    parser.add_argument('--grad_clipping', default=5, type=float)
    parser.add_argument('--dataset', default='TCGA_lung', choices=['TCGA_lung', 'sum'], type=str)
    parser.add_argument('--causal_alpha', default=0.1, type=float)
    parser.add_argument('--structural_depth', default=1, type=int)
    parser.add_argument('--use_demographic', action='store_true')
    parser.add_argument('--causal_reg', default=0.01, type=float)
    parser.add_argument('--act', default='relu', type=str)
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--log_iter', default=100, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--pretrain', default='natural_supervised')
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument('--use_causal_graph', action='store_true')
    parser.add_argument('--perform_sensitivity', action='store_true')
    parser.add_argument('--save_interpretability', action='store_true', default=True)
    parser.add_argument('--min_patches', type=int, default=1)
    return parser.parse_args()


def main():
    args = get_arguments()
    
    with open(args.config, "r") as ymlfile:
        c = yaml.load(ymlfile, Loader=yaml.FullLoader)
        c.update(vars(args))
        conf = Struct(**c)

    if conf.pretrain == 'medical_ssl':
        conf.feats_size = 384
    elif conf.pretrain == 'natural_supervised':
        conf.feats_size = 2048
    elif conf.pretrain == 'path-clip-L-336':
        conf.feats_size = 768
    
    conf.num_classes = 2 if args.dataset == 'TCGA_lung' else 5
    if not hasattr(conf, 'dropout_patch'):
        conf.dropout_patch = 0.0

    project_name = "CausalMIL-Optimized-Demo" if conf.use_demographic else "CausalMIL-Optimized-NoDemo"
    wandb.init(project=project_name, config=vars(conf), mode=conf.wandb_mode)
    
    ckpt_dir = "saved_models"
    os.makedirs(ckpt_dir, exist_ok=True)

    print("\n" + "="*60)
    print("CausalMIL Training (Optimized)")
    print("="*60)
    pprint(vars(conf))
    print("="*60)
    
    set_seed(args.seed)

    h5_file_name = f'{args.dataset}_pretrain_{conf.pretrain}.h5'
    h5_file_path = os.path.join(conf.data_dir, h5_file_name)
    
    if not os.path.exists(h5_file_path):
        print(f"H5 file not found: {h5_file_path}")
        return

    with h5py.File(h5_file_path, 'r') as h5_file:
        all_slide_ids = list(h5_file.keys())
        print(f"\nH5 file: {len(all_slide_ids)} slides")

    train_slides, temp_slides = train_test_split(all_slide_ids, test_size=0.4, random_state=conf.seed)
    val_slides, test_slides = train_test_split(temp_slides, test_size=0.5, random_state=conf.seed)
    
    print(f"Split: train={len(train_slides)}, val={len(val_slides)}, test={len(test_slides)}")

    train_loader, val_loader, test_loader = create_h5_dataloaders(
        h5_file_path, conf, train_slides, val_slides, test_slides,
        use_demo=conf.use_demographic, batch_size=1,
        num_workers=conf.num_workers, min_patches=args.min_patches)

    if len(train_loader.dataset) > 0:
        verify_data_consistency(train_loader.dataset, 0)

    i_classifier = causalmil.FCLayer(conf.feats_size, conf.num_classes)
    
    if conf.use_demographic:
        b_classifier = causalmil.BClassifier(
            input_size=conf.feats_size, output_class=conf.num_classes,
            u_dim=8, hidden_dim=128, dropout_v=args.dropout, 
            nonlinear=True, passing_v=False, causal=True, 
            convDepth=args.structural_depth, use_causal_graph=args.use_causal_graph)
    else:
        b_classifier = causalmil.BClassifier(
            input_size=conf.feats_size, output_class=conf.num_classes,
            u_dim=0, hidden_dim=128, dropout_v=args.dropout,
            nonlinear=True, passing_v=False, causal=False, convDepth=0)
    
    model = causalmil.MILNet(i_classifier, b_classifier).to(device)

# ========== 添加参数量统计 ==========
    def count_parameters(model):
        """统计模型参数量"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print("\n" + "="*60)
        print("MODEL PARAMETERS")
        print("="*60)
        print(f"Total parameters:      {total_params:,}")
        print(f"Trainable parameters:  {trainable_params:,}")
        print(f"Non-trainable params:  {total_params - trainable_params:,}")
        print(f"Model size (MB):       {total_params * 4 / 1024 / 1024:.2f}")
        
        return total_params, trainable_params

    total_params, trainable_params = count_parameters(model)
    print (total_params, trainable_params)

    # 可选：记录到wandb
    if conf.wandb_mode != 'disabled':
        wandb.config.update({
            'total_params': total_params,
            'trainable_params': trainable_params
        })

    criterion = nn.BCEWithLogitsLoss()
    param_groups = [
    {'params': model.i_classifier.parameters(), 'lr': conf.lr},
    {'params': model.b_classifier.graph.parameters(), 'lr': conf.lr * 0.5},  # 因果图用更小学习率
    {'params': model.b_classifier.fcc.parameters(), 'lr': conf.lr}
    ]

    optimizer = torch.optim.AdamW(  # AdamW比Adam更好
        param_groups,
        weight_decay=0.01,  # 增加权重衰减
        betas=(0.9, 0.999)
    )

    # Warmup调度器
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # 线性预热
        else:
            # 余弦退火
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (conf.train_epoch - warmup_epochs)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr, 
    #                              betas=(0.5, 0.9), weight_decay=conf.wd)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, conf.train_epoch, 0.000005)

    demo_suffix = "with_demo" if conf.use_demographic else "no_demo"
    best_model_name = f"causal_{args.dataset}_{demo_suffix}_best.pth"
    last_model_name = f"causal_{args.dataset}_{demo_suffix}_last.pth"
    best_state_name = f"causal_{args.dataset}_{demo_suffix}_state.json"
    
    best_state_path = os.path.join(ckpt_dir, best_state_name)
    best_state = load_best_state(best_state_path) or {
        'epoch': -1, 'val_acc': 0, 'val_auc': 0, 'val_f1': 0,
        'test_acc': 0, 'test_auc': 0, 'test_f1': 0
    }
    
    print(f"\nTraining starts...")
    
    for epoch in range(conf.train_epoch):
        train_loss = train_one_epoch_with_h5(model, criterion, train_loader, 
                                             optimizer, device, epoch, conf)
        val_auc, val_acc, val_f1, val_loss = evaluate_with_h5(model, criterion, val_loader, 
                                                              device, conf, 'Val')
        test_auc, test_acc, test_f1, test_loss = evaluate_with_h5(model, criterion, test_loader, 
                                                                  device, conf, 'Test')
        
        if scheduler:
            scheduler.step()

        if conf.wandb_mode != 'disabled':
            wandb.log({'train/loss': train_loss, 'val/auc': val_auc, 'val/acc': val_acc, 
                      'val/f1': val_f1, 'test/auc': test_auc, 'test/acc': test_acc, 
                      'test/f1': test_f1, 'epoch': epoch})

        current_score = val_f1 + val_auc
        best_score = best_state['val_f1'] + best_state['val_auc']
        
        print(f"Epoch {epoch}: Val AUC={val_auc:.4f} F1={val_f1:.4f} | Test AUC={test_auc:.4f}")
        
        if current_score > best_score:
            best_state.update({'epoch': epoch, 'val_acc': val_acc, 'val_auc': val_auc, 
                             'val_f1': val_f1, 'test_acc': test_acc, 'test_auc': test_auc, 
                             'test_f1': test_f1})
            save_model(conf=conf, model=model, optimizer=optimizer, epoch=epoch,
                       save_path=os.path.join(ckpt_dir, best_model_name))
            save_best_state(best_state, best_state_path)
            print(f"  ✓ New best model saved!")

    save_model(conf=conf, model=model, optimizer=optimizer, epoch=epoch,
               save_path=os.path.join(ckpt_dir, last_model_name))
    save_best_state(best_state, best_state_path)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Best Epoch: {best_state['epoch']}")
    print(f"Val  - ACC: {best_state['val_acc']:.4f}, AUC: {best_state['val_auc']:.4f}, F1: {best_state['val_f1']:.4f}")
    print(f"Test - ACC: {best_state['test_acc']:.4f}, AUC: {best_state['test_auc']:.4f}, F1: {best_state['test_f1']:.4f}")
    print(f"\nSaved: {best_model_name}, {last_model_name}, {best_state_name}")
    print("="*60)

    if not args.no_log:
        print("\nPOST-TRAINING ANALYSIS")
        
        if args.perform_sensitivity and conf.use_demographic:
            perform_sensitivity_analysis(model, test_loader, device, conf, ckpt_dir)
        
        if args.save_interpretability:
            save_interpretability_analysis(model, test_loader, device, conf, ckpt_dir)
        
        if conf.use_demographic and hasattr(model.b_classifier, 'graph'):
            causal_graph_info = {
                'adjacency_matrix': model.b_classifier.graph.causal_adjacency.cpu().numpy(),
                'node_names': ['X (Image)', 'U (Demo)', 'Z (Latent)']
            }
            graph_path = os.path.join(ckpt_dir, f"causal_graph_{conf.dataset}.pkl")
            with open(graph_path, 'wb') as f:
                pickle.dump(causal_graph_info, f)
            print(f"Saved: {graph_path}")
    
    wandb.finish()
    print("\n✓ All done!")


if __name__ == '__main__':
    main()
import causalmil as mil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, glob
import pandas as pd
import numpy as np
from PIL import Image
from collections import OrderedDict
from skimage import exposure, io, img_as_ubyte, transform
import warnings
import openslide
from xml.etree import ElementTree as ET
import cv2
import torch.nn.functional as F

class BagDataset():
    def __init__(self, csv_file, transform=None):
        self.files_list = csv_file
        self.transform = transform
    def __len__(self):
        return len(self.files_list)
    def __getitem__(self, idx):
        path = self.files_list[idx]
        img = Image.open(path)
        img_name = path.split(os.sep)[-1]
        img_pos = np.asarray([int(img_name.split('.')[0].split('_')[0]), int(img_name.split('.')[0].split('_')[1])]) # row, col
        sample = {'input': img, 'position': img_pos}
        
        if self.transform:
            sample = self.transform(sample)
        return sample 

class ToTensor(object):
    def __call__(self, sample):
        img = sample['input']
        img = VF.to_tensor(img)
        sample['input'] = img
        return sample
    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

def bag_dataset(args, csv_file_path):
    transformed_dataset = BagDataset(csv_file=csv_file_path,
                                    transform=Compose([
                                        ToTensor()
                                    ]))
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)

def load_wsi_thumbnail(wsi_path, target_size=2048):
    """Load WSI and generate thumbnail"""
    try:
        slide = openslide.OpenSlide(wsi_path)
        # Get appropriate level for thumbnail
        thumb = slide.get_thumbnail((target_size, target_size))
        slide.close()
        return np.array(thumb)
    except Exception as e:
        print(f"Error loading WSI {wsi_path}: {e}")
        return None

def load_mask(mask_path, target_size=None):
    """Load mask file and resize if needed"""
    try:
        mask = io.imread(mask_path)
        if target_size is not None:
            mask = transform.resize(mask, target_size, order=0, preserve_range=True, anti_aliasing=False)
        return mask.astype(np.uint8)
    except Exception as e:
        print(f"Error loading mask {mask_path}: {e}")
        return None

def parse_asap_annotation(xml_path, wsi_dimensions, target_size):
    """Parse ASAP XML annotation and create mask"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get scale factor
        wsi_width, wsi_height = wsi_dimensions
        scale_x = target_size[1] / wsi_width
        scale_y = target_size[0] / wsi_height
        
        # Create empty mask
        anno_mask = np.zeros(target_size, dtype=np.uint8)
        
        # Parse annotations
        for annotation in root.findall('.//Annotation'):
            group = annotation.get('PartOfGroup', '')
            coordinates = []
            
            for coord in annotation.findall('.//Coordinate'):
                x = float(coord.get('X'))
                y = float(coord.get('Y'))
                coordinates.append([int(x * scale_x), int(y * scale_y)])
            
            if len(coordinates) > 0:
                coordinates = np.array(coordinates, dtype=np.int32)
                if group == 'Tumor':
                    cv2.fillPoly(anno_mask, [coordinates], 2)  # Tumor = 2
                elif group == 'Exclusion':
                    cv2.fillPoly(anno_mask, [coordinates], 0)  # Exclusion = 0
        
        return anno_mask
    except Exception as e:
        print(f"Error parsing annotation {xml_path}: {e}")
        return None

def create_overlay(wsi_thumb, heatmap=None, mask=None, anno_mask=None, 
                   heatmap_alpha=0.5, mask_alpha=0.15):
    """Create overlay visualization
    
    Args:
        wsi_thumb: WSI thumbnail
        heatmap: Attention heatmap (red for predicted tumor regions)
        mask: Ground truth mask (1=normal, 2=tumor)
        anno_mask: Annotation mask
        heatmap_alpha: Alpha for attention heatmap (default 0.5 for deeper red)
        mask_alpha: Alpha for mask overlay (default 0.15 for high transparency)
    """
    overlay = wsi_thumb.copy().astype(np.float32) / 255.0
    
    # Add heatmap overlay (deeper red for attention)
    if heatmap is not None:
        heatmap_resized = transform.resize(heatmap, 
                                          (wsi_thumb.shape[0], wsi_thumb.shape[1]), 
                                          order=1, anti_aliasing=True)
        # Convert to RGB heatmap (deeper red channel)
        heatmap_rgb = np.zeros_like(overlay)
        heatmap_rgb[:, :, 0] = heatmap_resized[:, :, 0]  # Red channel for attention
        overlay = overlay * (1 - heatmap_alpha) + heatmap_rgb * heatmap_alpha
    
    # Add mask overlay (ground truth)
    if mask is not None:
        mask_resized = transform.resize(mask, 
                                       (wsi_thumb.shape[0], wsi_thumb.shape[1]), 
                                       order=0, preserve_range=True, anti_aliasing=False)
        # Only overlay tumor regions (mask value = 2) with high transparency red
        # Normal tissue (mask value = 1) keeps original WSI color
        mask_rgb = np.zeros_like(overlay)
        tumor_mask = (mask_resized == 2)
        mask_rgb[tumor_mask, 0] = 1.0  # Tumor = red
        
        # Only overlay where tumor is present with high transparency
        overlay[tumor_mask] = (overlay[tumor_mask] * (1 - mask_alpha) + 
                              mask_rgb[tumor_mask] * mask_alpha)
    
    # Add annotation overlay
    if anno_mask is not None:
        anno_resized = transform.resize(anno_mask, 
                                       (wsi_thumb.shape[0], wsi_thumb.shape[1]), 
                                       order=0, preserve_range=True, anti_aliasing=False)
        # Tumor annotations in yellow contours
        tumor_contours = (anno_resized == 2)
        if tumor_contours.any():
            overlay[tumor_contours, 0:2] = [1.0, 1.0]  # Yellow for tumor
    
    return np.clip(overlay, 0, 1)

def test(args, bags_list, milnet):
    milnet.eval()
    num_bags = len(bags_list)
    Tensor = torch.FloatTensor
    
    for i in range(0, num_bags):
        feats_list = []
        pos_list = []
        classes_list = []
        csv_file_path = glob.glob(os.path.join(bags_list[i], '*.jpg'))
        dataloader, bag_size = bag_dataset(args, csv_file_path)
        
        # Get slide name
        slide_name = bags_list[i].split(os.sep)[-1]
        print(f"\nProcessing {slide_name}...")
        
        with torch.no_grad():
            for iteration, batch in enumerate(dataloader):
                patches = batch['input'].float().cuda()
                patch_pos = batch['position']
                feats, classes = milnet.i_classifier(patches)
                feats = feats.cpu().numpy()
                classes = classes.cpu().numpy()
                feats_list.extend(feats)
                pos_list.extend(patch_pos)
                classes_list.extend(classes)
            
            pos_arr = np.vstack(pos_list)
            feats_arr = np.vstack(feats_list)
            classes_arr = np.vstack(classes_list)
            bag_feats = torch.from_numpy(feats_arr).cuda()
            ins_classes = torch.from_numpy(classes_arr).cuda()
            bag_prediction, A, _ = milnet.b_classifier(bag_feats, ins_classes)
            
            # Handle different output formats
            if args.num_classes == 1:
                # Binary classification with threshold (sigmoid output)
                bag_prediction = torch.sigmoid(bag_prediction).squeeze().cpu().numpy()
                is_tumor = bag_prediction >= args.thres_tumor
                confidence = bag_prediction
            else:
                # Multi-class classification (softmax output)
                bag_prediction = F.softmax(bag_prediction, dim=1).squeeze().cpu().numpy()
                if bag_prediction.ndim == 0:  # Single value
                    bag_prediction = np.array([1 - bag_prediction, bag_prediction])
                is_tumor = bag_prediction[1] > bag_prediction[0]  # Class 1 is tumor
                confidence = bag_prediction[1]  # Probability of tumor class
            
            # Determine prediction
            color = [0, 0, 0]
            if is_tumor:
                print(f'{slide_name} is detected as MALIGNANT (confidence: {confidence:.4f})')
                color = [1, 0, 0]
            else:
                print(f'{slide_name} is detected as BENIGN (confidence: {confidence:.4f})')
            
            # Create attention heatmap
            # Handle attention dimensions for num_classes=1 or 2
            attentions = A.cpu().numpy()
            
            # If num_classes=2, A shape is (num_patches, 2), we need to extract tumor class attention
            if attentions.ndim > 1 and attentions.shape[1] > 1:
                # Use the tumor class (index 1) attention weights
                attentions = attentions[:, 1]
            else:
                # For num_classes=1, attentions is already (num_patches,) or (num_patches, 1)
                attentions = attentions.squeeze()
            
            attentions = exposure.rescale_intensity(attentions, out_range=(0, 1))
            
            # Create color map
            color_map = np.zeros((np.amax(pos_arr, 0)[0]+1, np.amax(pos_arr, 0)[1]+1, 3))
            for k, pos in enumerate(pos_arr):
                tile_color = np.asarray(color) * attentions[k]
                color_map[pos[0], pos[1]] = tile_color
            
            # Resize attention map
            color_map_resized = transform.resize(color_map, 
                                                (color_map.shape[0]*32, color_map.shape[1]*32), 
                                                order=0)
            
            # Save original attention heatmap
            output_dir = os.path.join('test-c16', 'output')
            os.makedirs(output_dir, exist_ok=True)
            io.imsave(os.path.join(output_dir, f'{slide_name}_attention.png'), 
                     img_as_ubyte(color_map_resized), check_contrast=False)
            
            # Load WSI thumbnail
            wsi_path = os.path.join('test-c16', 'input', f'{slide_name}.tif')
            wsi_thumb = load_wsi_thumbnail(wsi_path, target_size=args.thumbnail_size)
            
            if wsi_thumb is not None:
                # Save WSI thumbnail
                io.imsave(os.path.join(output_dir, f'{slide_name}_thumbnail.png'), 
                         wsi_thumb, check_contrast=False)
                
                # Load mask
                mask_path = os.path.join('/projects/standard/lin01231/public/datasets/camelyon16/masks',
                                        f'{slide_name}_mask.tif')
                mask = load_mask(mask_path, target_size=(wsi_thumb.shape[0], wsi_thumb.shape[1]))
                
                # Load annotation
                anno_path = os.path.join('/projects/standard/lin01231/public/datasets/camelyon16/annotations',
                                        f'{slide_name}.xml')
                anno_mask = None
                if os.path.exists(anno_path):
                    slide = openslide.OpenSlide(wsi_path)
                    wsi_dims = slide.dimensions
                    slide.close()
                    anno_mask = parse_asap_annotation(anno_path, wsi_dims, 
                                                     (wsi_thumb.shape[0], wsi_thumb.shape[1]))
                
                # Create overlays with adjusted transparency
                # 1. WSI + attention (deeper red, alpha=0.5)
                overlay_attention = create_overlay(wsi_thumb, heatmap=color_map_resized, 
                                                  heatmap_alpha=0.5)
                io.imsave(os.path.join(output_dir, f'{slide_name}_overlay_attention.png'),
                         img_as_ubyte(overlay_attention), check_contrast=False)
                
                # 2. WSI + mask (only tumor in red with high transparency, normal keeps original color)
                if mask is not None:
                    overlay_mask = create_overlay(wsi_thumb, mask=mask, mask_alpha=0.15)
                    io.imsave(os.path.join(output_dir, f'{slide_name}_overlay_mask.png'),
                             img_as_ubyte(overlay_mask), check_contrast=False)
                
                # 3. WSI + attention + mask (all-in-one)
                if mask is not None:
                    overlay_all = create_overlay(wsi_thumb, 
                                               heatmap=color_map_resized,
                                               mask=mask,
                                               anno_mask=anno_mask,
                                               heatmap_alpha=0.4,
                                               mask_alpha=0.12)
                    io.imsave(os.path.join(output_dir, f'{slide_name}_overlay_all.png'),
                             img_as_ubyte(overlay_all), check_contrast=False)
                
                print(f"Saved visualizations for {slide_name}")
            else:
                print(f"Warning: Could not load WSI for {slide_name}")
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing workflow includes attention computing and color map production')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes [1 for threshold-based, 2 for softmax]')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size of feeding patches')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--feats_size', type=int, default=512)
    parser.add_argument('--thres_tumor', type=float, default=0.5282700061798096, help='Threshold for tumor detection (only used when num_classes=1)')
    parser.add_argument('--thumbnail_size', type=int, default=2048, help='Target size for WSI thumbnail')
    args = parser.parse_args()
    
    resnet = models.resnet18(weights=None, norm_layer=nn.InstanceNorm2d)
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = nn.Identity()
    i_classifier = mil.IClassifier(resnet, args.feats_size, output_class=args.num_classes).cuda()
    b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes).cuda()
    milnet = mil.MILNet(i_classifier, b_classifier).cuda()
    
    aggregator_weights = torch.load('/projects/standard/lin01231/song0760/dsmil-wsi/weights/20260106/causalmil_Camelyon16_fold_4_16.pth')
    milnet.load_state_dict(aggregator_weights, strict=False)
    
    state_dict_weights = torch.load(os.path.join('test-c16', 'weights', 'embedder.pth'))
    new_state_dict = OrderedDict()
    i_classifier = mil.IClassifier(resnet, args.feats_size, output_class=args.num_classes).cuda()
    for i in range(4):
        state_dict_weights.popitem()
    state_dict_init = i_classifier.state_dict()
    for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
        name = k_0
        new_state_dict[name] = v
    new_state_dict["fc.weight"] = aggregator_weights["i_classifier.fc.0.weight"]
    new_state_dict["fc.bias"] = aggregator_weights["i_classifier.fc.0.bias"]
    i_classifier.load_state_dict(new_state_dict, strict=True)
    milnet.i_classifier = i_classifier
    
    bags_list = glob.glob(os.path.join('test-c16', 'patches', '*'))
    os.makedirs(os.path.join('test-c16', 'output'), exist_ok=True)
    test(args, bags_list, milnet)
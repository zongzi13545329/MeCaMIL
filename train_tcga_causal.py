import torch
import torch.nn as nn
import sys, argparse, os, copy, glob, datetime
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import KFold
import json
from tqdm import tqdm


# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

def load_demographic_data(demographic_file):
    """Load demographic data from JSON file"""
    with open(demographic_file, 'r') as f:
        return {entry['submitter_id']: entry for entry in json.load(f)}


def extract_submitter_id(feats_csv_path):
    """Extract submitter_id from feature file path"""
    base = os.path.basename(feats_csv_path).split('.')[0]
    return '-'.join(base.split('-')[:3])


def get_bag_feats(csv_file_df, args, demographic_dict=None):
    """
    Get bag features and labels
    
    Returns:
        label: label array
        feats: feature array  
        feats_csv_path: feature file path
        demo_vector: demographic vector (8-dim, only when use_demo=True)
    """
    # Build feature file path
    if args.dataset == 'TCGA-lung-default':
        feats_csv_path = 'datasets/tcga-dataset/tcga_lung_data_feats/' + \
                         csv_file_df.iloc[0].split('/')[1] + '.csv'
    else:
        feats_csv_path = csv_file_df.iloc[0]
    
    # Read features
    df = pd.read_csv(feats_csv_path)
    feats = shuffle(df).reset_index(drop=True).to_numpy()
    
    # Extract labels
    label = np.zeros(args.num_classes)
    if args.num_classes == 1:
        label[0] = csv_file_df.iloc[1]
    else:
        if int(csv_file_df.iloc[1]) <= (len(label) - 1):
            label[int(csv_file_df.iloc[1])] = 1
    
    # Extract demographic data (only when use_demo=True)
    demo_vector = None
    if args.use_demo and demographic_dict:
        submitter_id = extract_submitter_id(feats_csv_path)
        demo_entry = demographic_dict.get(submitter_id, {})
        gender = demo_entry.get('gender_vector', [0, 0])
        race = demo_entry.get('race_vector', [0, 0, 0, 0, 0])
        age = demo_entry.get('age_normalized', 0.0)
        demo_vector = np.array(gender + race + [age], dtype=np.float32)
    
    return label, feats, feats_csv_path, demo_vector


def generate_pt_files(args, df, demographic_file=None):
    """
    Generate .pt files for training
    
    Data format:
    - Without demo: [feats(512), label(num_classes)]
    - With demo: [feats(512), demographic(8), label(num_classes)]
    """
    # Clean and create temporary directory
    temp_train_dir = "temp_train"
    if os.path.exists(temp_train_dir):
        import shutil
        shutil.rmtree(temp_train_dir, ignore_errors=True)
    os.makedirs(temp_train_dir, exist_ok=True)
    
    # Load demographic data if needed
    demographic_dict = None
    if args.use_demo and demographic_file:
        demographic_dict = load_demographic_data(demographic_file)

    print('Creating intermediate training files.')
    for i in tqdm(range(len(df))):
        label, feats, feats_csv_path, demo = get_bag_feats(df.iloc[i], args, demographic_dict)
        
        # Convert to tensors
        bag_label = torch.tensor(np.array([label]), dtype=torch.float32)
        bag_feats = torch.tensor(np.array(feats), dtype=torch.float32)
        repeated_label = bag_label.repeat(bag_feats.size(0), 1)
        
        if args.use_demo and demo is not None:
            # With demographic: [feats, demo, label]
            demo_tensor = torch.tensor(demo, dtype=torch.float32)
            demo_repeated = demo_tensor.unsqueeze(0).repeat(bag_feats.size(0), 1)
            stacked_data = torch.cat([bag_feats, demo_repeated, repeated_label], dim=1)
        else:
            # Without demographic: [feats, label]
            stacked_data = torch.cat([bag_feats, repeated_label], dim=1)
        
        # Save as .pt file
        pt_file_path = os.path.join(temp_train_dir, 
                                    os.path.splitext(feats_csv_path)[0].split(os.sep)[-1] + ".pt")
        torch.save(stacked_data, pt_file_path)


def dropout_patches(feats, p):
    """Random dropout patches"""
    num_rows = feats.size(0)
    num_rows_to_select = int(num_rows * p)
    random_indices = torch.randperm(num_rows)[:num_rows_to_select]
    selected_rows = feats[random_indices]
    return selected_rows


# ============================================================================
# Training and Evaluation
# ============================================================================

def extract_data(stacked_data, args):
    """Extract features, labels and demographics from stacked_data"""
    if args.use_demo:
        bag_feats = stacked_data[:, :args.feats_size].clone()
        demographic = stacked_data[0, args.feats_size:args.feats_size + 8].clone()
        bag_label = stacked_data[0, args.feats_size + 8:].clone().unsqueeze(0)
        return bag_feats, bag_label, demographic
    else:
        bag_label = stacked_data[0, args.feats_size:].clone().unsqueeze(0)
        bag_feats = stacked_data[:, :args.feats_size].clone()
        return bag_feats, bag_label, None


def compute_mil_loss(args, milnet, criterion, bag_feats, bag_label, u):
    # Forward pass
    if args.model == 'causalmil' and args.use_demo and u is not None:
        ins_prediction, bag_prediction, result, B = milnet(bag_feats, u)  
    else:
        ins_prediction, bag_prediction, result, B = milnet(bag_feats)
    
    max_prediction, _ = torch.max(ins_prediction, 0)
    
    # L_cls
    bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
    max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
    L_cls = 0.5 * bag_loss + 0.5 * max_loss
    
    # L_causal
    L_causal = torch.tensor(0.0, device=bag_feats.device)
    
    if args.model == 'causalmil' and args.use_demo and u is not None:
        # Term 1: ||Z - h_X||²
        causal_consistency = torch.tensor(0.0, device=bag_feats.device)
        if 'Z' in result and 'causal_info' in result:
            Z = result['Z']  # [1, 256]
            if 'node_representations' in result['causal_info']:
                nodes = result['causal_info']['node_representations']
                h_X = nodes[:, 0, :]  # X node
                causal_consistency = torch.norm(Z - h_X, p=2) ** 2
        
        # Term 2: L_demo
        L_demo = torch.tensor(0.0, device=bag_feats.device)
        if 'decoded_demographics' in result:
            decoded = result['decoded_demographics']
            target = u.unsqueeze(0) if u.dim() == 1 else u
            
            if decoded.size(0) != target.size(0):
                if decoded.size(0) > 1:
                    decoded = decoded.mean(dim=0, keepdim=True)
                else:
                    target = target.expand(decoded.size(0), -1)
            
            L_demo = nn.MSELoss()(decoded, target)  
        
        # Combine
        L_causal = causal_consistency + 0.1 * L_demo 
    
    # Total loss
    total_loss = L_cls + 0.1 * L_causal  
    
    return total_loss, ins_prediction, bag_prediction


def train(args, train_df, milnet, criterion, optimizer):
    """Train one epoch"""
    milnet.train()
    dirs = shuffle(train_df)
    total_loss = 0
    
    for i, item in enumerate(dirs):
        optimizer.zero_grad()
        stacked_data = torch.load(item, map_location='cuda:0')
        
        # Extract data
        bag_feats, bag_label, u = extract_data(stacked_data, args)
        
        # Dropout patches
        bag_feats = dropout_patches(bag_feats, 1 - args.dropout_patch)
        bag_feats = bag_feats.view(-1, args.feats_size)
        
        # Compute loss
        loss, _, _ = compute_mil_loss(args, milnet, criterion, bag_feats, bag_label, u)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        
        sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_df), loss.item()))
    
    return total_loss / len(train_df)


def test(args, test_df, milnet, criterion, thresholds=None):
    """Test function"""
    milnet.eval()
    total_loss = 0
    test_labels = []
    test_predictions = []
    
    with torch.no_grad():
        for i, item in enumerate(test_df):
            stacked_data = torch.load(item, map_location='cuda:0')
            
            # Extract data
            bag_feats, bag_label, u = extract_data(stacked_data, args)
            
            # Dropout patches
            bag_feats = dropout_patches(bag_feats, 1 - args.dropout_patch)
            bag_feats = bag_feats.view(-1, args.feats_size)
            
            # Compute loss
            loss, ins_prediction, bag_prediction = compute_mil_loss(args, milnet, criterion, 
                                                                     bag_feats, bag_label, u)
            max_prediction, _ = torch.max(ins_prediction, 0)
            
            total_loss = total_loss + loss.item()
            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_df), loss.item()))
            
            # Collect predictions
            test_labels.extend([bag_label.squeeze().cpu().numpy().astype(int)])
            if args.average:
                test_predictions.extend([(torch.sigmoid(max_prediction) + 
                                        torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
            else:
                test_predictions.extend([torch.sigmoid(bag_prediction).squeeze().cpu().numpy()])
    
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)
    
    # Compute AUC and optimal thresholds
    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, 
                                                        args.num_classes, pos_label=1)
    
    if thresholds:
        thresholds_optimal = thresholds
    
    # Binarize predictions
    if args.num_classes == 1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions >= thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions < thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
    else:
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i] >= thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i] < thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
    
    # Compute accuracy
    bag_score = 0
    for i in range(0, len(test_df)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score
    avg_score = bag_score / len(test_df)
    
    return total_loss / len(test_df), avg_score, auc_value, thresholds_optimal


def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    """Compute ROC curves and optimal thresholds"""
    thresholds_optimal = []
    aucs = []
    
    if len(predictions.shape) == 1:
        predictions = predictions[:, None]
    if labels.ndim == 1:
        labels = np.expand_dims(labels, axis=-1)
    
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        _, _, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        
        try:
            c_auc = roc_auc_score(label, prediction)
            print("ROC AUC score:", c_auc)
        except ValueError as e:
            if str(e) == "Only one class present in y_true. ROC AUC score is not defined in that case.":
                print("ROC AUC score is not defined when only one class is present. c_auc is set to 1.")
                c_auc = 1
            else:
                raise e

        aucs.append(c_auc)
        thresholds_optimal.append(threshold_optimal)
    
    return aucs, None, thresholds_optimal


def optimal_thresh(fpr, tpr, thresholds, p=0):
    """Compute optimal threshold"""
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


# ============================================================================
# Model Management
# ============================================================================

def apply_sparse_init(m):
    """Weight initialization"""
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def init_model(args, mil_module):
    """Initialize model (supports DSMIL/ABMIL/CausalMIL)"""
    # Instance classifier (shared by all models)
    i_classifier = mil_module.FCLayer(in_size=args.feats_size, 
                                      out_size=args.num_classes).cuda()
    
    # Bag classifier (choose based on model type)
    if args.model == 'causalmil' and args.use_demo:
        # CausalMIL with demographics
        b_classifier = mil_module.BClassifier(
            input_size=args.feats_size, 
            output_class=args.num_classes,
            u_dim=8,  # Fixed 8-dim demographic
            hidden_dim=128,
            dropout_v=args.dropout_node,
            nonlinear=True,
            passing_v=False,
            causal=True,
            convDepth=args.structural_depth,
            use_causal_graph=args.use_causal_graph
        ).cuda()
    elif args.model == 'abmil':
        # ABMIL with gated attention
        b_classifier = mil_module.BClassifier(
            input_size=args.feats_size,
            output_class=args.num_classes,
            dropout_v=args.dropout_node,
            nonlinear=True,
            passing_v=False,
            D=128
        ).cuda()
    else:
        # DSMIL standard configuration
        b_classifier = mil_module.BClassifier(
            input_size=args.feats_size,
            output_class=args.num_classes,
            dropout_v=args.dropout_node,
            nonlinear=args.non_linearity
        ).cuda()
    
    # Build complete MIL network
    milnet = mil_module.MILNet(i_classifier, b_classifier).cuda()
    milnet.apply(lambda m: apply_sparse_init(m))
    
    # Optimizer and scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, 
                                betas=(0.5, 0.9), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)
    
    return milnet, criterion, optimizer, scheduler


def save_model(args, fold, run, save_path, model, thresholds_optimal):
    """Save model"""
    save_name = os.path.join(save_path, f'{args.model}_{args.dataset}_fold_{fold}_{run+1}.pth')
    torch.save(model.state_dict(), save_name)
    print_save_message(args, save_name, thresholds_optimal)
    
    file_name = os.path.join(save_path, f'fold_{fold}_{run+1}.json')
    with open(file_name, 'w') as f:
        json.dump([float(x) for x in thresholds_optimal], f)


def print_save_message(args, save_name, thresholds_optimal):
    """Print save message"""
    if args.dataset.startswith('TCGA-lung'):
        print('Best model saved at: ' + save_name + ' Best thresholds: LUAD %.4f, LUSC %.4f' % 
              (thresholds_optimal[0], thresholds_optimal[1]))
    else:
        print('Best model saved at: ' + save_name)
        print('Best thresholds ===>>> ' + '|'.join('class-{}>>{}'.format(*k) 
              for k in enumerate(thresholds_optimal)))


def print_epoch_info(epoch, args, train_loss_bag, test_loss_bag, avg_score, aucs):
    """Print epoch information"""
    if args.dataset.startswith('TCGA-lung'):
        print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, auc_LUAD: %.4f, auc_LUSC: %.4f' % 
              (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score, aucs[0], aucs[1]))
    else:
        print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: ' % 
              (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + 
              '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs)))


def save_training_summary(save_path, fold_results, args):
    """Save training summary"""
    accuracies = np.array([i[0] for i in fold_results])
    aucs = np.array([i[1] for i in fold_results])
    
    def convert_to_serializable(value):
        """Convert numpy/list/float to JSON serializable format"""
        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, list):
            return [float(v) for v in value]
        else:
            return float(value)
    
    summary = {
        'experiment_info': {
            'dataset': args.dataset,
            'model': args.model,
            'use_demo': getattr(args, 'use_demo', False),
            'num_epochs': args.num_epochs,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
        },
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'statistics': {
            'mean_accuracy': float(accuracies.mean()),
            'std_accuracy': float(accuracies.std()),
            'mean_auc': convert_to_serializable(aucs.mean(axis=0) if aucs.ndim > 1 else aucs.mean()),
            'std_auc': convert_to_serializable(aucs.std(axis=0) if aucs.ndim > 1 else aucs.std()),
            'num_folds': len(fold_results)
        },
        'fold_results': [
            {
                'fold': i,
                'accuracy': float(acc),
                'auc': convert_to_serializable(auc)
            }
            for i, (acc, auc) in enumerate(fold_results)
        ]
    }
    
    summary_file = os.path.join(save_path, 'training_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[SAVE] Training summary: {summary_file}")


def train_one_fold(args, train_path, test_path, milnet, criterion, optimizer, 
                   scheduler, fold, run, save_path):
    """Train one fold"""
    fold_best_score = 0
    best_ac = 0
    best_auc = 0
    counter = 0

    for epoch in range(1, args.num_epochs + 1):
        counter += 1
        train_loss_bag = train(args, train_path, milnet, criterion, optimizer)
        test_loss_bag, avg_score, aucs, thresholds_optimal = test(args, test_path, milnet, criterion)
        
        print_epoch_info(epoch, args, train_loss_bag, test_loss_bag, avg_score, aucs)
        scheduler.step()

        current_score = (sum(aucs) + avg_score) / 2
        if current_score > fold_best_score:
            counter = 0
            fold_best_score = current_score
            best_ac = avg_score
            best_auc = aucs
            save_model(args, fold, run, save_path, milnet, thresholds_optimal)
        
        if counter > args.stop_epochs:
            break
    
    return best_ac, best_auc


def run_5_fold_cv(args, bags_path, save_path, mil_module):
    """Run 5-fold cross validation"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    run = len(glob.glob(os.path.join(save_path, '*.pth')))

    for fold, (train_index, test_index) in enumerate(kf.split(bags_path)):
        print(f"\n{'='*60}")
        print(f"Starting CV fold {fold}")
        print(f"{'='*60}")
        
        milnet, criterion, optimizer, scheduler = init_model(args, mil_module)
        train_path = [bags_path[i] for i in train_index]
        test_path = [bags_path[i] for i in test_index]
        
        best_ac, best_auc = train_one_fold(
            args, train_path, test_path, milnet, criterion, optimizer, 
            scheduler, fold, run, save_path)
        
        fold_results.append((best_ac, best_auc))
    
    return fold_results


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train MIL models on patch features')
    
    # Basic parameters
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=0.0001, type=float, help='Initial learning rate [0.0001]')
    parser.add_argument('--num_epochs', default=50, type=int, help='Number of total training epochs')
    parser.add_argument('--stop_epochs', default=10, type=int, help='Early stopping patience')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight decay [1e-3]')
    
    # Dataset parameters
    parser.add_argument('--dataset', default='TCGA-lung-default', type=str, 
                       help='Dataset name: TCGA-lung-default | Camelyon16')
    parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
    
    # Model parameters
    parser.add_argument('--model', default='abmil', type=str, help='MIL model [dsmil | abmil | causalmil]')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation')
    parser.add_argument('--average', type=bool, default=False, 
                       help='Average the score of max-pooling and bag aggregating')
    
    # CausalMIL specific parameters
    parser.add_argument('--use_demo', action='store_true', help='Use demographic information')
    parser.add_argument('--demographic_file', type=str, 
                       default="./datasets/tcga-dataset/processed_clinical_data.json",
                       help='Path to demographic JSON file')
    parser.add_argument('--structural_depth', default=1, type=int, help='Depth of causal graph layers')
    parser.add_argument('--use_causal_graph', action='store_true', help='Use causal graph structure')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"Model: {args.model.upper()}")
    print(f"Dataset: {args.dataset}")
    print(f"Use Demographics: {args.use_demo}")
    print(f"Evaluation: 5-fold cross validation")
    print(f"{'='*60}\n")

    # GPU setup
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)
    
    # Import model module
    if args.model == 'dsmil':
        import dsmil as mil
    elif args.model == 'abmil':
        import abmil as mil
    elif args.model == 'causalmil':
        import causalmil as mil
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Load dataset
    if args.dataset == 'TCGA-lung-default':
        bags_csv = 'datasets/tcga-dataset/TCGA.csv'
    else:
        bags_csv = os.path.join('datasets', args.dataset, args.dataset + '.csv')
    
    # Check demographic file
    if args.use_demo:
        if not hasattr(args, 'demographic_file') or args.demographic_file is None:
            print("Error: --demographic_file is required when using --use_demo")
            return
        if not os.path.exists(args.demographic_file):
            print(f"Error: Demographic file not found: {args.demographic_file}")
            return
    
    # Generate .pt files
    generate_pt_files(args, pd.read_csv(bags_csv), 
                     args.demographic_file if args.use_demo else None)
    
    # Prepare save path
    save_path = os.path.join('weights', datetime.date.today().strftime("%Y%m%d"))
    os.makedirs(save_path, exist_ok=True)
    
    bags_path = glob.glob('temp_train/*.pt')
    print(f"[INFO] Found {len(bags_path)} training bags\n")
    
    # Run 5-fold cross validation
    fold_results = run_5_fold_cv(args, bags_path, save_path, mil)
    
    # Print final results
    mean_ac = np.mean(np.array([i[0] for i in fold_results]))
    mean_auc = np.mean(np.array([i[1] for i in fold_results]), axis=0)
    
    print(f"\n{'='*60}")
    print(f"Final Results: {args.model.upper()} on {args.dataset}")
    print(f"{'='*60}")
    print(f"Mean Accuracy: {mean_ac:.4f}")
    for i, mean_score in enumerate(mean_auc):
        print(f"Class {i}: Mean AUC = {mean_score:.4f}")
    print(f"{'='*60}\n")
    
    # Save training summary
    save_training_summary(save_path, fold_results, args)


if __name__ == '__main__':
    main()
import torch
from models.resnet_simclr import ResNetSimCLR
import torch.nn.functional as F
from loss.nt_xent import NTXentLoss
import os
import shutil
import sys
import logging

apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp
    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False

import numpy as np

torch.manual_seed(0)

def _save_config_file(model_checkpoints_folder, dataset_name):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, f'config_{dataset_name}.yaml'))

class SimCLR(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        self.dataset = dataset
        self.dataset_name = self.config.get('datasets', 'unknown')
        
        # 配置日志
        os.makedirs('logs', exist_ok=True)
        log_filename = os.path.join('logs', f'training_{self.dataset_name}.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_filename)
            ]
        )

        self.nt_xent_criterion = NTXentLoss(self.device, config['batch_size'], **config['loss'])

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info("Running on: %s", device)
        return device

    def _step(self, model, xis, xjs, n_iter):
        ris, zis = model(xis)
        rjs, zjs = model(xjs)
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)
        loss = self.nt_xent_criterion(zis, zjs)
        return loss

    def train(self):
        train_loader, valid_loader = self.dataset.get_data_loaders()

        model = ResNetSimCLR(**self.config["model"])
        if self.config['n_gpu'] > 1:
            device_n = len(eval(self.config['gpu_ids']))
            model = torch.nn.DataParallel(model, device_ids=range(device_n))
        model = self._load_pre_trained_weights(model)
        model = model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=eval(self.config['weight_decay']))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['epochs'], eta_min=0, last_epoch=-1)

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(model, optimizer, opt_level='O2', keep_batchnorm_fp32=True)

        model_checkpoints_folder = os.path.join(self.config['log_dir'], f'checkpoints_{self.dataset_name}')
        _save_config_file(model_checkpoints_folder, self.dataset_name)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.config['epochs']):
            for (xis, xjs) in train_loader:
                optimizer.zero_grad()
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)
                loss = self._step(model, xis, xjs, n_iter)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    logging.info("[%s] Iteration %d, Train Loss: %.4f", self.dataset_name, n_iter, loss.item())

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter += 1

            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader)
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    model_path = os.path.join(model_checkpoints_folder, f'model_{self.dataset_name}.pth')
                    torch.save(model.state_dict(), model_path)
                    logging.info('[%s] Model saved at epoch %d', self.dataset_name, epoch_counter)

                logging.info("[%s] Epoch %d, Validation Loss: %.4f", self.dataset_name, epoch_counter, valid_loss)

            if epoch_counter >= 10:
                scheduler.step()
            logging.info("[%s] Epoch %d, Cosine LR Decay: %.6f", self.dataset_name, epoch_counter, scheduler.get_lr()[0])

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./runs', self.config['fine_tune_from'], f'checkpoints_{self.dataset_name}')
            model_path = os.path.join(checkpoints_folder, f'model_{self.dataset_name}.pth')
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict)
            logging.info("[%s] Loaded pre-trained model with success.", self.dataset_name)
        except FileNotFoundError:
            logging.info("[%s] Pre-trained weights not found. Training from scratch.", self.dataset_name)
        return model

    def _validate(self, model, valid_loader):
        with torch.no_grad():
            model.eval()
            valid_loss = 0.0
            counter = 0
            for (xis, xjs) in valid_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)
                loss = self._step(model, xis, xjs, counter)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        model.train()
        return valid_loss
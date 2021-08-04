import argparse
import csv
import numpy as np
import pickle
import torch
import time
import torch.nn as nn
import torch.nn.functional as F

from box import Box
from collections import OrderedDict
from pathlib import Path

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import utils as vutils
from tqdm import tqdm, trange

#### Need to change after real implementation 
import lpips
from utils import crop_img_by_part, scale, revert_scale
from models import weights_init, copy_model_params, load_params, Discriminator, Generator
from torchvision import transforms
from diffaug import DiffAugment
from metrics import calc_fid, extract_features
from inception import fid_inception_v3

DEFAULT_NUM_EPOCHS = 20
lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config_path', default='./config.yaml', type=Path, help='config file path')
    args = parser.parse_args()
    return vars(args)

class Trainer:

    def __init__(self, config):
        print(f"[*** Starting Training For {config.model.name} ***]")
        print(f"[--- Experiment Name: {config.train.exp_name} ---]")
        self.config = config # input config dict from yaml
        self.exp_name = config.train.exp_name # Experiment Name
        self.model_name = config.model.name # Set model name

        # Path to image data folders
        self.train_path = Path(config.data.path_train) 
        self.val_path = Path(config.data.path_val)
        self.test_path = Path(config.data.path_test)

        # Path to pretrained model / optimizer
        self.pretrain_netD_path = Path(config.data.pretrain_netD_path) # Path to pretrained model
        self.pretrain_netG_path = Path(config.data.pretrain_netG_path) # Path to pretrained model
        self.pretrain_optD_path = Path(config.data.pretrain_optD_path) # Path to pretrained optimizer
        self.pretrain_optG_path = Path(config.data.pretrain_optG_path) # Path to pretrained optimizer
        self.packaged = config.model.packaged # True if it the saved model is packaged with the format as the following
                                             # exp_name-epoch-number.ckpt
                                             #      |- model_state_dict
                                             #      |- optim_state_dict
                                             #      |- train_stats
                                             #      |- val_stats
        
        # Set checkpoint directory and log file
        self.checkpoint_dir = Path(config.data.checkpoint_dir) / config.train.exp_name # Path to checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True) # Generate checkpoint directory
        self.log_file = self.checkpoint_dir / "log.csv" # Generate log file for current experiment
        with open(self.log_file, 'w') as file:
            pass

        # Set training parameters
        self.num_epochs = getattr(config.train, "num_epochs", DEFAULT_NUM_EPOCHS) # Set Total number of epochs
        self.save_iters = config.train.save_iters
        self.batch_size = config.train.batch_size # Set batch size
        self.num_workers = config.train.num_workers # Set num_workers
        self.device = config.model.device # Set device
        self.criterion = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True) # Different
        self.policy = config.train.policy

        # Set Model Parameters
        self.ndf = config.model.ndf
        self.ngf = config.model.ngf
        self.nz = config.model.nz
        self.im_size = config.model.im_size
        self.noise = config.model.use_noise


        print(f"[*] Building data loader...")
        self.build_dataset()
        
        print(f"[*] Building model ...")
        self.build_model()

        print(f"[*] Building optimizer ...")
        self.build_optimizer()

        # Define Training Metrics and Validation Metrics       
        self.train_metrics = ["Loss_D", "Loss_G"]
        self.valid_metrics = ["FID"]
        self.all_metrics = self.train_metrics + self.valid_metrics
        self.train_stats = OrderedDict([(metric, 0) for metric in self.train_metrics]) 
        self.val_stats = OrderedDict([(metric, 0) for metric in self.valid_metrics]) 
    
    ##############################################################################
    ##               LOADING AND SAVING DATA / MODEL / OPTIMIZER                ##
    ##############################################################################

    def build_dataset(self):        
        transform_list = [transforms.Resize((int(self.im_size), int(self.im_size))),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
                        ]
        self.transform = transforms.Compose(transform_list)
    
        train_dataset, val_dataset, test_dataset = ImageFolder(root = self.train_path, transform = self.transform), \
                                                    ImageFolder(root = self.val_path, transform = self.transform), \
                                                    ImageFolder(root = self.test_path, transform = self.transform)

        self.train_data_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.val_data_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_data_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def build_model(self):
        self.netD = Discriminator(config = self.config.model.netD, ndf=self.ndf, im_size=self.im_size)
        self.netG = Generator(config = self.config.model.netG, ngf=self.ngf, nz=self.nz, im_size=self.im_size, noise=self.noise)
        self.netD.apply(weights_init)
        self.netG.apply(weights_init)
        self.netD.to(self.device)
        self.netG.to(self.device)

        # Load Pretrained Model if specified
        if getattr(self.config.model, "pretrained_D", False):
            print(f">>> Loading pre-trained model from {self.pretrain_netD_path}...")
            netD_path = self.pretrain_netD_path
            if not netD_path.exists():
                print("[*] ERROR: Pre-trained model does not exist!")
                exit()
            if self.packaged:
                self.netD.load_state_dict(torch.load(netD_path)['netD_state_dict'])
            else:
                self.netD.load_state_dict(torch.load(netD_path))
        
        if getattr(self.config.model, "pretrained_G", False):
            print(f">>> Loading pre-trained model from {self.pretrain_netG_path}...")
            netG_path = self.pretrain_netG_path
            if not netG_path.exists():
                print("[*] ERROR: Pre-trained model does not exist!")
                exit()
            if self.packaged:
                self.netG.load_state_dict(torch.load(netG_path)['netG_state_dict'])
                self.avg_G = torch.load(netG_path)['avg_g_params']
            else:
                self.netG.load_state_dict(torch.load(netG_path))
                

    def build_optimizer(self):
        self.optD = getattr(torch.optim, self.config.train.optD)(
                params=self.netD.parameters(),
                **self.config.train.optD_params)

        self.optG = getattr(torch.optim, self.config.train.optG)(
                params=self.netG.parameters(),
                **self.config.train.optG_params)

        # Load Pretrained Optimizer if specified
        if getattr(self.config.train, "load_optD", False):
            print(f">>> Loading pre-trained optD from {self.pretrain_optD_path}...")
            optimizer_path = self.pretrain_optD_path
            if not optimizer_path.exists():
                print("[*] ERROR: Pre-trained optimizer does not exist!")
                exit()
            if self.packaged:
                self.optD.load_state_dict(torch.load(optimizer_path)['optD_state_dict'])
            else:
                self.optD.load_state_dict(torch.load(optimizer_path))

        if getattr(self.config.train, "load_optG", False):
            print(f">>> Loading pre-trained optG from {self.pretrain_optG_path}...")
            optimizer_path = self.pretrain_optG_path
            if not optimizer_path.exists():
                print("[*] ERROR: Pre-trained optimizer does not exist!")
                exit()
            if self.packaged:
                self.optG.load_state_dict(torch.load(optimizer_path)['optG_state_dict'])
            else:
                self.optG.load_state_dict(torch.load(optimizer_path))
        

    def save_model(self, checkpoint_path):
        tqdm.write(f"[-] Saving model to {checkpoint_path}")
        torch.save({
            'netD_state_dict': self.netD.state_dict(),
            'netG_state_dict': self.netG.state_dict(),
            'avg_g_params': self.avg_G,
            'optD_state_dict': self.optD.state_dict(),
            'optG_state_dict': self.optG.state_dict(),
            'train_stats': self.train_stats,
            'val_stats': self.val_stats
        }, checkpoint_path)


    ##############################################################################
    ##                               TRAIN LOOP                                 ##
    ##############################################################################

    def one_epoch(self):
        self.netD.train()
        self.netD.to(self.device)
        self.netG.train()
        self.netG.to(self.device)

        bar = tqdm(self.train_data_loader, leave=False)

        # Initialize train metric dictionary
        self.train_stats = OrderedDict([(metric, 0) for metric in self.train_metrics]) 

        for idx, data in enumerate(bar):
            x, y = data
            # x = x.cuda(non_blocking=True)
            x = x.to(self.device)

            real_x = DiffAugment(x, policy=self.policy)

            noise = torch.Tensor(x.size(0), self.nz).normal_(0,1).to(self.device)
            fake_x = self.netG(noise)
            fake_x = [DiffAugment(x, policy=self.policy) for x in fake_x]

            self.optD.zero_grad()
            self.optG.zero_grad()

            self.netD.zero_grad()
            err_d, rec_all, rec_small, rec_part = self.train_d(real_x, label="real")
            self.train_d([img.detach() for img in fake_x], label="fake")
            self.optD.step()

            self.netG.zero_grad()
            pred_g = self.netD(fake_x, "fake")
            err_g = -pred_g.mean()
            err_g.backward()
            self.optG.step()
        
            for param, avg_param in zip(self.netG.parameters(), self.avg_G):
                avg_param.mul_(0.999).add_(0.001 * param.data)
            
            self.train_stats['Loss_D'] += err_d
            self.train_stats['Loss_G'] += -err_g
            bar.set_postfix_str(', '.join([
                f"{metric}: {val / (idx + 1):.2f}"
                for metric, val in self.train_stats.items()]))

        self.train_stats = {
            key: value / len(self.train_data_loader)
            for key, value in self.train_stats.items()}
    
    def train_d(self, data, label = "real"):
        if label == "real":
            pred, [rec_all, rec_small, rec_part], part = self.netD(data, label)
            err = F.relu(torch.rand_like(pred) * 0.2 + 0.8 - pred).mean() \
                    + self.criterion(rec_all, F.interpolate(data, rec_all.shape[2])).sum() \
                    + self.criterion(rec_small, F.interpolate(data, rec_small.shape[2])).sum() \
                    + self.criterion(rec_part, F.interpolate(crop_img_by_part(data, part), rec_part.shape[2])).sum()
            err.backward()
            return pred.mean().item(), rec_all, rec_small, rec_part
        else:
            pred = self.netD(data, label)
            err = F.relu(torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
            err.backward()
            return pred.mean().item()

    def train(self):
        print(f"[*] Start of training loop...")
        self.netD.train()
        self.netG.train()

        if not getattr(self.config.model, "pretrained_G", False):
            self.avg_G = copy_model_params(self.netG)

        if self.num_workers > 1:
            self.netD = nn.DataParallel(self.netD.cuda())
            self.netG = nn.DataParallel(self.netG.cuda())
        
        with open(self.log_file, 'a') as file:
            writer = csv.DictWriter(file, self.all_metrics)
            writer.writeheader()

        end_time = time.time()

        for epoch in trange(self.num_epochs):
            start_time = end_time
            self.one_epoch()
            # Train result is stored in self.train_stats after training one epoch
            tqdm.write(
                f"[>>>] Epoch {epoch + 1} [Train] " +
                ', '.join(f"{metric}: {value:.4f}"
                    for metric, value in self.train_stats.items()))
            

            if((epoch + 1) % self.save_iters == 0):
                self.validation(epoch)
                # Validation result is stored in self.val_stats after running_validation
                tqdm.write(
                    f"[>>>] Epoch {epoch + 1} [Val] " +
                    ', '.join(f"{metric}: {value:.4f}"
                        for metric, value in self.val_stats.items()))
                
                self.write_log_file()

                
                checkpoint_path = self.checkpoint_dir / f"{self.exp_name}-epoch-{epoch+1}.ckpt"
                img_ckpt_path = self.checkpoint_dir / f"{self.exp_name}-epoch-{epoch+1}.jpg"
                self.save_model(checkpoint_path)
                self.gen_result(img_ckpt_path)

            end_time = time.time()
            lr_d, lr_g = self.find_learning_rate()
            tqdm.write("[>>>] Time : {:.4f}s  ".format(end_time - start_time) + "LR - D,G : {:.4f} , {:.4f}".format(lr_d, lr_g))
    
    def gen_result(self, save_path):
        # Use average param to generate result
        backup_G = copy_model_params(self.netG)
        load_params(self.netG, self.avg_G)
        noise = torch.FloatTensor(8, self.nz).normal_(0,1).to(self.device)

        with torch.no_grad():
            vutils.save_image(revert_scale(self.netG(noise)[0]), save_path, nrow=4)
        
        load_params(self.netG, backup_G)


    def validation(self, epoch):
        self.netG.eval()
        self.netG.to(self.device)
        bar = tqdm(self.val_data_loader, leave=False)

        # Initialize validation metric dictionary
        self.val_stats = OrderedDict([(metric, 0) for metric in self.valid_metrics]) 
        validation_dir = self.checkpoint_dir / f"val_{epoch}" # Path to checkpoint directory
        validation_path = validation_dir / "img"
        validation_path.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for i, data in enumerate(bar):
                x = data[0]
                x = x.to(self.device)

                noise = torch.randn(self.batch_size, self.nz).to(self.device)
                g_imgs = self.netG(noise)[0] #Use img big for evaluation
                g_imgs = F.interpolate(g_imgs, self.im_size)

                for j, img in enumerate(g_imgs):
                    vutils.save_image(revert_scale(img), validation_path / f"{i*self.batch_size + j}.png")

        self.val_stats['FID'] = self.fid_score(self.val_path, validation_dir)


    def fid_score(self, folder1_path, folder2_path):
        inception = InceptionV3([3], normalize_input=False).eval().to(self.device)
        dataset1 = ImageFolder(folder1_path, self.transform)
        dataloader1 = DataLoader(dataset1, batch_size = self.batch_size, num_workers = self.num_workers)
        feature1 = extract_features(dataloader1, inception, self.device).numpy()
        real_mean = np.mean(feature1, 0)
        real_covariance = np.cov(feature1, rowvar=False)

        dataset2 = ImageFolder(folder2_path, self.transform)
        dataloader2 = DataLoader(dataset2, batch_size = self.batch_size, num_workers = self.num_workers)
        feature2 = extract_features(dataloader2, inception, self.device).numpy()
        sample_mean = np.mean(feature2, 0)
        sample_covariance = np.cov(feature2, rowvar=False)

        fid = calc_fid(sample_mean, sample_covariance, real_mean, real_covariance)
        return fid

    def find_learning_rate(self):
        lr_d, lr_g = 0, 0
        for g in self.optD.param_groups:
            lr_d = g['lr']
            break

        for g in self.optG.param_groups:
            lr_g = g['lr']
            break
        return lr_d, lr_g

    def write_log_file(self):
        with open(self.log_file, 'a') as file:
            writer = csv.DictWriter(file, self.all_metrics)
            stats = {**self.train_stats, **self.val_stats}
            # If you want to remove a metric use stats.pop('Metric you want to remove')
            stats = {key: f"{value:.4f}" for key, value in stats.items()}
            writer.writerow(stats)

def main(config_path):
    torch.cuda.empty_cache()    
    config = Box.from_yaml(open(config_path, 'r'))
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    args = parse_args()
    main(**args)

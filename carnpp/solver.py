import os
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torchsummaryX import summary
from dataset import generate_loader
from utils import *

class Solver():
    def __init__(self, model, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if config.scale > 0:
            self.net = model(scale=config.scale, group=config.group)
        else:
            self.net = model(multi_scale=True, group=config.group)
        
        self.net = self.net.to(self.device)
        self.loss_fn = nn.L1Loss()
        self.optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.net.parameters()), 
            config.lr
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optim, 
            config.decay, gamma=0.5,
        )
    
        self.train_loader = generate_loader(
            path=config.train_data, 
            scale=config.scale, train=True,
            size=config.patch_size, 
            batch_size=config.batch_size, num_workers=1,
            shuffle=True, drop_last=True
        )
        
        self.step = 0
        self.config = config

        summary(self.net, torch.zeros((1, 3, 180, 320)).to(self.device), scale=4)
        self.writer = SummaryWriter(log_dir=os.path.join("./runs", config.memo))
        os.makedirs(config.ckpt_dir, exist_ok=True)

    def fit(self):
        config = self.config
        net = nn.DataParallel(self.net, device_ids=range(config.num_gpu))
        
        while True:
            for inputs in self.train_loader:
                if config.scale > 0:
                    scale = config.scale
                    HR, LR = inputs[-1][0], inputs[-1][1]
                else:
                    # only use one of multi-scale data
                    # i know this is stupid but just temporary
                    scale = np.random.randint(2, 5)
                    HR, LR = inputs[scale-2][0], inputs[scale-2][1]
                
                HR = HR.to(self.device)
                LR = LR.to(self.device)
                
                SR = net(LR, scale)
                loss = self.loss_fn(SR, HR)
                
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), config.clip)
                self.optim.step()

                self.step += 1
                self.scheduler.step()

                if self.step % config.print_interval == 0:
                    if config.scale > 0:
                        psnr = self.evaluate("dataset/Urban100", config.scale)
                        self.writer.add_scalar("Urban100", psnr, self.step)
                    else:
                        psnr_x2 = self.evaluate("dataset/Urban100", 2)
                        psnr_x3 = self.evaluate("dataset/Urban100", 3)
                        psnr_x4 = self.evaluate("dataset/Urban100", 4)

                        self.writer.add_scalar("Urban100/x2", psnr_x2, self.step)
                        self.writer.add_scalar("Urban100/x3", psnr_x3, self.step)
                        self.writer.add_scalar("Urban100/x4", psnr_x4, self.step)
                    self.save(config.ckpt_dir)
            if self.step > config.max_steps: break

    def evaluate(self, test_data, scale=2):
        config = self.config
        
        test_loader = generate_loader(
            path=test_data, scale=scale,
            train=False,
            batch_size=1, num_workers=1, 
            shuffle=False, drop_last=False
        )
        
        mean_psnr = 0.0
        for step, inputs in enumerate(test_loader):
            HR = inputs[0].to(self.device)
            LR = inputs[1].to(self.device)
            with torch.no_grad():
                SR = self.net(LR, scale).detach()

            HR = HR.cpu().clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
            SR = SR.cpu().clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
            mean_psnr += psnr(HR, SR, scale) / len(test_loader)

        return mean_psnr

    def save(self, ckpt_dir):
        save_path = os.path.join(ckpt_dir, "{}.pth".format(self.step))
        torch.save(self.net.state_dict(), save_path)

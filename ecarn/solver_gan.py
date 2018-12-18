import os
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torchsummaryX import summary
import utils
from dataset import generate_loader

class Solver():
    def __init__(self, module, config):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        kwargs = {
            "num_channels": config.num_channels,
            "groups": config.group,
            "mobile": config.mobile
        }
        if config.scale > 0:
            kwargs["scale"] = config.scale
        else:
            kwargs["multi_scale"] = True

        self.G = module.Net(**kwargs).to(self.device)
        self.D = module.Discriminator().to(self.device)
        load(self.G, config.pretrained_ckpt)

        self.loss_l2 = nn.MSELoss()
        self.loss_gan = utils.GANLoss(self.device)
        self.loss_vgg = utils.VGGLoss().to(self.device)

        self.optim_G = torch.optim.Adam(self.G.parameters(), config.lr)
        self.optim_D = torch.optim.Adam(self.D.parameters(), config.lr)

        self.train_loader = generate_loader(
            path=config.train_data,
            scale=config.scale, train=True,
            size=config.patch_size,
            batch_size=config.batch_size, num_workers=1,
            shuffle=True, drop_last=True
        )

        self.step = 0
        self.config = config

        summary(
            self.G,
            torch.zeros((1, 3, 720//4, 1280//4)).to(self.device),
            scale=4
        )
        self.writer = SummaryWriter(log_dir=os.path.join("./runs", config.memo))
        os.makedirs(config.ckpt_dir, exist_ok=True)

    def fit(self):
        config = self.config

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

                # train the discriminator
                D_real = self.D(HR)
                if config.msd:
                    D_real_loss = self.loss_gan(D_real[0], is_real=True) + \
                                  self.loss_gan(D_real[1], is_real=True) + \
                                  self.loss_gan(D_real[2], is_real=True)
                else:
                    D_real_loss = self.loss_gan(D_real[2], is_real=True)

                SR = self.G(LR, scale)
                D_fake = self.D(SR)
                if config.msd:
                    D_fake_loss = self.loss_gan(D_fake[0], is_real=False) + \
                                  self.loss_gan(D_fake[1], is_real=False) + \
                                  self.loss_gan(D_fake[2], is_real=False)
                else:
                    D_fake_loss = self.loss_gan(D_fake[2], is_real=False)

                D_loss = D_real_loss + D_fake_loss
                self.optim_D.zero_grad()
                D_loss.backward()
                self.optim_D.step()

                # train the generator
                SR = self.G(LR, scale)
                D_fake = self.D(SR)
                if config.msd:
                    D_fake_loss = self.loss_gan(D_fake[0], is_real=True) + \
                                  self.loss_gan(D_fake[1], is_real=True) + \
                                  self.loss_gan(D_fake[2], is_real=True)
                else:
                    D_fake_loss = self.loss_gan(D_fake[2], is_real=True)
                D_vgg_loss = self.loss_vgg(SR, HR)

                G_loss = config.gamma_gan * D_fake_loss + \
                         config.gamma_vgg * D_vgg_loss

                self.optim_G.zero_grad()
                G_loss.backward()
                nn.utils.clip_grad_norm_(self.G.parameters(), config.clip)
                self.optim_G.step()

                self.step += 1

                if self.step % config.print_interval == 0:
                    if config.scale > 0:
                        rmse, lpips = self.evaluate("dataset/Set14", config.scale)
                        self.writer.add_scalar("RMSE", rmse, self.step)
                        self.writer.add_scalar("LPIPS", lpips, self.step)
                    else:
                        raise NotImplementedError
                    self.save(config.ckpt_dir)

            if self.step > config.max_steps:
                break

    def evaluate(self, test_data, scale):
        test_loader = generate_loader(
            path=test_data, scale=scale,
            train=False,
            batch_size=1, num_workers=1,
            shuffle=False, drop_last=False
        )

        HRs, SRs = list(), list()
        for _, inputs in enumerate(test_loader):
            HR = inputs[0].to(self.device)
            LR = inputs[1].to(self.device)
            with torch.no_grad():
                SR = self.G(LR, scale).detach()

            HR = HR.cpu().clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
            SR = SR.cpu().clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
            HRs.append(HR)
            SRs.append(SR)

        rmse = utils.rmse(HRs, SRs, scale)
        lpips = utils.LPIPS(HRs, SRs, scale)

        return rmse, lpips

    def save(self, ckpt_dir):
        save_path = os.path.join(ckpt_dir, "{}.pth".format(self.step))
        torch.save(self.G.state_dict(), save_path)

def load(module, path):
    state_dict = torch.load(path)
    module.load_state_dict(state_dict)
    print("Load pretrained model: {}".format(path))

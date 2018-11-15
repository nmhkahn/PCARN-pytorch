import torch
import torch.nn as nn
import torch.nn.functional as F
import model.ops as ops


class Net(nn.Module):
    def __init__(self, scale=4, multi_scale=True, group=1):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(1, 64, 5, 1, 2),
            nn.Tanh(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.Tanh(),
        )
            
        self.up2 = nn.Sequential(
            nn.Conv2d(32, 4, 3, 1, 1),
            nn.Tanh(),
            nn.PixelShuffle(2)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(32, 9, 3, 1, 1),
            nn.Tanh(),
            nn.PixelShuffle(3)
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(32, 1*16, 3, 1, 1),
            nn.Tanh(),
            nn.PixelShuffle(4)
        )

                
    def forward(self, x, scale):
        out = self.body(x[:,:1])

        if scale==2:
            out = self.up2(out)
        elif scale==3:
            out = self.up3(out)
        elif scale==4:
            out = self.up4(out)

        return out

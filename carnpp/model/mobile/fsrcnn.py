import sys
sys.path.append("../")
import torch
import torch.nn as nn
import torch.nn.functional as F
import model.ops as ops

class Net(nn.Module):
    def __init__(self, scale=4, multi_scale=True, group=1):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(1, 56, 5, 1, 2),
            nn.PReLU(),
            nn.Conv2d(56, 12, 1, 1, 0),
            nn.PReLU(),

            nn.Conv2d(12, 12, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(12, 12, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(12, 12, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(12, 12, 3, 1, 1),
            nn.PReLU(),

            nn.Conv2d(12, 56, 1, 1, 0),
            nn.PReLU()
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(56, 1, 9, 2, 4, output_padding=1)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(56, 1, 9, 3, 4, output_padding=2)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(56, 1, 9, 4, 4, output_padding=3)
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

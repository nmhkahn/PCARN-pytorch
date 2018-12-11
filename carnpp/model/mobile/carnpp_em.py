import sys
sys.path.append("../")
import torch
import torch.nn as nn
import model.ops as ops

class Block(nn.Module):
    def __init__(self, channel=64, groups=1):
        super().__init__()

        self.b1 = ops.EResidualBlock(channel, channel, groups=groups)

    def forward(self, x):
        b1 = self.b1(x)
        return b1


class Net(nn.Module):
    def __init__(
        self,
        scale=2, multi_scale=True,
        num_channels=64,
        groups=1
    ):
        super().__init__()

        num_channels, groups = 32, 4
        self.entry = nn.Conv2d(3, num_channels, 5, 1, 2)

        self.b1 = Block(num_channels, groups)
        self.c1 = nn.Conv2d(num_channels*2, num_channels, 1, 1, 0)

        self.up4 = nn.Sequential(
            nn.Conv2d(num_channels, 16, 3, 1, 1),
            nn.ReLU(),
            nn.PixelShuffle(4)
        )

    def forward(self, x, scale):
        x = self.entry(x)
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        out = self.up4(o1)
        return out

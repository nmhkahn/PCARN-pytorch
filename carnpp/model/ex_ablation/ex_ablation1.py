import torch
import torch.nn as nn
import model.ops as ops

class Block(nn.Module):
    def __init__(self, channel=64, group=1):
        super().__init__()

        self.b1 = ops.ResidualBlock(channel, channel)
        self.b2 = ops.ResidualBlock(channel, channel)
        self.b3 = ops.ResidualBlock(channel, channel)
        self.c4 = nn.Conv2d(channel, channel, 3, 1, 1)

    def forward(self, x):
        c0 = o0 = x

        o1 = self.b1(o0)
        o2 = self.b2(o1)
        o3 = self.b3(o2)

        out = self.c4(o3)
        return out
        

class Net(nn.Module):
    def __init__(self, scale=2, multi_scale=True, group=1):
        super().__init__()

        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        self.entry = nn.Conv2d(3, 64, 3, 1, 1)

        self.b1 = Block(64)
        self.b2 = Block(64)
        self.b3 = Block(64)
        self.c4 = nn.Conv2d(64, 64, 3, 1, 1)

        self.upsample = ops.UpsampleBlock(
            64, 
            scale=scale, multi_scale=multi_scale,
            group=group
        )
        self.exit = nn.Conv2d(64, 3, 3, 1, 1)
                
    def forward(self, x, scale):
        x = self.sub_mean(x)
        x = self.entry(x)
        c0 = o0 = x

        o1 = self.b1(o0)
        o2 = self.b2(o1)
        o3 = self.b3(o2)
        out = self.c4(o3)
        out = self.upsample(out, scale=scale)

        out = self.exit(out)
        out = self.add_mean(out)
        return out

import torch
import torch.nn as nn
import model.ops as ops

class Block(nn.Module):
    def __init__(self, channel=64, group=1):
        super().__init__()

        self.b1 = ops.ResidualBlock(channel, channel)
        self.b2 = ops.ResidualBlock(channel, channel)
        self.b3 = ops.ResidualBlock(channel, channel)
        self.c1 = nn.Conv2d(channel*2, channel, 1, 1, 0)
        self.c2 = nn.Conv2d(channel*3, channel, 1, 1, 0)
        self.c3 = nn.Conv2d(channel*4, channel, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3
        

class Net(nn.Module):
    def __init__(self, scale=2, multi_scale=True, group=1):
        super().__init__()

        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        self.entry = nn.Conv2d(3, 64, 3, 1, 1)

        self.b1 = Block(64)
        self.b2 = Block(64)
        self.b3 = Block(64)
        self.c1 = nn.Conv2d(64*2, 64, 1, 1, 0)
        self.c2 = nn.Conv2d(64*3, 64, 1, 1, 0)
        self.c3 = nn.Conv2d(64*4, 64, 1, 1, 0)

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

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)
        out = self.upsample(o3, scale=scale)

        out = self.exit(out)
        out = self.add_mean(out)
        return out

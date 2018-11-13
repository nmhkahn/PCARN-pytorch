import torch
import torch.nn as nn
import torch.nn.functional as F
import model.ops as ops

class EResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, group=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 
            3, 1, 1, 
            groups=group
        )
        self.pw = nn.Conv2d(out_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = F.relu(self.pw(out)+x, inplace=True)
        return out


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, group=1):
        super().__init__()

        self.b1 = EResidualBlock(16, 16, group=group)
        self.b2 = EResidualBlock(16, 16, group=group)
        self.c2 = ops.BasicBlock(16*2, 16, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        o1 = c1 = b1 = self.b1(o0)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        return o2 + x
        

class Net(nn.Module):
    def __init__(self, scale=2, multi_scale=True, group=1):
        super().__init__()

        group = 4

        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)

        self.entry = nn.Conv2d(3, 16, 3, 1, 1)
        self.b1 = Block(16, 16, group=group)
        self.b2 = Block(16, 16, group=group)
        self.c2 = ops.BasicBlock(16*2, 16, 1, 1, 0)
        
        self.up2 = nn.Sequential(
            nn.Conv2d(16, 3*4, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(16, 3*9, 3, 1, 1),
            nn.PixelShuffle(3)
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(16, 3*16, 3, 1, 1),
            nn.PixelShuffle(4)
        )
                
    def forward(self, x, scale):
        x = self.sub_mean(x)
        x = self.entry(x)
        c0 = o0 = x

        o1 = c1 = b1 = self.b1(o0)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        out = o2 + x
        
        if scale==2:
            out = self.up2(out)
        elif scale==3:
            out = self.up3(out)
        elif scale==4:
            out = self.up4(out)
        
        out = self.add_mean(out)
        return out

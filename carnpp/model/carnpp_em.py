import torch
import torch.nn as nn
import torch.nn.functional as F
import model.ops as ops

class EResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, group=1):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 
                3, 1, 1, 
                groups=group
            ),
            nn.ReLU(inplace=True)
        )
        self.pw = nn.Conv2d(out_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.pw(out)+x, inplace=True)
        return out


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, group=1):
        super().__init__()

        self.b1 = EResidualBlock(12, 12, group=group)
        self.b2 = EResidualBlock(12, 12, group=group)
        self.c1 = nn.Conv2d(12*2, 12, 1, 1, 0)
        self.c2 = nn.Conv2d(12*3, 12, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
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

        self.entry = nn.Sequential(
            nn.Conv2d(3, 56, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(56, 12, 1, 1, 0),
            nn.ReLU(inplace=True)
        )
        self.b1 = Block(12, 12, group=group)
        self.b2 = Block(12, 12, group=group)
        self.c1 = nn.Conv2d(12*2, 12, 1, 1, 0)
        self.c2 = nn.Conv2d(12*3, 12, 1, 1, 0)        
        
        self.up2 = nn.Sequential(
            nn.Conv2d(12, 56, 1, 1, 0), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(56, 3, 9, 2, 4, output_padding=1)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(12, 56, 1, 1, 0), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(56, 3, 9, 3, 4, output_padding=2)

        )
        self.up4 = nn.Sequential(
            nn.Conv2d(12, 56, 1, 1, 0), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(56, 3, 9, 4, 4, output_padding=3)
        )
                
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
        
        out = o2 + x
        
        if scale==2:
            out = self.up2(out)
        elif scale==3:
            out = self.up3(out)
        elif scale==4:
            out = self.up4(out)
        
        out = self.add_mean(out)
        return out

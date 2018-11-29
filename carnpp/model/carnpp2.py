import math
import torch
import torch.nn as nn
import model.ops as ops
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mobile=False, groups=1):
        super().__init__()
        
        if mobile:
            self.conv = nn.Conv2d(
                in_channels, in_channels, 
                3, 1, 1, 
                groups=groups
            )
            self.conv_gc1 = nn.Conv2d(
                in_channels, in_channels,
                1, 1, 0,
                groups=groups
            )
            self.conv_gc2 = nn.Conv2d(
                in_channels, out_channels,
                1, 1, 0,
                groups=groups
            )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

        self.mobile = mobile
        self.groups = groups
    
    def shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
		
        x = x.view(
            batchsize, groups, 
            channels_per_group, height, width
        )

        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x

    def forward(self, x):
        if self.mobile:
            out = F.relu(self.conv_gc1(x), inplace=True)
            out = self.shuffle(out, self.groups)
            out = self.conv(out)
            out = F.relu(self.conv_gc2(out), inplace=True)
        else:
            out = F.relu(self.conv(x), inplace=True)
        return out


class SResidualBlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        
        self.conv = nn.Conv2d(n_channels, n_channels, 3, 1, 1)

    def forward(self, x):
        out = F.relu(self.conv(x)+x, inplace=True)
        return out
        

class SEResidualBlock(nn.Module):
    def __init__(self, n_channels, groups=1):
        super().__init__()

        self.conv = nn.Conv2d(
            n_channels, n_channels, 
            3, 1, 1, 
            groups=groups
        )
        self.conv_gc1 = nn.Conv2d(
            n_channels, n_channels,
            1, 1, 0,
            groups=groups
        )
        self.conv_gc2 = nn.Conv2d(
            n_channels, n_channels,
            1, 1, 0,
            groups=groups
        )
        self.groups = groups
        
    def shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
		
        x = x.view(
            batchsize, groups, 
            channels_per_group, height, width
        )

        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x

    def forward(self, x):
        out = F.relu(self.conv_gc1(x), inplace=True)
        out = self.shuffle(out, self.groups)
        out = self.conv(out)
        out = F.relu(self.conv_gc2(out)+x, inplace=True)
        return out
        

class ResidualBlock(nn.Module):
    def __init__(self, n_channels, mobile=False, groups=1):
        super().__init__()
        
        if mobile:
            self.block1 = SEResidualBlock(n_channels, groups)
            self.block2 = SEResidualBlock(n_channels, groups)
        else:
            self.block1 = SResidualBlock(n_channels)
            self.block2 = SResidualBlock(n_channels)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out += x
        return out


class UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, multi_scale, mobile=False, groups=1):
        super().__init__()

        if multi_scale:
            self.up2 = _UpsampleBlock(n_channels, scale=2, mobile=mobile, groups=groups)
            self.up3 = _UpsampleBlock(n_channels, scale=3, mobile=mobile, groups=groups)
            self.up4 = _UpsampleBlock(n_channels, scale=4, mobile=mobile, groups=groups)
        else:
            self.up =  _UpsampleBlock(n_channels, scale=scale, mobile=mobile, groups=groups)
        self.multi_scale = multi_scale

    def forward(self, x, scale):
        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            elif scale == 3:
                return self.up3(x)
            elif scale == 4:
                return self.up4(x)
        else:
            return self.up(x)


class _UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, mobile=False, groups=1):
        super().__init__()

        self.body = nn.ModuleList()
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                if mobile:
                    self.body.append(BasicBlock(n_channels, 4*n_channels, True,  groups))
                else:
                    self.body.append(BasicBlock(n_channels, 4*n_channels))
                self.body.append(nn.PixelShuffle(2))
        elif scale == 3:
            if mobile:
                self.body.append(BasicBlock(n_channels, 9*n_channels, True,  groups))
            else:
                self.body.append(BasicBlock(n_channels, 9*n_channels))
            self.body.append(nn.PixelShuffle(3))

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        return out


class Block(nn.Module):
    def __init__(self, channel=64, mobile=False, groups=1):
        super().__init__()
        
        if mobile:
            self.b1 = ResidualBlock(channel, mobile, groups=groups)
            self.b2 = self.b3 = self.b1
        else:
            self.b1 = ResidualBlock(channel)
            self.b2 = ResidualBlock(channel)
            self.b3 = ResidualBlock(channel)
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
    def __init__(
        self, 
        scale=2, multi_scale=True, 
        num_channels=64,
        mobile=False, groups=1
    ):
        super().__init__()

        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        self.entry = nn.Conv2d(3, num_channels, 3, 1, 1)

        self.b1 = Block(num_channels, mobile, groups)
        self.b2 = Block(num_channels, mobile, groups)
        self.b3 = Block(num_channels, mobile, groups)
        self.c1 = nn.Conv2d(num_channels*2, num_channels, 1, 1, 0)
        self.c2 = nn.Conv2d(num_channels*3, num_channels, 1, 1, 0)
        self.c3 = nn.Conv2d(num_channels*4, num_channels, 1, 1, 0)

        self.upsample = UpsampleBlock(
            num_channels,
            scale=scale, multi_scale=multi_scale,
            mobile=mobile, groups=groups
        )
        self.exit = nn.Conv2d(num_channels, 3, 3, 1, 1)
                
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
        out = self.upsample(o3+x, scale=scale)

        out = self.exit(out)
        out = self.add_mean(out)
        return out

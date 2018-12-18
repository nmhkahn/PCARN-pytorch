import torch
import torch.nn as nn
import model.ops as ops

class Block(nn.Module):
    def __init__(self, channel=64, mobile=False, groups=1):
        super().__init__()

        if mobile:
            self.b1 = ops.EResidualBlock(channel, channel, groups=groups)
            self.b2 = self.b3 = self.b1
        else:
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
    def __init__(
        self,
        scale=4, multi_scale=True,
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

        self.upsample = ops.UpsampleBlock(
            num_channels,
            scale=scale, multi_scale=multi_scale,
            groups=groups
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


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        def conv_bn_lrelu(in_channels, out_channels, ksize, stride, pad):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU()
            )

        self.entry = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.block1 = nn.Sequential(
            conv_bn_lrelu(64, 64, 3, 2, 1),
            conv_bn_lrelu(64, 128, 3, 1, 1),
            conv_bn_lrelu(128, 128, 3, 2, 1),
            conv_bn_lrelu(128, 256, 3, 1, 1)
        )
        self.block2 = nn.Sequential(
            conv_bn_lrelu(256, 256, 3, 2, 1),
            conv_bn_lrelu(256, 512, 3, 1, 1)
        )
        self.block3 = nn.Sequential(
            conv_bn_lrelu(512, 512, 3, 2, 1),
            conv_bn_lrelu(512, 512, 3, 1, 1)
        )

        self.exit1 = nn.Conv2d(256, 1, 3, 1, 1)
        self.exit2 = nn.Conv2d(512, 1, 3, 1, 1)
        self.exit3 = nn.Conv2d(512, 1, 3, 1, 1)

    def forward(self, x):
        out = x

        b0 = self.entry(out)
        b1 = self.block1(b0)
        b2 = self.block2(b1)
        b3 = self.block3(b2)

        o1 = self.exit1(b1)
        o2 = self.exit2(b2)
        o3 = self.exit3(b3)

        return torch.sigmoid(o1), torch.sigmoid(o2), torch.sigmoid(o3)
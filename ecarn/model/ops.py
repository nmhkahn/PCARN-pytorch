import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class MeanShift(nn.Module):
    def __init__(self, mean_rgb, sub):
        super(MeanShift, self).__init__()

        sign = -1 if sub else 1
        r = mean_rgb[0] * sign
        g = mean_rgb[1] * sign
        b = mean_rgb[2] * sign

        self.shifter = nn.Conv2d(3, 3, 1, 1, 0)
        self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.shifter.bias.data = torch.Tensor([r, g, b])

        # Freeze the mean shift layer
        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)
        return x


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels, out_channels,
        ksize=3, stride=1, pad=1,
    ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, ksize, stride, pad)

    def forward(self, x):
        out = F.relu(self.conv(x), inplace=True)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = F.relu(self.conv2(out)+x, inplace=True)
        return out


class EResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            3, 1, 1,
            groups=groups
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            3, 1, 1,
            groups=groups
        )
        self.pw = nn.Conv2d(out_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = F.relu(self.conv2(out), inplace=True)
        out = F.relu(self.pw(out)+x, inplace=True)
        return out


class UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, multi_scale, groups=1):
        super().__init__()

        if multi_scale:
            self.up2 = _UpsampleBlock(n_channels, scale=2, groups=groups)
            self.up3 = _UpsampleBlock(n_channels, scale=3, groups=groups)
            self.up4 = _UpsampleBlock(n_channels, scale=4, groups=groups)
        else:
            self.up = _UpsampleBlock(n_channels, scale=scale, groups=groups)
        self.multi_scale = multi_scale

    def forward(self, x, scale):
        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            if scale == 3:
                return self.up3(x)
            if scale == 4:
                return self.up4(x)
            raise NotImplementedError
        else:
            return self.up(x)


class _UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, groups=1):
        super().__init__()

        self.body = nn.ModuleList()
        if scale in [2, 4, 8]:
            for _ in range(int(math.log(scale, 2))):
                self.body.append(
                    nn.Conv2d(n_channels, 4*n_channels, 3, 1, 1, groups=groups)
                )
                self.body.append(nn.ReLU(inplace=True))
                self.body.append(nn.PixelShuffle(2))
        elif scale == 3:
            self.body.append(
                nn.Conv2d(n_channels, 9*n_channels, 3, 1, 1, groups=groups)
            )
            self.body.append(nn.ReLU(inplace=True))
            self.body.append(nn.PixelShuffle(3))

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        return out

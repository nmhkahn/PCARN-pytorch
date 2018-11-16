import torch
import torch.nn as nn
import model.ops as ops
import torch.nn.functional as F

class DenseConv(nn.Module):
    def __init__(
		self, 
		in_channels, grow_rate, 
        ksize=3, stride=1, pad=1
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, grow_rate, ksize, stride, pad)

    def forward(self, x):
        out = F.relu(self.conv(x), inplace=True)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(
        self, 
        init_grow_rate, grow_rate, 
        num_layers
    ):
        super().__init__()

        self.body = list()
        for n in range(num_layers):
            self.body += [DenseConv(init_grow_rate+n*grow_rate, grow_rate)]
        self.body = nn.Sequential(*self.body)
        
        # Local Feature Fusion
        self.fusion = nn.Conv2d(
            init_grow_rate+num_layers*grow_rate, 
            init_grow_rate, 
            1, 1, 0
        )

    def forward(self, x):
        return torch.cat((x, self.fusion(self.body(x)) + x), 1)


class Net(nn.Module):
    def __init__(self, scale=2, multi_scale=True, group=1):
        super().__init__()
        
        self.num_blocks = 3
        self.num_layers = 6
        self.grow_rate = 64 # 32

        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)

        self.entry = nn.Conv2d(3, 64, 3, 1, 1)

        self.blocks = nn.ModuleList()
        for n in range(self.num_blocks):
            self.blocks.append(
                RDB(64*2**n, self.grow_rate, self.num_layers)
            )

        # Global Feature Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(128+256+512, 64, 1, 1, 0),
            nn.Conv2d(64, 64, 3, 1, 1)
        )

        self.upsample = ops.UpsampleBlock(
            64, 
            scale=scale, multi_scale=multi_scale,
            group=1,
        )
        self.exit = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x, scale):
        x = self.sub_mean(x)
        out = x = self.entry(x)

        outs = list()
        for i in range(self.num_blocks):
            out = self.blocks[i](out)
            outs.append(out)

        out = self.fusion(torch.cat(outs, 1))
        out += x

        out = self.upsample(out, scale=scale)
        out = self.exit(out)
        out = self.add_mean(out)
        return out

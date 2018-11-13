import torch
import torch.nn as nn
import torch.nn.functional as F
import model.ops as ops


class Net(nn.Module):
    def __init__(self, scale=4, multi_scale=False, group=1):
        super().__init__()

        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        
        self.body = nn.Sequential(
            nn.Conv2d(1, 56, 5, 1, 0),
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
            nn.PReLU(),

            nn.ConvTranspose2d(56, 1, 9, 4, 3, output_padding=1)
        )

                
    def forward(self, x, scale):
        # x = self.sub_mean(x)
        out = self.body(x[:,:1])
        # out = self.add_mean(out)
        return out

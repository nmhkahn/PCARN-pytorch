import torch
import torch.nn as nn
import numpy as np
import skimage.color as color
import skimage.measure as measure
from PIL import Image
from PerceptualSimilarity.models import dist_model as dm

class GANLoss(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.real_label = torch.tensor(1.0).to(device)
        self.fake_label = torch.tensor(0.0).to(device)
        self.loss_fn = nn.BCELoss()

    def __call__(self, inp, is_real):
        label_tensor = self.real_label if is_real else self.fake_label
        label_tensor = label_tensor.expand_as(inp)
        return self.loss_fn(inp, label_tensor)


def save_image(tensor, filename):
    tensor = tensor.cpu()
    ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)


def psnr(ims1, ims2, scale):
    mean_psnr = 0.0
    for im1, im2 in zip(ims1, ims2):
        im1 = color.rgb2yuv(im1[scale:-scale, scale:-scale])[..., 0]
        im2 = color.rgb2yuv(im2[scale:-scale, scale:-scale])[..., 0]
        mean_psnr += measure.compare_psnr(im1, im2) / len(ims1)
    return mean_psnr


def ssim(ims1, ims2, scale):
    mean_ssim = 0.0
    for im1, im2 in zip(ims1, ims2):
        im1 = color.rgb2yuv(im1[scale:-scale, scale:-scale])[..., 0]
        im2 = color.rgb2yuv(im2[scale:-scale, scale:-scale])[..., 0]
        mean_ssim += measure.compare_ssim(im1, im2) / len(ims1)
    return mean_ssim


def rmse(ims1, ims2, scale):
    mean_rmse = 0.0
    for im1, im2 in zip(ims1, ims2):
        im1 = color.rgb2yuv(im1[scale:-scale, scale:-scale])[..., 0]
        im2 = color.rgb2yuv(im2[scale:-scale, scale:-scale])[..., 0]

        mean_rmse += np.sum(np.square(im2-im1)) / len(ims1)
    mean_rmse = np.sqrt(mean_rmse)
    return mean_rmse


def LPIPS(ims1, ims2, scale):
    model = dm.DistModel()
    model.initialize(model="net-lin", net="alex", use_gpu=True, version="0.1")

    mean_distance = 0.0
    for im1, im2 in zip(ims1, ims2):
        im1 = im1[scale:-scale, scale:-scale]
        im2 = im2[scale:-scale, scale:-scale]

        im1 = torch.from_numpy(im1).permute(2, 0, 1).unsqueeze(0)
        im2 = torch.from_numpy(im2).permute(2, 0, 1).unsqueeze(0)

        # LPIPS needs [-1, 1] range tensor
        im1 = im1*2-1
        im2 = im2*2-1

        mean_distance += model.forward(im1, im2)[0] / len(ims1)
    return mean_distance

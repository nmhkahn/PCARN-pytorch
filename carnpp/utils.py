import numpy as np
import torch
import skimage.color as color
import skimage.measure as measure
from PIL import Image

def save_image(tensor, filename):
    tensor = tensor.cpu()
    ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)


def psnr(im1, im2, scale):
    im1 = color.rgb2yuv(im1[scale:-scale, scale:-scale])[...,0]
    im2 = color.rgb2yuv(im2[scale:-scale, scale:-scale])[...,0]
    return measure.compare_psnr(im1, im2)


def ssim(im1, im2, scale):
    im1 = color.rgb2yuv(im1[scale:-scale, scale:-scale])[...,0]
    im2 = color.rgb2yuv(im2[scale:-scale, scale:-scale])[...,0]
    return measure.compare_ssim(im1, im2)

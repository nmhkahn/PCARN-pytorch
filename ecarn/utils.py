import numpy as np
import skimage.color as color
import skimage.measure as measure
from PIL import Image
from brisque import BRISQUE

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


def brisque(ims, scale):
    brisq = BRISQUE()
    mean_brisque = 0.0
    for im in ims:
        im = im[scale:-scale, scale:-scale]
        mean_brisque += brisq.get_score(im) / len(ims)
    return mean_brisque

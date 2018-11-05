import numpy as np
import skimage.color as color
import skimage.measure as measure

def psnr(im1, im2, scale):
    im1 = color.rgb2yuv(im1[scale:-scale, scale:-scale])
    im2 = color.rgb2yuv(im2[scale:-scale, scale:-scale])
    return measure.compare_psnr(im1, im2)

def ssim(im1, im2, scale):
    im1 = color.rgb2yuv(im1[scale:-scale, scale:-scale])
    im2 = color.rgb2yuv(im2[scale:-scale, scale:-scale])
    return measure.compare_ssim(im1, im2)

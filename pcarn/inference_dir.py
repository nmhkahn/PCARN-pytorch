import os
import glob
import argparse
import importlib
from collections import OrderedDict
from tqdm import tqdm
import skimage.io as io
import skimage.color as color
import torch
import utils
import torchvision as tv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="pcarn")
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--save_root", type=str, default="./sample")
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--data_root", type=str, default="./dataset/")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--num_channels", type=int, default=64)
    parser.add_argument("--mobile", action="store_true", default=False)

    return parser.parse_args()


def read_and_preprocess(path, transform):
    image = io.imread(path)
    if len(image.shape) == 2:
        image = color.gray2rgb(image)

    return transform(image).unsqueeze(0) # add batch-axis


def inference(net, config):
    os.makedirs(config.save_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    transform = tv.transforms.ToTensor()

    paths = sorted(glob.glob(os.path.join(config.data_root, "*.png")))

    for path in tqdm(paths):
        LR = read_and_preprocess(path, transform).to(device)

        with torch.no_grad():
            SR = net(LR, config.scale).detach()

        filename = path.split("/")[-1]
        SR_path = os.path.join(config.save_root, filename)
        utils.save_image(SR.squeeze(0), SR_path)

def main(config):
    model = importlib.import_module("model.{}".format(config.model)).Net

    kwargs = {
        "num_channels": config.num_channels,
        "groups": config.groups,
        "mobile": config.mobile,
        "scale": config.scale,
    }

    net = model(**kwargs)
    state_dict = torch.load(config.ckpt)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        # name = k[7:] # remove "module."
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

    inference(net, config)


if __name__ == "__main__":
    args = parse_args()
    main(args)

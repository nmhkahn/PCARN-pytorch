import os
import argparse
import importlib
from collections import OrderedDict
import torch
from torchsummaryX import summary
from dataset import generate_loader
import utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ecarn")
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--sample_dir", type=str, default="./sample")
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--data", type=str, default="./dataset/Urban100")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--num_channels", type=int, default=64)
    parser.add_argument("--mobile", action="store_true", default=False)
    parser.add_argument("--no_gt", action="store_true", default=False)

    return parser.parse_args()


def inference(net, config):
    loader = generate_loader(
        config.data,
        scale=config.scale,
        train=False,
        batch_size=1, num_workers=1,
        gt=not config.no_gt,
        shuffle=False, drop_last=False
    )
    SR_root = os.path.join(
        config.sample_dir,
        config.data.split("/")[-1],
        "x{}/SR".format(config.scale)
    )
    HR_root = os.path.join(
        config.sample_dir,
        config.data.split("/")[-1],
        "x{}/HR".format(config.scale)
    )
    os.makedirs(SR_root, exist_ok=True)
    os.makedirs(HR_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    summary(
        net,
        torch.zeros((1, 3, 720//4, 1280//4)).to(device),
        scale=4
    )

    for i, inputs in enumerate(loader):
        if not config.no_gt:
            HR = inputs[0].to(device)
            LR = inputs[1].to(device)
        else:
            LR = inputs[0].to(device)

        with torch.no_grad():
            SR = net(LR, config.scale).detach()

        SR_path = os.path.join(SR_root, "{:05d}.png".format(i+1))
        utils.save_image(SR.squeeze(0), SR_path)

        if not config.no_gt:
            HR_path = os.path.join(HR_root, "{:05d}.png".format(i+1))
            utils.save_image(HR.squeeze(0), HR_path)


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

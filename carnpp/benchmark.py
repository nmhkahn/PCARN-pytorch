import time
import argparse
import importlib
import numpy as np
import torch
from torchsummaryX import summary
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--scale", type=int)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--num_exp", type=int, default=100)

    return parser.parse_args()


def inference(net, config):
    device = torch.device("cpu")
    net = net.to(device)

    inp = torch.zeros(
        (1, 3, config.height//config.scale, config.width//config.scale)
    ).to(device)

    summary(net, inp, scale=4)

    with torch.no_grad():
        net(inp, config.scale)

    avg_time = 0.0
    for _ in range(config.num_exp):
        t1 = time.time()
        with torch.no_grad():
            net(inp, config.scale)
        t2 = time.time()

        avg_time += (t2-t1)/config.num_exp
    print("{} x{}: {:.3f}".format(config.model, config.scale, avg_time))

def main(config):
    model = importlib.import_module("model.{}".format(config.model)).Net
    net = model(scale=config.scale, groups=1)
    inference(net, config)


if __name__ == "__main__":
    args = parse_args()
    main(args)

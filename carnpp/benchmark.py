import time
import argparse
import importlib
import numpy as np
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--scale", type=int)

    return parser.parse_args()


def inference(net, config):
    device = torch.device("cpu")
    net = net.to(device)
        
    inp = torch.zeros((1, 3, 720//config.scale, 1280//config.scale)).to(device)
    with torch.no_grad():
       SR = net(inp, config.scale)

    avg_time = 0.0
    for i in range(10):
        t1 = time.time()
        with torch.no_grad():
            SR = net(inp, config.scale)
        t2 = time.time()

        avg_time += (t2-t1)/20
    print("{} x{}: {:.3f}".format(config.model, config.scale, avg_time))

def main(config):
    model = importlib.import_module("model.{}".format(config.model)).Net
    net = model(scale=config.scale, group=1)
    inference(net, config)
 

if __name__ == "__main__":
    config = parse_args()
    main(config)

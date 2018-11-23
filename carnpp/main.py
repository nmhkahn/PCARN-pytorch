import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import json
import argparse
import importlib
from solver import Solver

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--train_data", type=str, default="./dataset/DIV2K.h5")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoint")
    
    parser.add_argument("--scale", type=int, default=0)
    parser.add_argument("--memo", type=str, default="")
    parser.add_argument("--print_interval", type=int, default=1000)
    parser.add_argument("--num_gpu", type=int, default=1)
    
    parser.add_argument("--group", type=int, default=1)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=600000)
    parser.add_argument("--decay", type=int, default=400000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--clip", type=float, default=10.0)

    return parser.parse_args()

def main(cfg):
    # dynamic import using --model argument
    net = importlib.import_module("model.{}".format(cfg.model)).Net
    print(json.dumps(vars(cfg), indent=4, sort_keys=True))
    
    solver = Solver(net, cfg)
    solver.fit()

if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)

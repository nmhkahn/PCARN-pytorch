import math
import functools
import torch.nn as nn

def default(m, scale=1.0):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        if m.weight.requires_grad:
            nn.init.kaiming_uniform_(m.weight.data, a=math.sqrt(5))
            m.weight.data *= scale
        if m.bias is not None and m.bias.requires_grad:
            m.bias.data.zero_()

def msra_uniform(m, scale=1.0):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        if m.weight.requires_grad:
            nn.init.kaiming_uniform_(m.weight.data, a=0)
            m.weight.data *= scale
        if m.bias is not None and m.bias.requires_grad:
            m.bias.data.zero_()

def msra_normal(m, scale=1.0):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        if m.weight.requires_grad:
            nn.init.kaiming_normal_(m.weight.data, a=0)
            m.weight.data *= scale
        if m.bias is not None and m.bias.requires_grad:
            m.bias.data.zero_()

def init_weights(net, init_type, scale=1.0):
    if init_type == "default":
        _init = functools.partial(default, scale=scale)
    elif init_type == "msra_uniform":
        _init = functools.partial(msra_uniform, scale=scale)
    elif init_type == "msra_normal":
        _init = functools.partial(msra_normal, scale=scale)
    else:
        raise NotImplementedError

    net.apply(_init)

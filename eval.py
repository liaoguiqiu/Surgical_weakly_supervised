import torch
import numpy as np

def cal_J(true, predict):
            AnB = true * predict  # assume that the lable are all binary
            AuB = true + predict
            AuB = torch.clamp(AuB, 0, 1)
            s = 0.000000001
            this_j = (torch.sum(AnB) + s) / (torch.sum(AuB) + s)
            return this_j
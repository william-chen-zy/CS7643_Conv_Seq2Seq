import torch
from torch import nn
import numpy as np
from math import floor, ceil


def get_padding(input_tensor, input_size: tuple, stride: tuple, kernel_size: tuple, padding='same', pad_mode='constant', c=0):
    h_old, w_old = input_size
    s_h, s_w = stride
    k_h, k_w = kernel_size
    if padding == 'same':
        h_new, w_new = h_old, w_old
        p_h = ((h_old - 1) * s_h - h_old + k_h)/2
        p_w = ((w_old - 1) * s_w - w_old + k_w)/2
    return nn.functional.pad(input_tensor,
                             (floor(p_w), ceil(p_w),
                              floor(p_h), ceil(p_h)),
                             pad_mode, c)
    
def train():
    pass
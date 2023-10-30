import numpy as np
import torch.nn as nn
import random
import os
import torch

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def conv_batch(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
        nn.BatchNorm1d(num_features=out_channels),
        nn.ReLU()
    )

def fc_batch(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU()
    )

def coordinates_normalize(coordinates):
    centroid = np.mean(coordinates, axis=0)
    coordinates = coordinates - centroid
    m = np.max(np.sqrt(np.sum(coordinates ** 2, axis=1)))
    coordinates /= m
    return coordinates

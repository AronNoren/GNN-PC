import sys
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)
import torch
from utils.data_loader import data_loader

dataset = data_loader()

print(dataset)
print(len(dataset))
data = dataset[0]
print(data)
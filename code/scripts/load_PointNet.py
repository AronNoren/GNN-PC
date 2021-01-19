import sys
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)
import torch
from torch_geometric.data import DataLoader
from utils.data_loader import get_ShapeNet
from models.PointNet import get_model
from utils.transform import transform
from utils.evaluation import evaluate_PN
from utils.equal_dataset import get_equal_dataset
SAVEPATH = 'code/models/saved_models/weight_rotate.pkl'
rotated_SAVEPATH = 'code/models/saved_models/new_PPFNetmodel_rotate.pkl'

train_data,test_data = get_ShapeNet(root = 'data/ShapeNet',split = 0.7,transformation=transform(points=64))
model = get_model(train_data.num_classes)
model.load_state_dict(torch.load(SAVEPATH))


test_acc = evaluate_PN(model,test_data,plots = True) #get_equal_dataset(test_data)
print(test_acc)

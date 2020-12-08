import sys
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)
import torch
from torch_geometric.data import DataLoader
from utils.data_loader import data_loader
from models.PointNet import get_model
from utils.training import train_PN
from utils.evaluation import evaluate_PN
dataset = data_loader(root = 'data/ShapeNet',categories = ['Airplane','Bag','Cap','Car','Chair','Earphone','Guitar'])
traindata = dataset[:int(4*len(dataset)/5)]
testdata = dataset[int(4*len(dataset)/5):]
train_PN(traindata,n_epochs = 5)
SAVEPATH = 'code/models/saved_models/model.pkl'
model = get_model(21)
model.load_state_dict(torch.load(SAVEPATH))
print(evaluate_PN(model,testdata))

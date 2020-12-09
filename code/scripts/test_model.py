import sys
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)
import torch
from utils.data_loader import get_ShapeNet
from models.PointNet import get_model
from utils.training import train_PN
from utils.evaluation import evaluate_PN
from torch_geometric.transforms import FixedPoints
dataset = get_ShapeNet(root = 'data/ShapeNet',transformation=FixedPoints(num=32))
traindata = dataset[:int(4*len(dataset)/5)]
testdata = dataset[int(4*len(dataset)/5):]
model = train_PN(traindata,n_epochs = 3)
SAVEPATH = 'code/models/saved_models/new_model.pkl'
torch.save(model.state_dict(), SAVEPATH)
#model = get_model()
#model.load_state_dict(torch.load(SAVEPATH))
print(evaluate_PN(model,testdata))

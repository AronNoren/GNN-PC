import sys
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)
import torch
from utils.data_loader import get_ShapeNet,get_equal_ShapeNet
from models.PointNet import get_model
from utils.training import train_PN
from utils.evaluation import evaluate_PN
from utils.transform import transform
from utils.visualizations import scatterplot,accuracies
from torch_geometric.data import Data, DataLoader
# train model on ordinary dataset
#traindata, testdata = get_ShapeNet(root = 'data/ShapeNet',transformation = transform(points = 256))

#model = train_PN(traindata,n_epochs = 10)
#SAVEPATH = 'code/models/saved_models/new_model.pkl'
#torch.save(model.state_dict(), SAVEPATH)

#train model on randomly rotated dataset
rotated_traindata, rotated_testdata = get_equal_ShapeNet(root = 'data/ShapeNet')
num = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for data in rotated_traindata:
	num[data.y] = num[data.y] +1
print("we done")
print(num) 
#rotated_traindata = DataLoader(rotated_traindata, batch_size=32)

#rotated_testdata = DataLoader(rotated_testdata, batch_size=32)
#print(rotated_traindata)

rotated_model, val = train_PN(rotated_traindata,rotated_testdata,n_epochs = 15)
print(val)
rotated_SAVEPATH = 'code/models/saved_models/weight_rotate.pkl'
torch.save(rotated_model.state_dict(), rotated_SAVEPATH)

#evaluate model
#print('Acc no rotations on training nor testdata: ' + str(evaluate_PN(model,testdata)))
#print('Acc no rotations on training but rotation on testdata: ' + str(evaluate_PN(model,rotated_testdata)))
print('Acc rotations on training and testdata: ' + str(evaluate_PN(rotated_model,rotated_testdata)))
accuracies(val)
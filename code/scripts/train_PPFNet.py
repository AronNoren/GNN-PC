import sys
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)
import torch
from utils.data_loader import get_ShapeNet,get_equal_ShapeNet
from utils.training import train_PPFNet
from utils.evaluation import evaluate_PN
from utils.transform import transform
from utils.visualizations import scatterplot,accuracies
from torch_geometric.data import DataLoader
# train model on ordinary dataset
#traindata, testdata = get_ShapeNet(root = 'data/ShapeNet',transformation = transform(points = 256))

#model = train_PPFNet(traindata,n_epochs = 10)
#SAVEPATH = 'code/models/saved_models/new_PPFNetmodel.pkl'
#torch.save(model.state_dict(), SAVEPATH)

#train model on randomly rotated dataset
points =256
rotated_traindata, rotated_testdata = get_equal_ShapeNet(root = 'data/ShapeNet',npoints = points )

ys = []
batches = DataLoader(rotated_testdata, batch_size=100,shuffle = True)
for data in batches:
    scatterplot(data.pos[0:points])
    break
#for data in batches:
#    hist = torch.bincount(data.y)
#    print(hist)

rotated_model, val_acc = train_PPFNet(rotated_traindata,rotated_testdata,n_epochs = 10)
rotated_SAVEPATH = 'code/models/saved_models/new_PPFNetmodel_rotate.pkl'
torch.save(rotated_model.state_dict(), rotated_SAVEPATH)

#evaluate model
#print('Acc no rotations on training nor testdata: ' + str(evaluate_PN(model,testdata)))
#print('Acc no rotations on training but rotation on testdata: ' + str(evaluate_PN(model,rotated_testdata)))
print('Acc rotations on training and testdata: ' + str(evaluate_PN(rotated_model,rotated_testdata,epoch = ' last')))
accuracies(val_acc)

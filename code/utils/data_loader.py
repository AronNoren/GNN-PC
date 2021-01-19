import sys
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)
from utils.ShapeNet import ShapeNet
from utils.transform import transform
import torch
'''
Here we can add more datasets
'''
def get_ShapeNet(root = 'data/ShapeNet',split = 0.8,categories = None,transformation=None,include_normal = True):

    '''
    loads/downloads the modified ShapeNet dataset into CPU.
    Input: 
    	root, directory of Dataset
    	categories, list of categories to include e.g. ['Airplane','Car']. None for all.
    	points, specified number of points in pointcloud. None for all avaliable points.
    '''
    dataset = ShapeNet(root,categories,transform = transformation)
    traindata = dataset[:int(split*len(dataset))]
    testdata = dataset[int(split*len(dataset)):]
    return traindata, testdata

def get_equal_ShapeNet(root = 'data/ShapeNet',split = 0.8,include_normal = True,npoints = 128):
    '''
    loads/downloads the modified ShapeNet dataset into CPU.
    Input: 
    	root, directory of Dataset
    	categories, list of categories to include e.g. ['Airplane','Car']. None for all.
    	points, specified number of points in pointcloud. None for all avaliable points.
    '''
    transformation = transform(points = npoints,rotate = 180)
    dataset = ShapeNet(root)
    traindata = dataset[:int(split*len(dataset))]
    testdata = dataset[int(split*len(dataset)):]
    ys = dataset.data.y
    hist = torch.bincount(ys[:int(split*len(dataset))])
    counter =max(hist)//hist
    print(counter)
    equal_train = []
    equal_test = []
    for data in traindata:
    	for nrsamples in range(counter[data.y]):
    		equal_train.append(transformation(data))
    for data in testdata:
        equal_test.append(transformation(data))
    return equal_train,equal_test

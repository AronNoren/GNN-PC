import sys
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)
from utils.node2graph import node2graph
from torch_geometric.datasets import ShapeNet
def data_loader(root = 'data/ShapeNet',categories = None,points = None):
	'''
	loads/downloads the ShapeNet dataset and created graph labels from node segment labeling.
	Input: 
		root, directory of Dataset
		categories, list of categories to include e.g. ['Airplane','Car']. None for all.
		points, specified number of points in pointcloud. None for all avaliable points.
	'''
	if points is not None:
		dataset = ShapeNet(root,categories,transform = FixedPoints(num=points))
	else:
		dataset = ShapeNet(root, categories)
	return node2graph(dataset)
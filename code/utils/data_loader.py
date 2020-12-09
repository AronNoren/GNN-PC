import sys
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)
from utils.ShapeNet import ShapeNet
from torch_geometric.transforms import FixedPoints
'''
Here we can add more datasets
'''
def get_ShapeNet(root = 'data/ShapeNet',categories = None,transformation=None,include_normal = True):

	'''
	loads/downloads the modified ShapeNet dataset into CPU.
	Input: 
		root, directory of Dataset
		categories, list of categories to include e.g. ['Airplane','Car']. None for all.
		points, specified number of points in pointcloud. None for all avaliable points.
	'''
	dataset = ShapeNet(root,categories,transform = transformation)
	return dataset
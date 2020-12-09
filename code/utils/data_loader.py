import sys
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)
from utils.ShapeNet import ShapeNet
from torch_geometric.transforms import FixedPoints
def get_ShapeNet(root = 'data/ShapeNet',categories = None,points = 256,include_normal = False):
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
		dataset = ShapeNet(root, categories,include_normals = False)
	return dataset
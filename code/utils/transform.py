import sys
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)
from torch_geometric.transforms import FixedPoints
from torch_geometric.transforms import Compose, RandomRotate

def transform(points = None,rotate = None):
	transform = None
	if points is not None:
		transform = FixedPoints(num=points)
	if rotate is not None:
		random_rotate = Compose([
    	RandomRotate(degrees=rotate, axis=0),
    	RandomRotate(degrees=rotate, axis=1),
    	RandomRotate(degrees=rotate, axis=2),
		])
		transform = Compose([transform,random_rotate])
	print(transform)
	return transform

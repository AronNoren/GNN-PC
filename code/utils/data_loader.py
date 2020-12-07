data_loader(root = 'data/ShapeNet',categories = None,points = None):
	'''
	loads the ShapeNet dataset and created graph labels from node segment labeling.
	Input: root, directory of Dataset
	'''
	if points None:
		dataset = ShapeNet(root,categories)
	else:
		dataset = ShapeNet(root, categories,transform = FixedPoints(num=points))
	return node2graph(dataset)
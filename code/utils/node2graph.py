def node2graph(dataset):

    '''
        simplifies the labeling from node-vise segmentation to graph-vise label according to.
	seg_classes = {
        'Airplane': [0, 1, 2, 3],
        'Bag': [4, 5],
        'Cap': [6, 7],
        'Car': [8, 9, 10, 11],
        'Chair': [12, 13, 14, 15],
        'Earphone': [16, 17, 18],
        'Guitar': [19, 20, 21],
        'Knife': [22, 23],
        'Lamp': [24, 25, 26, 27],
        'Laptop': [28, 29],
        'Motorbike': [30, 31, 32, 33, 34, 35],
        'Mug': [36, 37],
        'Pistol': [38, 39, 40],
        'Rocket': [41, 42, 43],
        'Skateboard': [44, 45, 46],
        'Table': [47, 48, 49],
    	}
        Input: dataset with dataset.y label tensor of size Nx1
	Output: dataset with dataset.y label tensor of size 1x1
    '''
    indexswap = [0,0,0,0,1,1,2,2,3,3,3,3,4,4,4,4,5,5,5,6,6,6,7,7,8,8,8,8,9,9,10,10,10,10,10,10,11,11,12,12,12,13,13,13,14,14,14,15,15,15]
    for i in range(0,len(dataset)):
        dataset[i].y[0] = indexswap[node]
        dataset[i].y.resize_(1)


    return dataset
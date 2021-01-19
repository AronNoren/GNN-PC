import torch
import numpy
def get_equal_dataset(test_dataset):
    ys = test_dataset.data.y[len(test_dataset.data.y)-len(test_dataset):]
    hist = torch.bincount(ys)
    minclasses = torch.min(hist).item()
    print(minclasses*test_dataset.num_classes)
    all_index = torch.tensor([])
    for classi in range(0,test_dataset.num_classes):
        index = (ys == classi).nonzero()
        index =torch.squeeze(index)
        all_index = torch.cat((all_index,index[0:minclasses]))
    # all_index = all_index.numpy()
    #print(all_index.tolist())
    indices = list(map(int,all_index.tolist()))
    print(len(test_dataset))
    print(len(ys))
    print(max(indices))
    print(test_dataset[indices])
    return test_dataset[indices]



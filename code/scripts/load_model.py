import sys
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)
import torch
from torch_geometric.data import DataLoader
from utils.data_loader import data_loader
from models.PointNet import get_model
SAVEPATH = 'code/models/saved_models/net_params.pkl'
model = get_model(4)
model.load_state_dict(torch.load(SAVEPATH))

dataset = data_loader(root = 'data/ShapeNet',categories = ['Airplane','Bag','Cap','Car'])
#train_loader = DataLoader(dataset[:int(4*len(dataset)/5)], batch_size=10,shuffle = True) #0.8/0.2 train/test-split
test_loader = DataLoader(dataset[int(4*len(dataset)/5):],batch_size = 100)

def test(model, loader):
    model.eval()

    total_correct = 0
    for data in loader:
        logits = model(data.pos, data.batch)
        pred = logits.argmax(dim=-1)
        print(pred)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)

test_acc = test(model, test_loader)
print(test_acc)

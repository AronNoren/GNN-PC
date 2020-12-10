import sys
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)
import torch
from torch_geometric.data import DataLoader
from utils.data_loader import get_ShapeNet
from models.PointNet import get_model
from utils.transform import transform
SAVEPATH = 'code/models/saved_models/new_model.pkl'


train_data,test_data = get_ShapeNet(root = 'data/ShapeNet',split = 0.8,transformation=transform(points=256))
model = get_model(train_data.num_classes)
model.load_state_dict(torch.load(SAVEPATH))
test_loader = DataLoader(test_data,batch_size = 10)

def test(model, loader):
    model.eval()

    total_correct = 0
    for data in loader:
        logits = model(data.pos, data.batch)
        pred = logits.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)

test_acc = test(model, test_loader)
print(test_acc)

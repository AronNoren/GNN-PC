import sys
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)
import torch
from torch_geometric.data import DataLoader
from models.PPFNet import get_model

def evaluate_PN(model,test_dataset):
    test_loader = DataLoader(test_dataset, batch_size=1)
    def test(model, loader):
        model.eval()
        with torch.no_grad():
            total_correct = 0
            for data in loader:
                logits = model(data.pos, data.batch, data.x)
                pred = logits.argmax(dim=-1)
                total_correct += int((pred == data.y).sum())

        return total_correct / len(loader.dataset)
    test_acc = test(model, test_loader)

    return test_acc
import sys
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)
import torch
from torch_geometric.data import DataLoader
from models.PPFNet import get_model
from utils.visualizations import heatplot, scatterplot
def test(model, loader):
        model.eval()
        preds = torch.tensor([])
        correct = torch.tensor([])
        with torch.no_grad():
            total_correct = 0
            for data in loader:
                logits = model(data.pos, data.batch, data.x)
                pred = logits.argmax(dim=-1)
                preds = torch.cat((preds,pred))
                correct = torch.cat((correct,data.y))
                total_correct += int((pred == data.y).sum())

        return total_correct / len(loader.dataset), preds, correct

def evaluate_PN(model,test_dataset,epoch,plots = False):
    #print(len(test_dataset))
    test_loader = DataLoader(test_dataset, batch_size=1)
    


    test_acc,preds,correct = test(model, test_loader)
    #scatterplot(test_dataset[0].data.pos,test_dataset[0].data.y)
    #print(preds)
    #print(correct)
    heatplot(preds,correct,epoch)
    return test_acc
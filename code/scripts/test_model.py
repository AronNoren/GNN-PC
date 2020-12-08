import sys
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)
import torch
from torch_geometric.data import DataLoader
from utils.data_loader import data_loader
from models.PointNet import get_model

dataset = data_loader(root = 'data/ShapeNet',categories = ['Airplane','Bag','Cap','Car','Chair','Earphone','Guitar'])
train_loader = DataLoader(dataset[:int(4*len(dataset)/5)], batch_size=10) #0.8/0.2 train/test-split
test_loader = DataLoader(dataset[int(4*len(dataset)/5):], batch_size=10)
model = get_model(7)
SAVEPATH = 'code/models/saved_models/net_params_7.pkl'
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
weight = torch.tensor([1.0,35.0,50.0,3.0,0.72,39.0,3.43])
criterion = torch.nn.CrossEntropyLoss(weight)  # Define loss criterion.
print("data loaded")
def train(model, optimizer, loader):
    model.train()
    
    total_loss = 0
    for data in train_loader:
        print(data.y)
        optimizer.zero_grad()  # Clear gradients.
        logits = model(data.pos, data.batch)  # Forward pass.
        loss = criterion(logits, data.y)  # Loss computation.

        loss.backward()  # Backward pass.
        optimizer.step()  # Update model parameters.
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_loader.dataset)


torch.no_grad()
def test(model, loader):
    model.eval()

    total_correct = 0
    for data in loader:
        logits = model(data.pos, data.batch)
        pred = logits.argmax(dim=-1)
        print(pred)
        total_correct += int((pred == data.y).sum())

    return total_correct / len(loader.dataset)

for epoch in range(1, 3):
    loss = train(model, optimizer, train_loader)
    test_acc = test(model, test_loader)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Test Accuracy: {test_acc:.4f}')
torch.save(model.state_dict(), SAVEPATH)
model = get_model(7)
model.load_state_dict(torch.load(SAVEPATH))
model.eval()
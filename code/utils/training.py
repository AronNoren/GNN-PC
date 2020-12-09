import sys
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)
import torch
from torch_geometric.data import DataLoader
from models.PointNet import get_model

def train_PN(train_dataset,SAVENAME = 'model',validation_dataset = None,batchsize = 10,n_epochs =5,shuffled = True,weight = False):
    SAVEPATH = 'code/models/saved_models/' + SAVENAME + '.pkl'
    n_classes = train_dataset.num_classes
    train_loader = DataLoader(train_dataset, batch_size=batchsize,shuffle = shuffled)
    print(n_classes)
    model = get_model(n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    if weight is not False:
        weight = torch.tensor([1.0,35.0,50.0,3.0,0.72,39.0,3.43])
        criterion = torch.nn.CrossEntropyLoss(weight)  # Define loss criterion.
    else:
        criterion = torch.nn.CrossEntropyLoss()

    def train(model, optimizer, train_loader):
        model.train()
    
        total_loss = 0
        for data in train_loader:
            #print(data.y)
            optimizer.zero_grad()  # Clear gradients.
            logits = model(data.pos, data.batch, data.x)  # Forward pass.
            loss = criterion(logits, data.y)  # Loss computation.

            loss.backward()  # Backward pass.
            optimizer.step()  # Update model parameters.
            total_loss += loss.item() * data.num_graphs

        return total_loss / len(train_loader.dataset)

    for epoch in range(1, n_epochs):
        print(epoch)
        loss = train(model, optimizer, train_loader)
        #test_acc = test(model, test_loader)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
    torch.save(model.state_dict(), SAVEPATH)
    return model


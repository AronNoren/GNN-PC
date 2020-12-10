import sys
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)
import torch
from torch_geometric.data import DataLoader
from models.PointNet import get_model

def train_PN(train_dataset,validation_dataset = None,batchsize = 10,n_epochs =5,shuffled = True,weight = False):
    '''
    Training function for PointNet, returns a trained model.
    Input:
    train_dataset (dataset) pointcloud data for training
    validation_dataset (dataset) optional pointcloud data for validation acc during training
    batchsize = 10 (int) nr of data from train_dataset to process in parallell.
    n_epochs = 5 (int) nr of times the train_dataset is trained on.
    shuffled = True (bool) shuffles train_dataset before dividing into batches.
    weight = False (bool) used for uneven dataset to 
    '''
    #SAVEPATH = 'code/models/saved_models/' + SAVENAME + '.pkl'
    n_classes = train_dataset.num_classes
    train_loader = DataLoader(train_dataset, batch_size=batchsize,shuffle = shuffled)
    model = get_model(n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    if weight is not False:
        weight = torch.bincount(train_dataset.data.y) #count number of each label
        weight = torch.true_divide(torch.max(weight),weight)
        criterion = torch.nn.CrossEntropyLoss(weight)  # Define loss criterion.
    else:
        criterion = torch.nn.CrossEntropyLoss()

    def train(model, optimizer, train_loader):
        model.train()
    
        total_loss = 0
        for data in train_loader:
            #print(data.y)
            optimizer.zero_grad()  # Clear gradients.
            logits = model(data.pos, data.batch)  # Forward pass.
            loss = criterion(logits, data.y)  # Loss computation.

            loss.backward()  # Backward pass.
            optimizer.step()  # Update model parameters.
            total_loss += loss.item() * data.num_graphs

        return total_loss / len(train_loader.dataset)

    for epoch in range(1, n_epochs):
        loss = train(model, optimizer, train_loader)
        val_acc = 0
        if validation_dataset is not None:
        	val_acc = evaluation(model, test_loader)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, val acc:{val_acc:.4f}')
    
    return model


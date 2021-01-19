import sys
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)
import torch
from torch_geometric.data import DataLoader
from models import PPFNet, PointNet
from utils.evaluation import evaluate_PN
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from utils.visualizations import printProgressBar
import time
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
    n_classes = 16
    train_loader = DataLoader(train_dataset, batch_size=batchsize,shuffle = shuffled)
    model = PointNet.get_model(n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    if weight is not False:
        weight = torch.sqrt(torch.bincount(train_dataset.data.y).float()) #count number of each label
        #print(weight)
        weight = torch.true_divide(torch.max(weight),weight)
        #print(weight)
        weight2 = compute_class_weight('balanced',np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]),train_dataset.data.y.tolist())
        #print("done")
        #print(weight)
        #print(weight2)
        criterion = torch.nn.CrossEntropyLoss(torch.tensor(weight2).float())  # Define loss criterion.
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
    val_accs = torch.tensor([0])
    for epoch in range(1, n_epochs):
        loss = train(model, optimizer, train_loader)
        val_acc = 0
        if validation_dataset is not None:
            val_acc = evaluate_PN(model, validation_dataset,epoch)
            val_accs = torch.cat((val_accs,torch.tensor([val_acc])))
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, val acc:{val_acc:.4f}')
    
    return model, val_accs


def train_PPFNet(train_dataset,validation_dataset = None,batchsize = 100,n_epochs =5,shuffled = True,weight = False):
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
    n_classes = 16
    train_loader = DataLoader(train_dataset, batch_size=batchsize,shuffle = shuffled)
    model = PPFNet.get_model(n_classes,hidden_layers = 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    if weight is not False:
        weight = torch.sqrt(torch.bincount(train_dataset.data.y).float()) #count number of each label
        #print(weight)
        weight = torch.true_divide(torch.max(weight),weight)
        #print(weight)
        weight2 = compute_class_weight('balanced',np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]),train_dataset.data.y.tolist())
        #print("done")
        #print(weight)
        #print(weight2)
        criterion = torch.nn.CrossEntropyLoss(torch.tensor(weight2).float())  # Define loss criterion.
    else:
        criterion = torch.nn.CrossEntropyLoss()

    def train(model, optimizer, train_loader):
        model.train()
        t_0 = time.time()
        L = len(train_loader)
        printProgressBar(0, L, prefix = 'Progress:', suffix = 'Complete, ETA: '+ 'NAN', length = 50)
        i = 0
        total_loss = 0
        for data in train_loader:
            print(torch.bincount(data.y))
            i = i+1
            t_1 = time.time()
            T = (t_1-t_0)
            printProgressBar(i, L, prefix = 'Progress:', suffix = 'Complete, ETA: '+str(round(((T)/(i/L)-T)/60,2)) +' min', length = 50)
            optimizer.zero_grad()  # Clear gradients.
            logits = model(data.pos, data.batch, data.x)  # Forward pass.
            loss = criterion(logits, data.y)  # Loss computation.

            loss.backward()  # Backward pass.
            optimizer.step()  # Update model parameters.
            total_loss += loss.item() * data.num_graphs
        print("total time: "+ str(round(T/60,2))+"min  ")
        return total_loss / len(train_loader.dataset)
    val_accs = torch.tensor([0])
    for epoch in range(1, n_epochs):
        loss = train(model, optimizer, train_loader)
        val_acc = 0
        
        if validation_dataset is not None:
            val_acc = evaluate_PN(model, validation_dataset,epoch)
            val_accs = torch.cat((val_accs,torch.tensor([val_acc])))
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, val acc:{val_acc:.4f}')
    
    return model, val_accs

    '''n_classes = train_dataset.num_classes
    train_loader = DataLoader(train_dataset, batch_size=batchsize,shuffle = shuffled)
    model = PPFNet.get_model(n_classes)
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
            logits = model(data.pos, data.batch, data.x)  # Forward pass.
            loss = criterion(logits, data.y)  # Loss computation.

            loss.backward()  # Backward pass.
            optimizer.step()  # Update model parameters.
            total_loss += loss.item() * data.num_graphs

        return total_loss / len(train_loader.dataset)

    for epoch in range(1, n_epochs+1):
        loss = train(model, optimizer, train_loader)
        val_acc = 0
        if validation_dataset is not None:
        	val_acc = evaluation(model, test_loader)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, val acc:{val_acc:.4f}')
    
    return model
'''

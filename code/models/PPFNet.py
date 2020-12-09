from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing, PPFConv
import torch
import torch.nn.functional as F
from torch_cluster import knn_graph
from torch_geometric.nn import global_mean_pool


class PPFNet(torch.nn.Module):
    def __init__(self,classes, num_features=0, hidden_layers=32):
        super(PPFNet, self).__init__()

        #torch.manual_seed(12345) # Seed for testing

        # To use the PPFConv we need to create a MLP (torch.nn).
        # The input to the MLP should be the number of features of each node + 4 
        # (PPFConv adds 4 terms which have to do with distance and angles between two points)

        # The MLP here is arbitrary and can be changed
        mlp1 = Sequential(
            Linear(num_features + 4,hidden_layers),
            ReLU(),
            Linear(hidden_layers, hidden_layers)
        )
        self.conv1 = PPFConv(mlp1)

        # The MLP here is arbitrary and can be changed
        mlp2 = Sequential(
            Linear(hidden_layers + 4,hidden_layers),
            ReLU(),
            Linear(hidden_layers, hidden_layers)
        ) 
        self.conv2 = PPFConv(mlp2)
        self.classifier = Linear(hidden_layers, classes)
        
    def forward(self, pos, batch, normal, h=None):
        # DON'T USE POSITION AS FEATURES -> LEAVE h=None

        # Compute the kNN graph:
        # Here, we need to pass the batch vector to the function call in order
        # to prevent creating edges between points of different examples.
        edge_index = knn_graph(pos, k=16, batch=batch, loop=False)
        
        # 3. Start bipartite message passing -> 2 levels deep message passing.
        h = self.conv1(x=h, pos=pos, edge_index=edge_index, normal=normal)
        h = h.relu()
        h = self.conv2(x=h, pos=pos, edge_index=edge_index, normal=normal)
        h = h.relu() 

        # 4. Global Pooling.
        h = global_mean_pool(h, batch)  # Pooling scheme can be changed to ex: glboal_max_pool
        
        # 5. Classifier.
        return self.classifier(h)

def get_model(classes,num_features=0, hidden_layers=32, weights=None):
	return PPFNet(classes, num_features, hidden_layers)
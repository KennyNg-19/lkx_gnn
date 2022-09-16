'''
Author: kenny_wu
Date: 2022-09-16 19:06:57
LastEditors: kenny_wu
LastEditTime: 2022-09-16 19:12:52
Description: 
'''
import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv, GATv2Conv, GATConv, SAGEConv

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(21, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 112)
        # self.conv2 = GCNConv(hidden_channels, 50)
        # self.lin1 = Linear(50,256)
        self.lin2 = nn.Linear(112,256)
        self.lin3 = nn.Linear(256,100)
        self.lin4 = nn.Linear(100,1)

        self.bnm1 = nn.BatchNorm1d(112, momentum=0.1)
        self.bnm2 = nn.BatchNorm1d(256, momentum=0.1)
        self.bnm3 = nn.BatchNorm1d(100, momentum=0.1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        # x = F.dropout(x, p=0.1, training=self.training)
        # x = self.dropout(x)
        x = self.conv2(x, edge_index)
        # x = self.lin1(x)
        x = self.bnm1(x)
        x = x.relu()
        x = self.dropout(x)
        x = self.lin2(x)
        x = self.bnm2(x)
        x = x.relu()
        x = self.lin3(x)
        x = self.bnm3(x)
        x = x.relu()
        x = self.lin4(x)  
        return x

class GCN1(torch.nn.Module):
    def __init__(self): 
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(15, 150)
        self.conv2 = GCNConv(150, 200)
        self.lin2 = nn.Linear(200,400)
        self.lin3 = nn.Linear(400,200)
        self.lin4 = nn.Linear(200,1)

        self.pool1 = nn.MaxPool2d(2,2)

        self.bnm1 = nn.BatchNorm1d(200, momentum=0.1)
        self.bnm2 = nn.BatchNorm1d(400, momentum=0.1)
        self.bnm3 = nn.BatchNorm1d(200, momentum=0.1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        # x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index)
        # x = self.lin1(x)
        x = self.bnm1(x)
        x = x.relu()
        x = self.dropout(x)
        x = self.lin2(x)
        x = self.bnm2(x)
        x = x.relu()
        # x = self.dropout(x)
        x = self.lin3(x)
        x = self.bnm3(x)
        x = x.relu()
        x = self.lin4(x)  
        return x


if __name__ == '__main__':
    pass

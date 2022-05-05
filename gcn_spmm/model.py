import torch.nn as nn
import torch
from torch.nn.modules.module import Module
import dgl.nn as dglnn
from dgl.nn.pytorch.glob import MaxPooling
class GNN(Module):
    def __init__(self,num_features=0,num_layers=2,num_hidden=32,dropout_ratio=0):
        super(GNN,self).__init__()
        self.nfeat = num_features
        self.nlayer = num_layers
        self.nhid = num_hidden
        self.dropout_ratio = dropout_ratio
        self.M_num = 2  #the number of global feature
        self.gc = nn.ModuleList([dglnn.SAGEConv(self.nfeat if i==0 else self.nhid, self.nhid,'pool') for i in range(self.nlayer)])
        self.bn = nn.ModuleList([nn.LayerNorm(self.nhid) for i in range(self.nlayer)])
        self.relu = nn.ModuleList([nn.ReLU() for i in range(self.nlayer)])
        self.pooling = MaxPooling()
        self.fc = nn.Linear(self.nhid,1)
        self.fc1 = nn.Linear(self.nhid+self.M_num,self.nhid)
        self.dropout = nn.ModuleList([nn.Dropout(self.dropout_ratio) for i in range(self.nlayer)])

    def forward_single_model(self,g,features):
        x = self.relu[0](self.bn[0](self.gc[0](g,features)))
        x = self.dropout[0](x)
        for i in range(1,self.nlayer):
            x = self.relu[i](self.bn[i](self.gc[i](g,x)))
            x = self.dropout[i](x)
        return x

    def forward(self, g, features, matrix_features):
        x = self.forward_single_model(g, features)
        with g.local_scope():
            g.ndata['h'] = x
            x = self.pooling(g,x)
            x = torch.cat([x, matrix_features], dim=-1)
            x = self.fc1(x)
            return self.fc(x)

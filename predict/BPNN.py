#!coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F 
import copy

# 定义 BPNN 全连接网络
class BPNN(nn.Module):
    def __init__(self):  # nfeat：输入的特征层长度, nhid 隐藏层, nclass 输出层，类别,
        super(BPNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=640, out_features=256),
            nn.ReLU(),# inplace=True
            nn.Linear(in_features=256, out_features=64)
            # nn.ReLU(),
            # nn.Linear(in_features=64, out_features=64),
            # nn.ReLU(),
        )     
        
        topo = [[0,0,1,0,0,0,0,1],
                [0,0,0,1,1,0,1,0],
                [1,0,0,1,0,1,0,1],
                [0,1,1,0,1,0,1,0],
                [0,1,0,1,0,1,1,0],
                [0,0,1,0,1,0,0,1],
                [0,1,0,1,1,0,0,1],
                [1,0,1,0,0,1,1,0]
                ]
        self.topo_tensor = torch.tensor(topo).float()

    def forward(self, x, adj):
        # x.size = bs * 10 * 8 * 8
        y1 = x.reshape(x.shape[0], -1) #.clone()
        # x = x.view(x.shape[0], -1)
        y3 = self.fc(y1)
        
        
        return y3.reshape(-1, 8, 8) # .clone() #  * self.topo_tensor.clone()
        # return out.view(-1, 8, 8) * self.topo_tensor
        
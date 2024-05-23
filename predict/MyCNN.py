#!coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F 

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)    # 只把batch_size留下
# 定义  CNN 
class MyCNN(nn.Module):
    def __init__(self, switchNumber,  feature_len, hidden_size, dropout, s2h = True):  # nfeat：输入的特征层长度, nhid 隐藏层, nclass 输出层，类别,
        super(MyCNN, self).__init__()
        self.switchNumber = switchNumber
        self.feature_len = feature_len
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=hidden_size, kernel_size=(3, 3), padding=1), # 加一个padding 使图像尺寸不变
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_size, out_channels=1, kernel_size=(1, 1)), # 通过1*1的卷据核将多层图像变成一个图像即可
            nn.ReLU(inplace=True),
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
        if s2h :
            self.topo_tensor = 1
        else:
            self.topo_tensor = torch.tensor(topo).float()

    def forward(self, input, adj):
        # 每次输入的input 都是一组feature： [batch_size, seq_len, 8, 8] ，把seq_len 当作cnn中的通道channel
        return self.conv(input).squeeze(dim=1) * self.topo_tensor    # 乘以拓扑，让不该出现值的位置不要出现  # 输出本来是 （bs,1,8,8）的，需要改成（bs,8,8），这样才和label的size一致
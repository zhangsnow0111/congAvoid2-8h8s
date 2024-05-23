#!coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F 

# 定义  CNN + LSTM
class CNN_LSTM(nn.Module):
    def __init__(self, hidden_size,  dropout):  # nfeat：输入的特征层长度, nhid 隐藏层, nclass 输出层，类别,
        super(CNN_LSTM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=hidden_size, kernel_size=(3, 3), padding=1), # 加一个padding 使图像尺寸不变
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_size, out_channels=1, kernel_size=(1, 1)), # 通过1*1的卷据核将多层图像变成一个图像即可
            nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(in_features=64, out_features=64)
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
        output = self.conv(x) 
        output = output.view(output.shape[0], output.shape[1], -1) 
        output, (hn, cn) = self.lstm(output)         # 输出（batch_size, seq_len, hidden_size）
        output = output[:,-1,:]           # （batch_size,  hidden_size）
        # print(output.shape)
        output = output.view(output.shape[0], 8, 8)
        # print("output size: {} \n".format(output.shape))
        # print(output)
        # print("output * self.topo_tensor size: {}\n" .format((output * self.topo_tensor).shape))
        # print(output * self.topo_tensor)
        return output * self.topo_tensor    # 乘以拓扑，让不该出现值的位置不要出现值 output * self.topo_tensor
        
        
        
        
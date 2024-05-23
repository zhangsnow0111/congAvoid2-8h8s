#!coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F 

# 定义  LSTM + CNN
class ALSTM_CNN(nn.Module):
    def __init__(self, switchNumber,  feature_len, hidden_size,  dropout):  # nfeat：输入的特征层长度, nhid 隐藏层, nclass 输出层，类别,
        super(ALSTM_CNN, self).__init__()
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
        self.weight = nn.Parameter(torch.randn(10, 64)) # 用作注意力机制的weight， size必须和lstm的输出完全一致
        # 定义偏置向量b，形状为(input_size,)
        self.bias = nn.Parameter(torch.zeros(64))
        self.fc = nn.Linear(in_features=64, out_features=64)

    def forward(self, x, adj):
        x = x.view(x.shape[0], x.shape[1], -1) 
        output, (hn, cn) = self.lstm(x)         # 输出（batch_size, seq_len, hidden_size）
        A = self.weight * output + self.bias        # * 就是对应位置直接乘
        output = A * output
        output = output.view(output.shape[0], output.shape[1], 8, 8) 
        return F.relu(self.conv(output).squeeze(dim=1)) 
        
        
        
        
#!coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F 

# 该模型其实是投机取巧的，不可用
# 因为先用lstm，再接CNN，实际上lstm就没用了；主流的将cnn与lstm结合的算法都是先cnn再lstm。
# 定义  LSTM + CNN
class LSTM_CNN(nn.Module):
    def __init__(self, switchNumber,  feature_len, hidden_size,  dropout):  # nfeat：输入的特征层长度, nhid 隐藏层, nclass 输出层，类别,
        super(LSTM_CNN, self).__init__()
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
        x = x.view(x.shape[0], x.shape[1], -1) 
        output, (hn, cn) = self.lstm(x)         # 输出（batch_size, seq_len, hidden_size）
        output = output.view(output.shape[0], output.shape[1], 8, 8) 
        # return F.relu(self.conv(output).squeeze(dim=1)) 
        output = self.conv(output).squeeze(dim=1)
        return output * self.topo_tensor    # 乘以拓扑，让不该出现值的位置不要出现值 
        
        
        
        
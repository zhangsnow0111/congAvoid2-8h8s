#!coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F 

# 定义  LSTM
class MyLSTM(nn.Module):
    def __init__(self):  # nfeat：输入的特征层长度, nhid 隐藏层, nclass 输出层，类别,
        super(MyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(in_features=64, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=8*8)        
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

    # 在LSTM前后都增加一层 fc， 减少reshape带来的影响
    def forward(self, x, adj):
        batch_size, _, _, _ = x.shape 
        output = x.view(x.shape[0], x.shape[1], -1)
        output = torch.clamp(output, min=1e-7)    # 限制最小值
        # output = self.fc1(output)
        # output = self.relu(output)
        output, (hn, cn) = self.lstm(output)     # 输出（batch_size, seq_len, hidden_size）
        output = F.dropout(output, 0.3, training=self.training)
        output = output[:,-1,:].squeeze()          # （batch_size,  hidden_size）
        # output = self.fc2(output)
        output = output.view(batch_size, 8, 8)
        return output
         
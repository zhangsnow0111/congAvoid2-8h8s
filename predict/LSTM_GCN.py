#!coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F 
from predict.gcn_layer import GraphConvolution

# 定义  LSTM_GCN
class LSTM_GCN(nn.Module):
    def __init__(self, feature_len, gcn_hidden_size, dropout):  # nfeat：输入的特征层长度, nhid 隐藏层, nclass 输出层，类别,
        super(LSTM_GCN, self).__init__()
        
        self.dropout = dropout      # 定义dropout
        self.gc1 = GraphConvolution(feature_len, gcn_hidden_size) # 自定义卷积层1： 
        self.gc2 = GraphConvolution(gcn_hidden_size, feature_len) # 自定义卷积层2：
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=64, batch_first=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=10, out_channels=1, kernel_size=3, padding=1)    # 将10个通道的数据转化为1个通道
        self.fc1 = nn.Linear(in_features=64, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
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
        output = x.view(x.shape[0], x.shape[1], -1)
        output, (hn, cn) = self.lstm(output)     # 输出（batch_size, seq_len, hidden_size）
        output = output.view(output.shape[0], output.shape[1], 8, 8)  # 输出（batch_size, seq_len, 8, 8)
        output = F.relu(output)
        # print("output: %s" % type(output))
        # print("adj: %s" % type(adj))
        output = self.gc1(output, adj)
        output = F.dropout(output, self.dropout, training=self.training)
        output = F.relu(self.gc2(output, adj))
        output = self.conv(output)
        # print("output size: {} \n".format(output.shape))
        # print(output)
        # print("output.squeeze() size: {} \n" .format((output.squeeze()).shape))
        # print(output.squeeze() * self.topo_tensor)
        # output = output.squeeze()
        # output = output.unsqueeze(0)
        # print("output final size: {} \n".format(output.shape))
        return output.squeeze(1) * self.topo_tensor    # 乘以拓扑，让不该出现值的位置不要出现值 
    
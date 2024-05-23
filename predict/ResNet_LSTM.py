#!coding=utf-8

import torch.nn as nn
import torch.nn.functional as F 

# 定义残差块:
class BasicResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # 传入x的size = bs, 1 , 8 , 8
        out = self.conv1(x)
        identity = out  # 这样才能和后面的out 相加
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = F.relu(out)
        return out
    
# 定义  ResNet + LSTM
class ResNet_LSTM(nn.Module):
    def __init__(self, switchNumber, feature_len):  # nfeat：输入的特征层长度, nhid 隐藏层, nclass 输出层，类别,
        super(ResNet_LSTM, self).__init__()
        self.switchNumber = switchNumber
        self.feature_len = feature_len
        self.res_block1 = BasicResBlock(10, 32) # 经过残差块，输入图像尺寸不变，但通道数会从input_channels=10 变成 hidden_channels
        self.res_block2 = BasicResBlock(32, 1) 
        # self.fc1 = nn.Linear(64, 64)
        self.lstm = nn.LSTM(input_size=switchNumber*feature_len, hidden_size=switchNumber*feature_len, batch_first=True)
        # LSTM模型接收的输入大小应为(batch_size, seq_len, input_size)，输出大小为(batch_size, seq_len, hidden_size)。
        self.fc2 = nn.Linear(64, 64)
 
    def forward(self, x, adj):
        out = self.res_block1(x) # 经过残差块1，结果 [batch_size, hidden_channels, 8, 8]，  # hidden_channels = 32
        out = self.res_block2(out) # 经过残差块2，结果 [batch_size, hidden_channels, 8, 8]，  # hidden_channels = 1
        
        # out = out.reshape(out.shape[0], out.shape[1], -1)   # 把二维的特征拉成一维，（bs, 1, 64）
        # out, (hidden, cell) = self.lstm(out)  # 输出out的尺寸（batch_size, seq_len, hidden_size）
        # out = out[:,-1,:].squeeze()          # （batch_size,  hidden_size）
        # out = self.fc2(out)
        return out.reshape(-1, self.switchNumber, self.feature_len)      #  （batch_size, switchNumber, feature_len）


'''
如果使用torch自带的gcn，先这样import
from torch_geometric.nn import GCNConv
然后在使用时需要对邻接矩阵进行转化：
    def forward(self, x, adj_matrix):
        # adj_matrix是网络拓扑的图邻接矩阵 size 8*8
        edge_index = torch.nonzero(adj_matrix).t()
        # 但torch内生的gcn需要的邻接矩阵是一个大小为 2 x num_edges 的张量，其中每一列表示一条边的起点和终点
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from predict.gcn_layer import GraphConvolution

# 定义残差块:
class BasicResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # 传入x的size = bs, 10 , 8 , 8
        out = self.conv1(x)
        identity = out  # 这样才能和后面的out 相加
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = F.relu(out)
        return out
        # 传入out的size = bs, out_channels , 8 , 8
    
# 定义整个模型：
class ResGCNLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, lstm_hidden_size, dropout):
        super(ResGCNLSTM, self).__init__()
        self.dropout = dropout  # 定义dropout
        self.gc1 = GraphConvolution(in_features = 8, out_features = 8) # in_features, out_features
        # GCN 每层传入的两个参数就定义了weight矩阵的大小
        self.gc2 = GraphConvolution(in_features = 8, out_features = 8)
        self.res_block1 = BasicResBlock(input_channels, hidden_channels) # 经过残差块，输入图像尺寸不变，但通道数会从input_channels=10 变成 hidden_channels
        self.res_block2 = BasicResBlock(hidden_channels, 1) 
        self.lstm = nn.LSTM(input_size=64, hidden_size=lstm_hidden_size, batch_first=True)
        # LSTM模型接收的输入大小应为(batch_size, seq_len, input_size)，输出out大小为(batch_size, seq_len, hidden_size)。
        self.fc = nn.Linear(lstm_hidden_size, 8 * 8)
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
        # 先通过gcn得到图信息
        x = F.relu(self.gc1(x, adj))  # 每次输入的x 都是一组feature： [batch_size, seq_len, 8, 8]， adj  [8, 8]
        # x = F.dropout(x, self.dropout, training=self.training)
        out = self.gc2(x, adj) # GCN 的计算结果 [batch_size, 10, 8, 8]，
        out = self.res_block1(out) # 经过残差块1，结果 [batch_size, hidden_channels, 8, 8]，  # hidden_channels = 32
        out = self.res_block2(out) # 经过残差块2，结果 [batch_size, hidden_channels, 8, 8]，  # hidden_channels = 1
        # 此时out.size (bs, 1, 8, 8)
        out = out.view(out.shape[0], out.shape[1], -1)   # 把二维的特征拉成一维，（bs, hidden_channels, 64）
        out, (hidden, cell) = self.lstm(out)  # 输出out的尺寸（batch_size, hidden_channels, 64）
        out = out[:,-1,:].squeeze()          # （batch_size,  64）
        # # out = hidden.squeeze(0)             # 这两个得到out的途径都可以
        out = self.fc(out)      # 通过全连接将（batch_size,  lstm_hidden_size）映射为 (batch_size, 64)
        return out.view(-1, 8, 8) * self.topo_tensor        #  （batch_size, switchNumber, feature_len）

#!coding=utf-8
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
        
# 本人的模型对GCN进行了改写，并未直接用
# 定义一个GCN 模型，其中有 两个卷积层
class MyGCN(nn.Module):
    def __init__(self, feature_len, gcn_hidden_size, dropout): # nfeat：输入的特征层长度, nhid 隐藏层, nclass 输出层，类别,
        super(MyGCN, self).__init__()

        self.gc1 = GraphConvolution(feature_len, gcn_hidden_size) # 自定义卷积层1： 
        self.gc2 = GraphConvolution(gcn_hidden_size, feature_len) # 自定义卷积层2： 
        self.conv1 = nn.Conv2d(in_channels=10, out_channels=1, kernel_size=3, padding=1)    # 将10个通道的数据转化为1个通道
        self.res_block1 = BasicResBlock(10, out_channels = 32) # 经过残差块，输入图像尺寸不变，但通道数会从input_channels=10 变成 out_channels
        self.res_block2 = BasicResBlock(32, out_channels = 1) # 经过残差块，输入图像尺寸不变，但通道数会从input_channels=10 变成 out_channels
        self.dropout = dropout      # 定义dropout
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
        # 每次输入的x 都是一组feature： [batch_size, seq_len, 8, 8]， adj  [8, 8]
        out = F.relu(self.gc1(x, adj)) 
        out = F.dropout(out, self.dropout, training=self.training)
        out = F.relu(self.gc2(out, adj))
        # out = self.conv1(out)
        out = self.res_block1(out)
        out = self.res_block2(out)
        # print("out size: {} \n".format(out.shape))
        # print(out)
        out = out.squeeze(1)  
        # print("out.squeeze(1) size: {} \n".format(out.shape))
        # print(out)

        return out * self.topo_tensor    # 乘以拓扑，让不该出现值的位置不要出现值 

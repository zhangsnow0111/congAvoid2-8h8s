#!coding=utf-8
import torch.nn as nn
import torch.nn.functional as F
from predict.gcn_layer import GraphConvolution

# 本人的模型对GCN进行了改写，并未直接用
# 定义一个GCN 模型，其中有 两个卷积层
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout): # nfeat：输入的特征层长度, nhid 隐藏层, nclass 输出层，类别,
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid) # 自定义卷积层1：输入的特征为nfeat，维度是2708，输出的特征为nhid，维度是16；
        self.gc2 = GraphConvolution(nhid, nclass) # 自定义卷积层2：输入的特征为nhid，维度是16，输出的特征为nclass，维度是7（即类别的结果）
        self.dropout = dropout      # 定义dropout

    def forward(self, x, adj):  # 网络投入数据后就进入这里 前向传播函数，可见网路的前向传播顺序是 ： gc1 --> relu --> dropout --> gc2 --> softmax
        x = F.relu(self.gc1(x, adj)) # 输入的x 就是feature： [2708, 1433]， adj  [2708, 2708]
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)  # softmax 将一个向量整合成 0~1 里的值，其总和为1
        # softmax 简单定义
        # scores = np.array([123, 456, 789])
        # softmax = np.exp(scores) / np.sum(np.exp(scores))

# if __name__ == "__main__":
#     print(1)
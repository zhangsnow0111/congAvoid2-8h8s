#!coding=utf-8

import math
import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

# 做GCN的一层。
class GraphConvolution(Module): #Simple GCN layer, 
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 由于weight，bias是可以训练的，因此使用parameter定义
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # size()函数主要是用来统计矩阵元素个数，或矩阵某一维上的元素个数的函数  size（1）为行
        stdv = 1. / math.sqrt(self.weight.size(1))
        # uniform() 方法将随机生成下一个实数，它在 [x, y] 范围内
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # 此处主要定义的是本层的前向传播，通常采用的是 A ∗ X ∗ W 的计算方法。由于A是一个sparse变量，因此其与X进行卷积的结果也是稀疏矩阵。
        # 这里的邻接矩阵A已经是和度矩阵计算过的了
        support = torch.matmul(input, self.weight)  # 这里必须用torch.matmul才能进行高维矩阵相乘
        # torch.spmm(a,b)是稀疏矩阵相乘
        output = torch.matmul(adj, support) # 这里也改了
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

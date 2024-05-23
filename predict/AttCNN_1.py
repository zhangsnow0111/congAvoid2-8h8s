
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Input shape: (batch_size, in_channels, 8, 8)
        batch_size, _, h, w = x.size()

        # Compute query, key, and value
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Reshape for matrix multiplication
        q = q.view(batch_size, -1, h * w).permute(0, 2, 1)  # (batch_size, h * w, in_channels)
        k = k.view(batch_size, -1, h * w)  # (batch_size, in_channels, h * w)
        v = v.view(batch_size, -1, h * w)  # (batch_size, in_channels, h * w)

        # Compute attention weights
        attn_weights = torch.matmul(q, k)  # (batch_size, h * w, h * w)
        attn_weights = self.softmax(attn_weights)

        # Compute output
        out = torch.matmul(v, attn_weights.permute(0, 2, 1))  # (batch_size, in_channels, h * w)
        out = out.view(batch_size, -1, h, w)  # (batch_size, in_channels, 8, 8)

        out = self.softmax(out + x)
        return out


class AttCNN(nn.Module):
    def __init__(self, seq_length=10, out_channels=1, num_filters=64, dropout=0.5):
        super(AttCNN, self).__init__()
        self.seq_length = seq_length
        self.conv1 = nn.Conv2d(seq_length, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.attention = SelfAttention(num_filters)
        self.conv2 = nn.Conv2d(num_filters, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x,adj):
        x = self.conv1(x)  # (batch_size, num_filters, 8, 8)
        x = self.bn1(x)     # 这里使用bn比不适用bn效果好很多！
        x = self.relu(x)
        x = self.dropout(x)
        x = self.attention(x)  # (batch_size, num_filters, 8, 8)

        x = self.conv2(x)  # (batch_size, out_channels, 8, 8)
 
        return x.squeeze(dim=1)
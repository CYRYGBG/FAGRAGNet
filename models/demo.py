# -*- coding = utf-8 -*-
# @File : demo.py
# @Software : PyCharm


import torch
from math import sqrt
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange, repeat
from torch_geometric.data import DataLoader
from torch_geometric.nn import DenseGraphConv, DenseGCNConv
from torch_geometric.nn import TopKPooling, SAGPooling, EdgePooling
from torch.nn import Linear, Dropout, PReLU, Conv2d, MaxPool2d, AvgPool2d, BatchNorm2d
from torch_geometric.utils import to_dense_batch, to_dense_adj, dense_to_sparse
import warnings
warnings.filterwarnings('ignore')


class MultiHeadSelfAttention(nn.Module):
	def __init__(self, dim_in, dim_out, dim_k=8*64, dim_v=8*64, num_heads=8, dropout=0.):
		super(MultiHeadSelfAttention, self).__init__()
		assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
		self.dim_in = dim_in
		self.dim_k = dim_k
		self.dim_v = dim_v
		self.num_heads = num_heads
		self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
		self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
		self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
		self._norm_fact = 1 / sqrt(dim_k // num_heads)

		self.to_out = nn.Sequential(
			nn.Linear(5 * dim_k, dim_out),
			nn.Dropout(dropout)
		)

	def forward(self, x):
		# x: tensor of shape (batch, n, dim_in)
		# (Batch*channels, Freq_bands, features)
		batch, n, dim_in = x.shape
		assert dim_in == self.dim_in

		nh = self.num_heads
		dk = self.dim_k // nh  # dim_k of each head
		dv = self.dim_v // nh  # dim_v of each head

		q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
		k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
		v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

		dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
		dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n

		att = torch.matmul(dist, v)  # batch, nh, n, dv
		att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v

		out = self.to_out(att.reshape((batch, -1)))
		return out

class FreqBandsAttention(nn.Module):
	def __init__(self, n_channels, hidden_size):
		super().__init__()
		self.linear_in = MultiHeadSelfAttention(dim_in=265, dim_out=5)
		self.linear_out = nn.Linear(n_channels, n_channels)
		self.tanh = nn.Tanh()
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, x):
		s = torch.squeeze(x)  # [batch*channels, 1, Freq_bands, features]
		s = self.linear_in(s)
		s = self.tanh(self.linear_out(s))
		attention_map = self.softmax(s)
		attention_map = attention_map.unsqueeze(1)
		attention_map = attention_map.unsqueeze(-1)
		out = x * attention_map

		return out

class ResidualGraph(nn.Module):
	"""Self-organized Graph Construction Module
	Args:
		in_features: size of each input sample
		bn_features: size of bottleneck layer
		out_features: size of each output sample
		topk: size of top k-largest connections of each channel
	"""

	def __init__(self, in_features, bn_features, out_features, topk, channels=62):
		super().__init__()

		self.channels = channels
		self.in_features = in_features
		self.bn_features = bn_features
		self.out_features = out_features
		self.topk = topk

		self.gate = nn.Linear(in_features, out_features)

		self.bnlin = Linear(in_features, bn_features)  # Linear Bottleneck layer#(44*32, 32)
		# 这个输入的in_features代表每个节点特征维度，out_features表示每个节点输出的特征维度
		# 修改处
		self.gconv = DenseGraphConv(in_features, 64)  # (44*32, 32)
		self.gconv2 = DenseGraphConv(64, 64)  # (44*32, 32)
		self.gconv3 = DenseGraphConv(64, 64)  # (44*32, 32)
		self.gconv4 = DenseGraphConv(64, 64)  # (44*32, 32)
		self.gconv5 = DenseGraphConv(64, 64)  # (44*32, 32)
		self.gconv6 = DenseGraphConv(64, 64)  # (44*32, 32)
		self.gconv7 = DenseGraphConv(64, out_features)  # (44*32, 32)
		# 修改处结尾

	def forward(self, x):
		# x:[Batch*channels, 32, 1, 65]
		# reshape:[Batch, channels, 65*32]
		# bnlin:[Batch,channels,bn_features]
		# adj:[Batch,channels,channels]
		# amask:[Batch,channels,channels]
		x = x.reshape(-1, self.channels, self.in_features)

		x_mask = F.tanh(self.gate(x))

		xa = torch.tanh(self.bnlin(x))
		adj = torch.matmul(xa, xa.transpose(2, 1))
		adj = torch.softmax(adj, 2)  # 对每个邻接矩阵的行使用softmax
		amask = torch.zeros(xa.size(0), self.channels, self.channels).to(next(self.parameters()).device)
		amask.fill_(0.0)
		s, t = adj.topk(self.topk, 2)  # 在维度2上保留self.topk个最大值，s是返回的最大值，t是返回的最大值的索引
		amask.scatter_(2, t, s.fill_(1))  # 将s中的值修改为1，填充到amask（所有元素为0）的对应位置上
		adj = adj * amask  # 对应位置元素相乘，也就是将非topk里的元素全部置为0
		# x:[Batch, channels, 65*32]
		# adj:[Batch, channels, channels]
		x = F.relu(self.gconv(x, adj))
		# x:[Batch, channels, out_features]

		# 修改处
		tmpX = F.relu(self.gconv2(x, adj))  # 后续可以直接加个全连接层实现:AXW1+XW2
		x = x + tmpX
		tmpX = F.relu(self.gconv3(x, adj))
		x = x + tmpX
		tmpX = F.relu(self.gconv4(x, adj))
		x = x + tmpX
		tmpX = F.relu(self.gconv5(x, adj))
		x = x + tmpX
		tmpX = F.relu(self.gconv6(x, adj))
		x = x + tmpX
		x = F.relu(self.gconv7(x, adj))
		# 修改处结尾
		# print(x.shape, x_mask.shape)
		return x * x_mask

class FAGRAGNet(nn.Module):
	def __init__(self):
		super().__init__()

		drop_rate = 0.1
		topk = 10
		self.channels = 62

		self.cwa = FreqBandsAttention(n_channels=5, hidden_size=32)

		# 增加处
		self.more_conv1 = Conv2d(1, 32, (3, 3), padding=(3 // 2, 3 // 2))
		self.more_drop1 = Dropout(drop_rate)
		self.more_pool1 = MaxPool2d((1, 4))
		self.more_sogc1 = ResidualGraph(66 * 32 * 5, 64, 32, topk)

		self.more_conv2 = Conv2d(32, 32, (3, 3), padding=(3 // 2, 3 // 2))
		self.more_drop2 = Dropout(drop_rate)
		self.more_pool2 = MaxPool2d((1, 4), padding=(0, 2))
		self.more_sogc2 = ResidualGraph(17 * 32 * 5, 64, 32, topk)
		# 增加处结尾

		self.conv1 = Conv2d(32, 32, (1, 3))
		self.drop1 = Dropout(drop_rate)
		self.pool1 = MaxPool2d((1, 2))
		self.sogc1 = ResidualGraph(5 * 32 * 7, 64, 32, topk)

		self.conv2 = Conv2d(32, 64, (1, 3))
		self.drop2 = Dropout(drop_rate)
		self.pool2 = MaxPool2d((1, 2))
		self.sogc2 = ResidualGraph(10 * 64, 64, 32, topk)

		self.conv3 = Conv2d(64, 128, (1, 2))
		self.drop3 = Dropout(drop_rate)
		# self.pool3 = MaxPool2d((1, 4))
		self.sogc3 = ResidualGraph(5 * 128, 64, 32, topk)

		self.drop4 = Dropout(drop_rate)

		self.linend = Linear(self.channels * 32 * 5, 1024)

		# 对图卷积模块的输出拼接的部分使用门控机制
		self.gate = nn.Linear(self.channels * 32 * 5, self.channels * 32 * 5)
		self.relu = nn.ReLU()

		self.linend2 = Linear(1024, 3)

	def forward(self, x):
		# input shape: [batch*channel, 1, Freq_bands, Features]
		x = x.reshape(-1, 1, 5, 265)  # (Batch*channels, 1, Freq_bands, Features)

		x = self.cwa(x)

		x = F.relu(self.more_conv1(x))
		x = self.more_drop1(x)
		x = self.more_pool1(x)
		x1 = self.more_sogc1(x)

		x = F.relu(self.more_conv2(x))
		x = self.more_drop2(x)
		x = self.more_pool2(x)
		x2 = self.more_sogc2(x)

		x = F.relu(self.conv1(x))
		x = self.drop1(x)
		x = self.pool1(x)
		x3 = self.sogc1(x)

		x = F.relu(self.conv2(x))
		x = self.drop2(x)
		x = self.pool2(x)
		x4 = self.sogc2(x)

		x = F.relu(self.conv3(x))
		x = self.drop3(x)
		x5 = self.sogc3(x)

		x = torch.cat([x1, x2, x3, x4, x5], dim=1)
		x = self.drop4(x)

		x = x.reshape(-1, self.channels * 32 * 5)

		x = torch.tanh(self.gate(x)) * x
		x = self.linend(x)
		x = self.relu(x)
		x = self.linend2(x)

		return x


if __name__ == "__main__":
	data = torch.randn((16 * 62, 1, 5, 265))
	model = FAGRAGNet()
	print(model(data).shape)

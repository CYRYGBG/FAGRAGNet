# -*- coding = utf-8 -*-
# @File : utils.py
# @Software : PyCharm


import math
import numpy as np


def to_categorical(y, num_classes=None, dtype='float32'):
	y = np.array(y, dtype='int16')
	input_shape = y.shape  # 在本代码中，输入的就是一个一维数据
	if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
		input_shape = tuple(input_shape[:-1])
	y = y.ravel()
	if not num_classes:  # 如果没有输入类别个数，则根据y的最大取值确定类别个数
		num_classes = np.max(y) + 1
	n = y.shape[0]  # 得到样本个数
	categorical = np.zeros((n, num_classes), dtype=dtype)  # [15*45, 3]
	categorical[np.arange(n), y] = 1  # 将对应位置元素置为1，得到每个样本的独热编码，[15*45, 3]
	output_shape = input_shape + (num_classes,)  # 元组之间的加法其实是元组之间的拼接，即得到[15*45, 3]
	categorical = np.reshape(categorical, output_shape)  #
	return categorical


def normalize(data):
	mee = np.mean(data, 0)  # 对最后一维进行标准化
	data = data - mee
	stdd = np.std(data, 0)
	data = data / (stdd + 1e-7)
	return data


def compute_DE(data):
	variance = np.var(data, ddof=1)  # 求得方差
	return math.log(2 * math.pi * math.e * variance) / 2  # 微分熵求取公式


def moving_average_filter(data, window_size):
	window = np.ones(window_size) / window_size
	return np.convolve(data, window, mode='same')




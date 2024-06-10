# -*- coding = utf-8 -*-
# @File : utils.py
# @Software : PyCharm


import math
import numpy as np


def to_categorical(y, num_classes=None, dtype='float32'):
	y = np.array(y, dtype='int16')
	input_shape = y.shape
	if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
		input_shape = tuple(input_shape[:-1])
	y = y.ravel()
	if not num_classes:
		num_classes = np.max(y) + 1
	n = y.shape[0]
	categorical = np.zeros((n, num_classes), dtype=dtype)
	categorical[np.arange(n), y] = 1
	output_shape = input_shape + (num_classes,)
	categorical = np.reshape(categorical, output_shape)
	return categorical


def normalize(data):
	mee = np.mean(data, 0)
	data = data - mee
	stdd = np.std(data, 0)
	data = data / (stdd + 1e-7)
	return data


def compute_DE(data):
	variance = np.var(data, ddof=1)
	return math.log(2 * math.pi * math.e * variance) / 2


def moving_average_filter(data, window_size):
	window = np.ones(window_size) / window_size
	return np.convolve(data, window, mode='same')




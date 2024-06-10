# -*- coding = utf-8 -*-
# @File : IVDataset.py
# @Software : PyCharm

import os
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
from utils import compute_DE, moving_average_filter, normalize, to_categorical
from sklearn.preprocessing import MinMaxScaler
from scipy import signal
import scipy.io as sio
import glob
import math
from torch.utils.data import TensorDataset
from scipy.signal import savgol_filter

def get_IV_official_data(path, save_path, num_subjects=15):
	print('Using IV Official De Feature!')
	if os.path.exists(os.path.join(save_path, 'IV_Official_DE_data.npy')):
		sub_data = np.load(os.path.join(save_path, 'IV_Official_DE_data.npy'), allow_pickle=True)
		sub_label = np.load(os.path.join(save_path, 'IV_Official_DE_labels.npy'), allow_pickle=True)
	else:
		label = [[1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
				 [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
				 [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]]
		sub_data = []  # 被试层次
		sub_label = []  # 被试层次
		for i in range(1, 16):  # 被试层次循环
			files_path = []
			for j in range(1, 4):  # 得到被试在三个文件夹下分别对应的文件
				files = os.listdir(os.path.join(path, str(j)))
				for tmp in files:
					if str(tmp.split('_')[0]) == str(i):
						files_path.append(os.path.join(path, str(j), tmp))

			medium_data = []  # 文件夹层次
			medium_label = []
			for num in range(1, 4):  # 单个实验文件层次循环
				data = sio.loadmat(files_path[num - 1], verify_compressed_data_integrity=False)
				keys = data.keys()
				de_mov = [k for k in keys if 'de_movingAve' in k]
				tmp_label = label[num - 1]

				# 不需要单个文件层次的添加，直接按照[被试, 实验]来进行添加
				for t in range(24):  # 文件内样本个数，每次实验中有15个视频测试
					temp_data = data[de_mov[t]].transpose(0, 2, 1)
					data_length = temp_data.shape[-1]
					mov_i = np.zeros((62, 5, 64))
					mov_i[:, :, :data_length] = temp_data
					medium_data.append(mov_i)
					medium_label.append(tmp_label[t])
			medium_data = np.array(medium_data)
			medium_label = np.array(medium_label)
			sub_data.append(medium_data)  # 被试层次添加
			sub_label.append(medium_label)
		sub_data = np.array(sub_data)
		sub_label = np.array(sub_label)

		np.save(os.path.join(save_path, 'IV_Official_DE_data.npy'), sub_data)
		np.save(os.path.join(save_path, 'IV_Official_DE_labels.npy'), sub_label)
		print('IV Depend Data Saved!')
	print(sub_data.shape, sub_label.shape)
	return sub_data, sub_label

def MakeDataset(sub_id, path, save_path, num_subjects=15, one_hot=False):
	data, labels = get_IV_official_data(path, save_path, num_subjects)  # (15, 72, 62, 5, 64) (15, 72)
	print('Label Range: ', labels.min(), labels.max())
	subjects, trials, channels, bands, features = data.shape
	index_list = list(range(num_subjects))
	del index_list[sub_id]
	test_index = sub_id
	train_index = index_list

	print('Building train and test dataset')
	print('Train id: ', train_index)
	print('Test id: ', test_index)
	# get train & test
	X = data[train_index, :].reshape(-1, channels, bands, features)
	Y = labels[train_index, :].reshape(-1)
	testX = data[test_index, :].reshape(-1, channels, bands, features)
	testY = labels[test_index, :].reshape(-1)
	# get labels
	if one_hot:
		_, Y = np.unique(Y, return_inverse=True)  # return_inverse的参数可以用于重构原来的数组
		Y = to_categorical(Y, 4)  # 由于原来的标签是-1,0,1，因为使用了np.unique所以将其转换为了0,1,2
		_, testY = np.unique(testY, return_inverse=True)
		testY = to_categorical(testY, 4)

	print(X.shape, testX.shape, Y.shape, testY.shape)
	X = torch.Tensor(X.astype(float))  # transform to torch tensor
	testX = torch.Tensor(testX.astype(float))
	Y = torch.Tensor(Y.astype(float))  # transform to torch tensor
	testY = torch.Tensor(testY.astype(float))

	train_dataset = TensorDataset(X, Y.long(), )  # create your datset
	test_dataset = TensorDataset(testX, testY.long())

	return train_dataset, test_dataset


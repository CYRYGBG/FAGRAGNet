# -*- coding = utf-8 -*-
# @File : DeDataset.py
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


def get_official_data(path, save_path, num_subjects=15):
	print('Using Official De Feature!')
	if os.path.exists(os.path.join(save_path, 'Official_DE_data.npy')):
		sub_mov = np.load(os.path.join(save_path, 'Official_DE_data.npy'), allow_pickle=True)
		sub_label = np.load(os.path.join(save_path, 'Official_DE_labels.npy'), allow_pickle=True)
	else:
		label = sio.loadmat(os.path.join(path, 'label.mat'))['label']
		files = sorted(glob.glob(os.path.join(path, '*_*')))
		sublist = set()

		for f in files:
			if torch.cuda.is_available():
				sublist.add(f.split('/')[-1].split('_')[0])
			else:
				sublist.add(f.split('\\')[-1].split('_')[0])

		print('Total number of subjects: {:.0f}'.format(len(sublist)))
		sublist = sorted(list(sublist))
		print(sublist)

		sub_mov = []
		sub_label = []

		for sub_i in range(num_subjects):
			sub = sublist[sub_i]
			# 调整为不会重复读取文件
			sub_files = os.listdir(path)
			sub_files = [os.path.join(path, k) for k in sub_files
						 if str(k.split('_')[0]) == sub]

			mov_data = []
			for f in sub_files:
				print(f)
				data = sio.loadmat(f, verify_compressed_data_integrity=False)
				keys = data.keys()
				de_mov = [k for k in keys if 'de_movingAve' in k]

				mov_datai = []
				for t in range(15):
					temp_data = data[de_mov[t]].transpose(0, 2, 1)
					data_length = temp_data.shape[-1]
					mov_i = np.zeros((62, 5, 265))
					mov_i[:, :, :data_length] = temp_data

					mov_datai.append(mov_i)
				mov_datai = np.array(mov_datai)
				mov_data.append(mov_datai)

			mov_data = np.vstack(mov_data)
			mov_data = normalize(mov_data)  # 对每个通道的数据进行归一化
			sub_mov.append(mov_data)
			sub_label.append(np.hstack([label, label, label]).squeeze())

		sub_mov = np.array(sub_mov, dtype=object)
		sub_label = np.array(sub_label, dtype=object) + 1
		np.save(os.path.join(save_path, 'Official_DE_data.npy'), sub_mov)
		np.save(os.path.join(save_path, 'Official_DE_labels.npy'), sub_label)
		print('Depend Data Saved!')
	print(sub_mov.shape, sub_label.shape)
	return sub_mov, sub_label

def MakeDataset(sub_id, path, save_path, num_subjects=15, one_hot=False):

	data, labels = get_official_data(path, save_path, num_subjects)  # (15, 45, 62, 5, 265) (15, 45)

	print('Label Range: ', labels.min(), labels.max())
	subjects, trials, channels, bands, features = data.shape
	# 被试留一实验：当前循环哪个被试，就把哪个被试取出来当做测试集
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
		Y = to_categorical(Y, 3)  # 由于原来的标签是-1,0,1，因为使用了np.unique所以将其转换为了0,1,2
		_, testY = np.unique(testY, return_inverse=True)
		testY = to_categorical(testY, 3)

	print(X.shape, testX.shape, Y.shape, testY.shape)
	X = torch.Tensor(X.astype(float))  # transform to torch tensor
	testX = torch.Tensor(testX.astype(float))
	Y = torch.Tensor(Y.astype(float))  # transform to torch tensor
	testY = torch.Tensor(testY.astype(float))

	train_dataset = TensorDataset(X, Y.long(), )  # create your datset
	test_dataset = TensorDataset(testX, testY.long())

	return train_dataset, test_dataset




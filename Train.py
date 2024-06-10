# -*- coding = utf-8 -*-
# @File : Train.py
# @Software : PyCharm


import os
import time
import numpy as np
import random
import pandas as pd
import scipy.io as sio
import torch
from scipy.special import softmax
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from utils import to_categorical

from models.demo import FAGRAGNet
from datasets.DeDataset import MakeDataset
import warnings

warnings.filterwarnings('ignore')
subjects = 15
epochs = 500
classes = 3
Network = FAGRAGNet


def set_seed(seed: int = 42):
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

	np.random.seed(seed)

	random.seed(seed)


set_seed()


def train(model, train_loader, crit, optimizer, device):
	model.train()
	loss_all = 0
	for data, label in train_loader:
		data = data.to(device)
		label = label.to(device)
		optimizer.zero_grad()

		output = model(data)

		loss = crit(output, label)
		loss.backward()
		loss_all += data.shape[0] * loss.item()
		optimizer.step()
	return loss_all / len(train_dataset)


def evaluate(model, loader, device, save_result=False):
	model.eval()

	predictions = []
	labels = []
	pred = []
	with torch.no_grad():
		for data, label in loader:
			data = data.to(device)
			label = label.to(device)
			output = model(data)
			output = output.detach().cpu().numpy()

			output = np.squeeze(output)
			predictions.append(np.argmax(output, axis=1))
			pred.append(output)
			labels.append(label.reshape(-1).cpu().numpy())

	pred = np.vstack(pred)
	predictions = np.concatenate(predictions, axis=0)
	labels = np.concatenate(labels, axis=0)

	# AUC score estimation
	AUC = roc_auc_score(to_categorical(labels, 3), softmax(np.array(pred).squeeze()), average='macro')
	f1 = f1_score(labels, predictions, average='macro')

	# Accuracy
	acc = accuracy_score(labels, predictions)

	return AUC, acc, f1


path = r'./SEED/ExtractedFeatures'
save_path = r'./SEED/MyModelData'

print('Cross Validation')
result_data = []
for cv_n in range(0, 15):
	train_dataset, test_dataset = MakeDataset(cv_n, path, save_path)
	train_loader = DataLoader(train_dataset, batch_size=16, drop_last=False, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=16)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = Network().to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
	crit = torch.nn.CrossEntropyLoss()

	epoch_data = []
	best_score = 0
	best_record = None
	for epoch in range(epochs):
		t0 = time.time()
		loss = train(model, train_loader, crit, optimizer, device)
		train_AUC, train_acc, train_f1 = evaluate(model, train_loader, device)
		val_AUC, val_acc, val_f1 = evaluate(model, test_loader, device)

		epoch_data.append([str(cv_n), epoch + 1, loss, train_acc, val_acc])
		t1 = time.time()
		print('V{:01d}, EP{:03d}, Loss:{:.3f}, AUC:{:.3f}, Acc:{:.3f}, VAUC:{:.2f}, Vacc:{:.2f}, Time: {:.2f}'.
			  format(cv_n, epoch + 1, loss, train_AUC, train_acc, val_AUC, val_acc, (t1 - t0)))

		if train_acc > 0.99:
			break

	print('Results::::::::::::')
	print('V{:01d}, EP{:03d}, Loss:{:.3f}, AUC:{:.3f}, Acc:{:.3f}, VAUC:{:.2f}, Vacc:{:.2f}'.
		  format(cv_n, epoch + 1, loss, train_AUC, train_acc, val_AUC, val_acc))

	result_data.append([str(cv_n), epoch + 1, loss, train_AUC, train_acc, val_AUC, val_acc, val_f1])

print(result_data)

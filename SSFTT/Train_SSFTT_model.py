import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
import sys
import time
import matplotlib.pyplot as plt
import os
import os
import zipfile
import argparse
import scipy.io as sio
import SSFTTnet
import get_cls_map

# Ensure the main directory is in the system path to import data_fetcher
main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir)

from data_fetcher import loadData

dataset_mapping = {
    'HanChuan': 'WHU-Hi-HanChuan',
    'HongHu': 'WHU-Hi-HongHu',
    'LongKou': 'WHU-Hi-LongKou'
}

def loadDataWrapper(dataset, kaggle_json_path):
    dataset_name = dataset_mapping[dataset]
    data, labels = loadData(dataset_name, kaggle_json_path)
    
    return data, labels

def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX

def create_data_loader():
    X, y = loadData()
    pca_components = 30

    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)

    print('\n... ... PCA tranformation ... ...')
    X_pca = applyPCA(X, numComponents=pca_components)
    print('Data shape after PCA: ', X_pca.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load dataset and train model.')
    parser.add_argument('--dataset', type=str, required=True, choices=dataset_mapping.keys(), help='Dataset name (HanChuan, HongHu, LongKou)')
    parser.add_argument('--kaggle_json_path', type=str, required=True, help='Path to the directory containing kaggle.json')
    
    args = parser.parse_args()

    # Use loadDataWrapper to load data and labels
    data, labels = loadDataWrapper(args.dataset, args.kaggle_json_path)

    train_loader, test_loader, all_data_loader, y_all= create_data_loader()
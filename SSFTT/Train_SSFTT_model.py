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

BATCH_SIZE_TRAIN = 64

NUM_CLASSES = {
    'HanChuan': 16,
    'HongHu': 22,
    'LongKou': 9
}

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

def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X

    return newX

def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):
    # Padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # Split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1

    return patchesData, patchesLabels

def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=testRatio, random_state=randomState, stratify=y)

    return X_train, X_test, y_train, y_test

def create_data_loader(dataset, kaggle_json_path):
    X, y = loadDataWrapper(dataset, kaggle_json_path)
    test_ratio = 0.90
    patch_size = 13
    pca_components = 30

    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)

    print('\n... ... PCA tranformation ... ...')
    X_pca = applyPCA(X, numComponents=pca_components)
    print('Data shape after PCA: ', X_pca.shape)

    print('\n... ... Create data cubes ... ...')
    X_pca, y_all = createImageCubes(X_pca, y, windowSize=patch_size)
    print('Data cube X shape: ', X_pca.shape)
    print('Data cube y shape: ', y.shape)

    print('\n... ... Create train & test data ... ...')
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y_all, test_ratio)
    print('Xtrain shape: ', Xtrain.shape)
    print('Xtest  shape: ', Xtest.shape)

    X = X_pca.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtest = Xtest.reshape(-1, patch_size, patch_size, pca_components, 1)
    print('before transpose: Xtrain shape: ', Xtrain.shape)
    print('before transpose: Xtest  shape: ', Xtest.shape)

    X = X.transpose(0, 4, 3, 1, 2)
    Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
    Xtest = Xtest.transpose(0, 4, 3, 1, 2)
    print('after transpose: Xtrain shape: ', Xtrain.shape)
    print('after transpose: Xtest  shape: ', Xtest.shape)

    X = TestDS(X, y_all)
    trainset = TrainDS(Xtrain, ytrain)
    testset = TestDS(Xtest, ytest)
    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=BATCH_SIZE_TRAIN, shuffle=False, num_workers=0)
    all_data_loader = torch.utils.data.DataLoader(dataset=X, batch_size=BATCH_SIZE_TRAIN, shuffle=False, num_workers=0)

    return train_loader, test_loader, all_data_loader, y

def train(train_loader, num_classes, epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = SSFTTnet.SSFTTnet(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    total_loss = 0
    for epoch in range(epochs):
        net.train()
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            outputs = net(data)
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' % (epoch + 1, total_loss / (epoch + 1), loss.item()))

    print('Finished Training')

    return net, device

def test(device, net, test_loader):
    count = 0
    net.eval()
    y_pred_test = 0
    y_test = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))

    return y_pred_test, y_test

def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def acc_reports(y_test, y_pred_test, dataset):
    target_mapping = {
    'HanChuan': ['Strawberry', 'Cowpea', 'Soybean', 'Sorghum', 'Water spinach', 'Watermelon', 'Greens', 'Trees', 'Grass', 'Red roof', 
                 'Gray roof', 'Plastic', 'Bare soil', 'Road', 'Bright object', 'Water'],
    'HongHu': ['Red roof', 'Road', 'Bare soil', 'Cotton', 'Cotton firewood', 'Rape', 'Chinese cabbage', 'Pakchoi', 'Cabbage', 'Tuber mustard', 
               'Brassica parachinensis', 'Brassica chinensis', 'Small Brassica chinensis', 'Lactuca sativa', 'Celtuce', 'Film covered lettuce', 
               'Romaine lettuce', 'Carrot', 'White radish', 'Garlic sprout', 'Broad bean', 'Tree'],
    'LongKou': ['Corn', 'Cotton', 'Sesame', 'Broad-leaf soybean', 'Narrow-leaf soybean', 'Rice', 'Water', 'Roads and houses', 'Mixed weed']
    }
    target_names = target_mapping[dataset]
    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, oa*100, confusion, each_acc*100, aa*100, kappa*100

class TestDS():
    def __init__(self, Xtest, ytest):
        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class TrainDS():
    def __init__(self, Xtrain, ytrain):
        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load dataset and train model.')
    parser.add_argument('--dataset', type=str, required=True, choices=dataset_mapping.keys(), help='Dataset name (HanChuan, HongHu, LongKou)')
    parser.add_argument('--kaggle_json_path', type=str, required=True, help='Path to the directory containing kaggle.json')
    args = parser.parse_args()

    os.makedirs("cls_result", exist_ok=True)
    os.makedirs("cls_params", exist_ok=True)

    num_classes = NUM_CLASSES[args.dataset]

    train_loader, test_loader, all_data_loader, y_all= create_data_loader(args.dataset, args.kaggle_json_path)
    tic1 = time.perf_counter()
    net, device = train(train_loader, num_classes, epochs=100)
    torch.save(net.state_dict(), f'cls_params/SSFTTnet_params_{args.dataset}.pth')
    toc1 = time.perf_counter()
    tic2 = time.perf_counter()
    y_pred_test, y_test = test(device, net, test_loader)
    toc2 = time.perf_counter()
    
    classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test, args.dataset)
    classification = str(classification)
    Training_Time = toc1 - tic1
    Test_time = toc2 - tic2
    report_file_name = f"cls_result/classification_report_{args.dataset}.txt"
    with open(report_file_name, 'w') as x_file:
        x_file.write('{} Training_Time (s)'.format(Training_Time))
        x_file.write('\n')
        x_file.write('{} Test_time (s)'.format(Test_time))
        x_file.write('\n')
        x_file.write('{} Kappa accuracy (%)'.format(kappa))
        x_file.write('\n')
        x_file.write('{} Overall accuracy (%)'.format(oa))
        x_file.write('\n')
        x_file.write('{} Average accuracy (%)'.format(aa))
        x_file.write('\n')
        x_file.write('{} Each accuracy (%)'.format(each_acc))
        x_file.write('\n')
        x_file.write('{}'.format(classification))
        x_file.write('\n')
        x_file.write('{}'.format(confusion))

    # get_cls_map.get_cls_map(net, device, all_data_loader, y_all)
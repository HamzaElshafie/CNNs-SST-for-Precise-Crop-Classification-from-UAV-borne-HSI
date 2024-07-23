import numpy as np
import matplotlib.pyplot as plt
import os

if 'COLAB_GPU' in os.environ:
    main_dir = '/content/Spectral-Spatial-Transformers-for-Precise-Crop-Classification-from-UAV-borne-Hyperspectral-Images'
else:
    main_dir = ''

def get_classification_map(y_pred, y):

    height = y.shape[0]
    width = y.shape[1]
    k = 0
    cls_labels = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            target = int(y[i, j])
            if target == 0:
                continue
            else:
                cls_labels[i][j] = y_pred[k]+1
                k += 1

    return  cls_labels

def list_to_colormap(x_list, dataset):
    if dataset == 'HanChuan':
        colors = [
            [143, 41, 72], # Strawberry
            [7, 247, 227], # Cowpea
            [232, 16, 210], # Soybean
            [137, 44, 168], #Sorghum
            [132, 240, 216], # Water Spinach
            [123, 237, 0], # Watermelon
            [55, 184, 4], # Greens
            [74, 250, 5], # Trees 
            [30, 92, 20], # Grass
            [247, 5, 5], # Red roof
            [209, 182, 195], # Grey Roof
            [224, 118, 72], # Plastic
            [135, 60, 27], # Bare Soil
            [255, 255, 255], # Road
            [194, 93, 187], # Bright object
            [13, 22, 209] # Water
        ]
    elif dataset == 'HongHu':
        colors = [
            [247, 5, 5], # Red roof
            [255, 255, 255], # Road 
            [143, 30, 71], # Bare Soil 
            [255, 255, 0], # Cotton 
            [224, 118, 72], # Cotton firewood 
            [74, 250, 5], # Rape 
            [55, 184, 4], # Chinese Cabbage
            [30, 92, 20], # Pakchoi
            [132, 240, 216], # Cabbage
            [137, 44, 168], # Tuber Mustard
            [209, 182, 195], # Brassica parachinensis
            [13, 22, 209], # Brassica chinensis
            [17, 4, 112], # Small brassica chinensis
            [194, 93, 187], # Lactuca Sativa 
            [135, 60, 27], # Celtuce 
            [7, 247, 227], # Film covered lettuce 
            [250, 167, 2], # Romaine lettuce
            [123, 237, 0], # Carrot
            [106, 115, 7], # White radish
            [2, 113, 117], # Garlic sprout
            [209, 182, 195], # Broad bean
            [214, 143, 2] # Tree
        ]
    elif dataset == 'LongKou':
       colors = [
            [255, 0, 0], # Corn
            [255, 165, 0], # Cotton
            [255, 255, 0], # Sesame
            [0, 255, 0], # Broad-leaf soybean
            [0, 255, 255], # Narrow-leaf soybean
            [0, 128, 128], # Rice
            [0, 0, 255], # Water
            [255, 255, 255], # Roads and houses
            [128, 0, 128] # Mixed weed
    ]

    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        item = int(item)  # Ensure item is an integer for indexing
        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.
        else:
            y[index] = np.array(colors[item - 1]) / 255.
    
    return y

def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1]*2.0/dpi, ground_truth.shape[0]*2.0/dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)

    return 0

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

def get_cls_map(net, device, test_loader, y, dataset):
    y_pred, y_new = test(device, net, test_loader)

    print(f"Shape of y_pred: {y_pred.shape}")
    print(f"Shape of y: {y.shape}")
    print(f"Type of y: {type(y)}")

    cls_labels = get_classification_map(y_pred, y)
    print(f"Shape of cls_labels: {cls_labels.shape}")

    x = np.ravel(cls_labels)
    gt = y.flatten()

    y_list = list_to_colormap(x, dataset)
    y_gt = list_to_colormap(gt, dataset)

    y_re = np.reshape(y_list, (y.shape[0], y.shape[1], 3))
    gt_re = np.reshape(y_gt, (y.shape[0], y.shape[1], 3))

    classification_map(y_re, y, 300, f'{main_dir}/SSFTT/classification_maps/{dataset}_predictions.eps')
    classification_map(y_re, y, 300, f'{main_dir}/SSFTT/classification_maps/{dataset}_predictions.png')
    classification_map(gt_re, y, 300, f'{main_dir}/SSFTT/classification_maps/{dataset}_gt.png')
    print('------Get classification maps successful-------')

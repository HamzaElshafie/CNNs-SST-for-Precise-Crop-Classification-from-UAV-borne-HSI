import numpy as np
import matplotlib.pyplot as plt

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
            [147, 67, 46], [0, 255, 123], [0, 0, 255], [255, 100, 0], [0, 255, 123], [255, 0, 0], 
            [0, 255, 0], [0, 255, 255], [128, 128, 128], [255, 255, 0], [0, 128, 255], [0, 0, 128],
            [128, 0, 128], [0, 128, 0], [128, 128, 0], [0, 0, 0]
        ]
    elif dataset == 'HongHu':
        colors = [
            [255, 0, 0], [0, 0, 0], [139, 69, 19], [255, 255, 0], [169, 169, 169], [255, 165, 0],
            [0, 255, 0], [0, 128, 0], [0, 255, 255], [0, 0, 255], [255, 20, 147], [255, 105, 180],
            [128, 0, 128], [75, 0, 130], [238, 130, 238], [255, 69, 0], [255, 228, 225], [255, 222, 173],
            [240, 230, 140], [128, 0, 0], [189, 183, 107], [255, 218, 185]  
        ]
    elif dataset == 'LongKou':
       colors = [
            [255, 0, 0],       # Corn
            [255, 165, 0],     # Cotton
            [255, 255, 0],     # Sesame
            [0, 255, 0],       # Broad-leaf soybean
            [0, 255, 255],     # Narrow-leaf soybean
            [0, 128, 128],     # Rice
            [0, 0, 255],       # Water
            [255, 255, 255],   # Roads and houses
            [128, 0, 128]      # Mixed weed
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

def get_cls_map(net, device, all_data_loader, y, dataset):
    y_pred, y_new = test(device, net, all_data_loader)
    cls_labels = get_classification_map(y_pred, y)
    x = np.ravel(cls_labels)
    gt = y.flatten()

    y_list = list_to_colormap(x, dataset)
    y_gt = list_to_colormap(gt, dataset)

    y_re = np.reshape(y_list, (y.shape[0], y.shape[1], 3))
    gt_re = np.reshape(y_gt, (y.shape[0], y.shape[1], 3))

    classification_map(y_re, y, 300, f'classification_maps/{dataset}_predictions.eps')
    classification_map(y_re, y, 300, f'classification_maps/{dataset}_predictions.png')
    classification_map(gt_re, y, 300, f'classification_maps/{dataset}_gt.png')
    print('------Get classification maps successful-------')

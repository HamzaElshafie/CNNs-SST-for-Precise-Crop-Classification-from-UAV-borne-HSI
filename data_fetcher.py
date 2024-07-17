import os
import zipfile
import scipy.io as sio
from kaggle.api.kaggle_api_extended import KaggleApi

def download_specific_dataset_from_kaggle(base_path, kaggle_dataset, kaggle_json_path, dataset):
    # Set up Kaggle API credentials
    os.environ['KAGGLE_CONFIG_DIR'] = kaggle_json_path
    os.makedirs(base_path, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    dataset_folder = f'{base_path}/{dataset}'
    if not os.path.exists(dataset_folder):
        try:
            # Download the specific dataset folder as a zip file
            api.dataset_download_files(kaggle_dataset, path=base_path, unzip=False)
            zip_file_path = os.path.join(base_path, kaggle_dataset.split('/')[-1] + '.zip')
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                for file in zip_ref.namelist():
                    if file.startswith(dataset):
                        zip_ref.extract(file, base_path)
            os.remove(zip_file_path)
            print(f"Downloaded and extracted the dataset {dataset} to {base_path}")
        except Exception as e:
            print(f"Failed to download dataset {dataset}: {e}")
            return None
    else:
        print(f"Dataset {dataset_folder} already exists, skipping download.")

    return dataset_folder

def loadData(dataset, kaggle_json_path, base_path):
    if base_path is None:
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data'))
    os.makedirs(base_path, exist_ok=True)

    kaggle_dataset = 'rupeshkumaryadav/whu-hyperspectral-dataset'

    dataset_path = download_specific_dataset_from_kaggle(base_path, kaggle_dataset, kaggle_json_path, dataset)
    if not dataset_path:
        print("Failed to download the dataset.")
        return None, None
    
    data_file = f'{dataset}/{dataset.replace("-", "_")}.mat'
    label_file = f'{dataset}/{dataset.replace("-", "_")}_gt.mat'

    data_path = os.path.join(base_path, data_file)
    labels_path = os.path.join(base_path, label_file)

    if not os.path.exists(data_path) or not os.path.exists(labels_path):
        print(f"Data file or label file not found in {dataset_path}")
        return None, None

    print(f'Loading data from {data_path}')
    print(f'Loading labels from {labels_path}')

    try:
        data_mat = sio.loadmat(data_path)
        labels_mat = sio.loadmat(labels_path)

        print('Keys in data_mat:', list(data_mat.keys()))
        print('Keys in labels_mat:', list(labels_mat.keys()))

        # Check for the dataset in the loaded .mat files
        data_key = dataset.replace("-", "_")
        label_key = f'{data_key}_gt'

        # Debugging: print the available keys to understand the structure
        print(f"Available keys in data_mat: {list(data_mat.keys())}")
        print(f"Available keys in labels_mat: {list(labels_mat.keys())}")

        if data_key in data_mat and label_key in labels_mat:
            data = data_mat[data_key]
            labels = labels_mat[label_key]
        else:
            raise ValueError(f"Expected keys '{data_key}' and '{label_key}' not found in the .mat files. Available keys: {list(data_mat.keys())}, {list(labels_mat.keys())}")

        return data, labels

    except Exception as e:
        print(f"An error occurred while loading .mat files: {e}")
        return None, None
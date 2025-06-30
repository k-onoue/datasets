# Dataset loader module for CSV splits
import os
import numpy as np


class DatasetLoader:

    def __init__(self, base_path):
        """
        Initializes the DatasetLoader with the base path where datasets are stored.
        
        Parameters:
            base_path (str): Root path where datasets/ lives.
        """
        self.base_path = base_path

    def list_datasets(self):
        """
        Returns a list of dataset names under the base_path directory.
        """
        return [name for name in os.listdir(self.base_path)
                if os.path.isdir(os.path.join(self.base_path, name))]

    def list_splits(self, dataset_name):
        """
        Returns a sorted list of split indices available for the given dataset.
        """
        ds_path = os.path.join(self.base_path, dataset_name)
        splits = []
        for name in os.listdir(ds_path):
            if name.startswith("split_"):
                try:
                    idx = int(name.split("_", 1)[1])
                    splits.append(idx)
                except ValueError:
                    pass
        return sorted(splits)

    def load_split(self, dataset_name, split_index):
        """
        Loads train/test features and targets for a specific split.
        
        Parameters:
            base_path (str): Root path where datasets/ lives.
            dataset_name (str): Name of the dataset folder.
            split_index (int): Split number (e.g., 0, 1, ...).
        
        Returns:
            X_train (ndarray), y_train (ndarray), X_test (ndarray), y_test (ndarray)
        """
        split_dir = os.path.join(self.base_path, dataset_name, f"split_{split_index}")
        X_train = np.loadtxt(os.path.join(split_dir, "train_features.csv"), delimiter=",")
        y_train = np.loadtxt(os.path.join(split_dir, "train_target.csv"),  delimiter=",")
        X_test  = np.loadtxt(os.path.join(split_dir, "test_features.csv"),  delimiter=",")
        y_test  = np.loadtxt(os.path.join(split_dir, "test_target.csv"),   delimiter=",")
        return X_train, y_train, X_test, y_test



if __name__ == "__main__":

    base = "./../datasets"

    loader = DatasetLoader(base)

    # 利用可能なデータセット
    print(loader.list_datasets())
    # -> ['Bike', 'Concrete', 'Diabetes', ...]

    # 'Bike' データセットの利用可能な分割
    print(loader.list_splits("Bike"))
    # -> [0, 1, 2, ..., 9]

    # split 0 をロード
    X_tr, y_tr, X_te, y_te = loader.load_split("Bike", 0)
    print(X_tr.shape, y_tr.shape, X_te.shape, y_te.shape)

# preprocessing.py
"""
Data preprocessing functions for NIR data analysis.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset


class NIRDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)  # Ensure y is 2D

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_and_preprocess_data(dataset_name, test_size=0.2, random_seed=42):
    """
    Load and preprocess the specified dataset.

    Args:
        dataset_name (str): Name of the dataset
        test_size (float): Test split ratio
        random_seed (int): Random seed for reproducibility

    Returns:
        tuple: X_train, X_test, y_train, y_test, X_scaled, y, input_size
    """
    if dataset_name == "Cal_ManufacturerB":
        data = pd.read_excel("data/Cal_ManufacturerB.xlsx")
        y = data['Protein'].values
        X = data.drop(['ID', 'Protein'], axis=1).values

    elif dataset_name == "Ramandata_tablets":
        data = pd.read_excel("data/Ramandata_tablets.xlsx")
        y = data['PE'].values
        X = data.drop(['PE'], axis=1).values

    elif dataset_name == "高光谱1":
        data = pd.read_excel("data/高光谱1.xlsx", index_col=0)
        y = data.iloc[:, 0].values
        X = data.iloc[:, 1:].values
        y = np.log2(y + 1)  # Log transform

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_seed
    )

    input_size = X_train.shape[1]

    return X_train, X_test, y_train, y_test, X_scaled, y, input_size


def create_data_loaders(X_train, X_test, y_train, y_test, X_all, y_all,
                        batch_size=64, test_batch_size=128):
    """
    Create PyTorch DataLoaders for training and testing.

    Args:
        X_train, X_test, y_train, y_test: Training and test data
        X_all, y_all: All data combined
        batch_size (int): Batch size for training
        test_batch_size (int): Batch size for testing

    Returns:
        tuple: train_loader, test_loader, all_loader
    """
    train_dataset = NIRDataset(X_train, y_train)
    test_dataset = NIRDataset(X_test, y_test)
    all_dataset = NIRDataset(X_all, y_all)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, all_loader


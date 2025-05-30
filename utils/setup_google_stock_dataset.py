import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def prepare_stock_dataset(
    df,
    feature_cols=['Close/Last', 'Open', 'High', 'Low', 'Volume'],
    target_col='Close/Last',
    date_col='Date',
    sequence_length=30,
    batch_size=32,
    train_ratio=0.7
):
    """
    Prepares a stock price dataset with normalization and sequence formatting for time series forecasting.
    Splits the dataset into train and test (no validation set).

    Returns:
        dict: {
            'train_loader': DataLoader,
            'test_loader': DataLoader,
            'train_size': int,
            'test_size': int,
            'min_max': dict,
            'dates': list of dates,
            'X': torch.Tensor,
            'Y': torch.Tensor
        }
    """
    df = df.copy()

    # Clean and normalize price columns
    for col in ['Close/Last', 'Open', 'High', 'Low']:
        df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)

    df[date_col] = pd.to_datetime(df[date_col])

    # Min-max normalization
    min_max = {}
    for col in feature_cols:
        min_val, max_val = df[col].min(), df[col].max()
        min_max[col] = (min_val, max_val)
        df[col] = (df[col] - min_val) / (max_val - min_val)

    # Create input/output sequences
    data = df[feature_cols].values.astype(np.float32)
    X, Y, dates = [], [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        Y.append(data[i+sequence_length][feature_cols.index(target_col)])
        dates.append(df[date_col].iloc[i + sequence_length])

    X = torch.tensor(X)  # (N, sequence_length, num_features)
    Y = torch.tensor(Y).unsqueeze(-1)  # (N, 1)

    # Split: 70% train, 30% test
    total_size = len(X)
    train_size = int(total_size * train_ratio)
    test_size = total_size - train_size

    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]

    # Dataloaders
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size)

    return {
        'train_loader': train_loader,
        'test_loader': test_loader,
        'train_size': train_size,
        'test_size': test_size,
        'min_max': min_max,
        'dates': dates,
        'X': X,
        'Y': Y,
    }


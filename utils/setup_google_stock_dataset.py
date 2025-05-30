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
    train_ratio=0.7,
    reverse=False
):
    """
    Prepares a stock price dataset with normalization and sequence formatting for time series forecasting.
    Optionally applies a reverse AFTER splitting train/test to ensure correct alignment.

    Args:
        df (pd.DataFrame): The input dataframe.
        feature_cols (list): List of features to use.
        target_col (str): The target column to predict.
        date_col (str): Column containing dates.
        sequence_length (int): Length of the input sequence.
        batch_size (int): Batch size for DataLoader.
        train_ratio (float): Ratio of train/test split.
        reverse (bool): Whether to reverse the train/test sets after splitting.

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

    # Apply reverse if needed
    if reverse:
        # Reverse only train and test separately (to keep alignment)
        X_train = X[:train_size].flip(dims=[0])
        Y_train = Y[:train_size].flip(dims=[0])
        X_test = X[train_size:].flip(dims=[0])
        Y_test = Y[train_size:].flip(dims=[0])
        # Also reverse the corresponding dates
        dates_train = list(reversed(dates[:train_size]))
        dates_test = list(reversed(dates[train_size:]))
        dates = dates_train + dates_test
    else:
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

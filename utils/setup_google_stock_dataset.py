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
    df = df.copy()

    # Nettoyage et conversion
    for col in ['Close/Last', 'Open', 'High', 'Low']:
        df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)

    df[date_col] = pd.to_datetime(df[date_col])

    # Normalisation min-max
    min_max = {}
    for col in feature_cols:
        min_val, max_val = df[col].min(), df[col].max()
        min_max[col] = (min_val, max_val)
        df[col] = (df[col] - min_val) / (max_val - min_val)

    data = df[feature_cols].values.astype(np.float32)
    dates_all = df[date_col].tolist()

    X, Y, dates = [], [], []

    if not reverse:
        # Mode standard : X = [t-30, ..., t-1], Y = t
        for i in range(len(data) - sequence_length):
            X.append(data[i:i+sequence_length])
            Y.append(data[i+sequence_length][feature_cols.index(target_col)])
            dates.append(dates_all[i+sequence_length])
    else:
        # Mode reverse : X = [t-1, ..., t-30], Y = t-31
        for i in range(sequence_length, len(data)):
            seq = data[i-sequence_length:i][::-1]  # inverse temporel
            target_idx = i - sequence_length - 1
            if target_idx >= 0:
                X.append(seq)
                Y.append(data[target_idx][feature_cols.index(target_col)])
                dates.append(dates_all[target_idx])
            else:
                continue  # on ignore les tout premiers indices

    X = torch.tensor(np.array(X))  # (N, seq_len, features)
    Y = torch.tensor(np.array(Y)).unsqueeze(-1)  # (N, 1)

    total_size = len(X)
    train_size = int(total_size * train_ratio)

    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]

    dates_train = dates[:train_size]
    dates_test = dates[train_size:]

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True,drop_last=True)
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size,drop_last=True)

    return {
        'train_loader': train_loader,
        'test_loader': test_loader,
        'train_size': train_size,
        'test_size': len(X) - train_size,
        'min_max': min_max,
        'dates': dates_train + dates_test,
        'X': X,
        'Y': Y
    }

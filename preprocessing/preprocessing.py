import torch
import scipy.io as sio
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from torch.utils.data import random_split, DataLoader

def data_preprocessing(sequence_length, num_shift=1, batch_size=32, shuffle=True, train_ratio=0.8):
    file_path = './timeseries.mat'
    data = sio.loadmat(file_path)['data'].T

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    data = torch.from_numpy(data).float()

    num_points = data.shape[0]
    inputs, targets = [], []

    for p in range(0, num_points - sequence_length, num_shift):
        inputs.append(data[p: p + sequence_length, :])
        targets.append(data[p + sequence_length, :])

    inputs = torch.cat([x.unsqueeze(0) for x in inputs], dim=0)
    targets = torch.cat([x.unsqueeze(0) for x in targets], dim=0)

    dataset = TensorDataset(inputs, targets)

    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

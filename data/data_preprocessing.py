import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import scipy.io as sio
from torch.utils.data import TensorDataset

def preprocess_data(sequence_length):
    data = sio.loadmat('data directory')
    data = np.array(data["data"]).transpose()

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    inputs, targets = [], []
    for i in range(len(data) - sequence_length):
        inputs.append(data[i : i + sequence_length])
        targets.append(data[i + sequence_length])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)
    dataset = TensorDataset(inputs, targets)

    return dataset
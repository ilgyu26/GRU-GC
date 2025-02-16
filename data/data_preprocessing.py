import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import scipy.io as sio
from torch.utils.data import TensorDataset

def preprocess_data(sequence_length):
    data = sio.loadmat('data directory')
    data = np.array(data["data"]).transpose()

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    inputs, targets = [], []
    for i in range(len(data) - sequence_length):
        inputs.append(data[i : i + sequence_length])
        targets.append(data[i + sequence_length])
    inputs = torch.tensor(np.array(inputs), dtype=torch.float32)
    targets = torch.tensor(np.array(targets), dtype=torch.float32)
    dataset = TensorDataset(inputs, targets)

    return dataset
import numpy as np
import copy
import torch
from torch.utils.data import TensorDataset, DataLoader

from models.custom_gru import CustomGRU
from models.custom_gru import train_model

class GRU_GC(object):
    def __init__(self):
        # model options
        self.num_channel = 5
        self.hidden_size = 32
        self.output_size = 1
        self.num_layers = 1
        self.dropout = 0.5

        # training options
        self.num_epochs = 10
        self.learning_rate = 0.001
        self.weight_decay = 0.001
        self.batch_size = 32
        self.sequence_length = 20

        # nue options
        self.theta = 0.09
    
    def gru_gc(self, dataset):
        data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        x, y = next(iter(data_loader))

        granger_matrix = np.zeros([self.num_channel, self.num_channel])
        var_denominator = np.zeros([1, self.num_channel])

        for k in range(self.num_channel):
            tmp_y = np.reshape(y[:, k], [y.shape[0], 1])
            channel_set = list(range(self.num_channel))
            input_set = []
            last_error = 0

            for i in range(self.num_channel):
                min_error = 1e7
                min_idx = 0
                for x_idx in channel_set:
                    candidate_set = copy.copy(input_set)
                    candidate_set.append(x_idx)
                    tmp_x = x[:, :, candidate_set]

                    print(f"Training a model to predict channel {k} using channels {candidate_set}...")

                    tmp_dataset = TensorDataset(tmp_x, tmp_y)
                    train_loader = DataLoader(tmp_dataset, batch_size=self.batch_size, shuffle=True)

                    gru = CustomGRU(len(candidate_set), self.hidden_size, self.output_size, self.num_layers, self.dropout)
                    tmp_error = train_model(gru, train_loader, self.learning_rate, self.weight_decay, self.num_epochs)
                    if tmp_error < min_error:
                        min_error = tmp_error
                        min_idx = x_idx
                if i != 0 and (np.abs(last_error - min_error) / last_error < self.theta or last_error < min_error):
                    break
                input_set.append(min_idx)
                channel_set.remove(min_idx)
                last_error = min_error
                
            gru.eval()

            with torch.no_grad():
                pred = gru(x[:, :, input_set].clone().detach()).detach().cpu().numpy()
                target = tmp_y.detach().cpu().numpy() if isinstance(tmp_y, torch.Tensor) else np.array(tmp_y)
                var_denominator[0][k] = np.var(pred - target, axis=0)

            for j in range(self.num_channel):
                with torch.no_grad(): 
                    if j not in input_set:
                        granger_matrix[j][k] = var_denominator[0][k]
                    elif len(input_set) == 1:
                        pred = gru(x[:, :, k].unsqueeze(-1).clone().detach()).detach().cpu().numpy()
                        target = tmp_y.detach().cpu().numpy() if isinstance(tmp_y, torch.Tensor) else np.array(tmp_y)
                        granger_matrix[j][k] = np.var(pred - target, axis=0)
                    else:
                        tmp_x = x[:, :, input_set].clone().detach()
                        tmp_x[..., input_set.index(j)] = 0
                        pred = gru(tmp_x).detach().cpu().numpy()
                        target = tmp_y.detach().cpu().numpy() if isinstance(tmp_y, torch.Tensor) else np.array(tmp_y)
                        granger_matrix[j][k] = np.var(pred - target, axis=0)
            '''            
            gru.eval()
            var_denominator[0][k] = np.var(gru(x[:, :, input_set]) - tmp_y, axis = 0)
            for j in range(self.num_channel):
                if j not in input_set:
                    granger_matrix[j][k] = var_denominator[0][k]
                elif len(input_set) == 1:
                    tmp_x = x[:, :, k]
                    tmp_x = tmp_x[:, :, np.newaxis]
                    granger_matrix[j][k] = np.var(gru(tmp_x) - tmp_y, axis=0)
                else:
                    tmp_x = x[:, :, input_set]
                    channel_del_idx = input_set.index(j)
                    tmp_x[:, :, channel_del_idx] = 0
                    granger_matrix[j][k] = np.var(gru(tmp_x) - tmp_y, axis=0)
            '''
        granger_matrix = granger_matrix / var_denominator
        for i in range(self.num_channel):
            granger_matrix[i][i] = 1
        granger_matrix[granger_matrix < 1] = 1
        granger_matrix = np.log(granger_matrix)

        return granger_matrix

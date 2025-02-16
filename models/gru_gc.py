import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from models.custom_gru import CustomGRU
from models.custom_gru import train_model

class GRU_GC(object):
    def __init__(self):
        # model options
        self.num_channel = 5
        self.hidden_size = 30
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

        # GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def gru_gc(self, dataset):
        x, y = dataset.tensors
        granger_matrix = np.zeros((self.num_channel, self.num_channel))
        var_denominator = np.zeros(self.num_channel)

        for k in range(self.num_channel):
            tmp_y = y[:, k].unsqueeze(1)
            tmp_y = tmp_y.to(self.device)
            channel_set = list(range(self.num_channel))
            input_set = []
            last_error = float("inf")

            for _ in range(self.num_channel):
                min_error, min_idx = float("inf"), None
                for x_idx in channel_set:
                    candidate_set = input_set + [x_idx]
                    tmp_x = x[:, :, candidate_set]
                    tmp_x = tmp_x.to(self.device)
                    train_loader = DataLoader(TensorDataset(tmp_x, tmp_y), batch_size=self.batch_size, shuffle=True)

                    print(f"Training a model to predict channel {k} using channels {candidate_set}...")

                    model = CustomGRU(len(candidate_set), self.hidden_size, self.output_size, self.num_layers, self.dropout).to(self.device)
                    error = train_model(model, train_loader, self.learning_rate, self.weight_decay, self.num_epochs, self.device)

                    if error < min_error:
                        min_error, min_idx = error, x_idx
                
                if len(input_set) > 0 and (abs(last_error - min_error) / last_error < self.theta or last_error < min_error):
                    break

                input_set.append(min_idx)
                channel_set.remove(min_idx)
                last_error = min_error

            print(f"The model predicting the channel {k} uses {input_set}")

            tmp_x = x[:, :, input_set].to(self.device)
            train_loader = DataLoader(TensorDataset(tmp_x, tmp_y), batch_size=self.batch_size, shuffle=True)
            model = CustomGRU(len(input_set), self.hidden_size, self.output_size, self.num_layers, self.dropout).to(self.device)
            train_model(model, train_loader, self.learning_rate, self.weight_decay, self.num_epochs, self.device)
                   
            model.eval()
            with torch.no_grad():
                var_denominator[k] = torch.var(model(tmp_x) - tmp_y, unbiased=False).item()
                for j in range(self.num_channel):
                    if j not in input_set:
                        granger_matrix[j, k] = var_denominator[k]
                    elif len(input_set) == 1:
                        tmp_x = x[:, :, k].unsqueeze(-1).to(self.device)
                        granger_matrix[j, k] = torch.var(model(tmp_x) - tmp_y, unbiased=False).item()
                    else:
                        tmp_x[:, :, input_set.index(j)] = 0
                        granger_matrix[j, k] = torch.var(model(tmp_x) - tmp_y, unbiased=False).item()

                granger_matrix[:, k] /= var_denominator[k]
                granger_matrix[k, k] = 1
                granger_matrix[:, k] = np.log(np.maximum(granger_matrix[:, k], 1))

                print(f"Updated Granger matrix column {k}: \n{granger_matrix[:, k]}\n")

        print(f"Granger Matrix: \n{granger_matrix}")

        return granger_matrix
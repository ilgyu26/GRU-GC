import torch.nn as nn
import torch.optim as optim

class CustomGRU(nn.Module):
    def __init__(self, num_channel, hidden_size, output_size, num_layers, dropout):
        super(CustomGRU, self).__init__()

        self.gru = nn.GRU(input_size=num_channel, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        output = gru_out[:, -1, :]
        output = self.fc(self.dropout(output))
        return output

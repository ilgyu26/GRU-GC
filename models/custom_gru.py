import torch.nn as nn
import torch.optim as optim

class CustomGRU(nn.Module):
    def __init__(self, num_channel, hidden_size, output_size, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(input_size=num_channel, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, h = self.gru(x)
        out = self.dropout(h[-1])
        out = self.fc(out)
        return out

def train_model(model, train_loader, learning_rate, weight_decay, num_epochs, device):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0 
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return avg_loss

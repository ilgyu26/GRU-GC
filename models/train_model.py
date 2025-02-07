import torch
import torch.optim as optim

class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float("inf")
        self.num_bad_epochs = 0
        self.stop_training = False
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            self.stop_training = True
        
        return self.stop_training
    
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, patience=10, delta=0):
    early_stopping = EarlyStopping(patience=patience, delta=delta)
    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        running_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if early_stopping(val_loss):
            break

import argparse

class GRUOptions:
    def __init__(self):
        self.num_channel = 10
        self.hidden_size = 64
        self.output_size = 1
        self.num_layers = 2
        self.dropout = 0.2
        self.sequence_length = 20
        self.batch_size = 32

    def parse(self):
        parser = argparse.ArgumentParser(description="GRU Model Options")

        parser.add_argument("--num_channel", type=int, default=self.num_channel, help="Number of input channels")
        parser.add_argument("--hidden_size", type=int, default=self.hidden_size, help="Hidden layer size")
        parser.add_argument("--output_size", type=int, default=self.output_size, help="Output size")
        parser.add_argument("--num_layers", type=int, default=self.num_layers, help="Number of GRU layers")
        parser.add_argument("--dropout", type=float, default=self.dropout, help="Dropout rate")
        parser.add_argument("--sequence_length", type=int, default=self.sequence_length, help="Sequence length for input")
        parser.add_argument("--batch_size", type=int, default=self.batch_size, help="Batch size")

        return parser.parse_args()


class TrainingOptions:
    def __init__(self):
        self.num_epochs = 50
        self.patience = 5
        self.delta = 0.01
        self.weight_decay = 1e-4
        self.theta = 0.01

    def parse(self):
        parser = argparse.ArgumentParser(description="Training Options")

        parser.add_argument("--num_epochs", type=int, default=self.num_epochs, help="Number of training epochs")
        parser.add_argument("--patience", type=int, default=self.patience, help="Early stopping patience")
        parser.add_argument("--delta", type=float, default=self.delta, help="Minimum change for early stopping")
        parser.add_argument("--weight_decay", type=float, default=self.weight_decay, help="Weight decay for optimizer")
        parser.add_argument("--theta", type=float, default=self.theta, help="Threshold for feature selection")

        return parser.parse_args()


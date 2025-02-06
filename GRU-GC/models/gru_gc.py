import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import datetime
import copy
from torch.utils.data import DataLoader
from models.custom_gru import CustomGRU
from models.train_model import train
from preprocessing.preprocessing import data_preprocessing

class GRU_GC(object):
    def __init__(self, gru_opt, training_opt):
        self.num_channel = gru_opt.num_channel
        self.hidden_size = gru_opt.hidden_size
        self.output_size = gru_opt.output_size
        self.num_layers = gru_opt.num_layers
        self.dropout = gru_opt.dropout
        self.sequence_length = gru_opt.sequence_length
        self.batch_size = gru_opt.batch_size
        self.weight_decay = getattr(training_opt, "weight_decay", 1e-4)  # 기본값 추가
        self.theta = getattr(training_opt, "theta", 0.01)  # 기본값 추가
        self.num_epochs = training_opt.num_epochs
        self.patience = training_opt.patience
        self.delta = training_opt.delta

    def nue(self, train_loader, val_loader):
        granger_matrix = np.zeros([self.num_channel, self.num_channel])
        var_denominator = np.zeros([1, self.num_channel])
        all_candidate = []
        error_model = []
        error_all = []

        hist_result = []
        start_time = datetime.datetime.now()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for k in range(self.num_channel):
            channel_set = list(range(self.num_channel))
            input_set = []
            last_error = float("inf")

            for i in range(self.num_channel):
                min_error = float("inf")
                min_idx = 0

                for x_idx in channel_set:
                    tmp_set = copy.copy(input_set)
                    tmp_set.append(x_idx)

                    # 모델 생성
                    model = CustomGRU(
                        num_channel=len(tmp_set),
                        hidden_size=self.hidden_size,
                        output_size=1,
                        num_layers=self.num_layers,
                        dropout=self.dropout
                    ).to(device)

                    criterion = nn.MSELoss()
                    optimizer = torch.optim.Adam(model.parameters(), weight_decay=self.weight_decay)

                    # 훈련
                    train(model, train_loader, val_loader, criterion, optimizer,
                          num_epochs=self.num_epochs, patience=self.patience, delta=self.delta)

                    # 검증 데이터로 손실 계산
                    model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for inputs, targets in val_loader:
                            inputs, targets = inputs.to(device), targets.to(device)
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            val_loss += loss.item() * inputs.size(0)

                    val_loss /= len(val_loader.dataset)

                    if val_loss < min_error:
                        min_error = val_loss
                        min_idx = x_idx

                    error_all.append([k, i, x_idx, val_loss])

                error_model.append([k, last_error, min_error])

                if i != 0 and (abs(last_error - min_error) / last_error < self.theta or last_error < min_error):
                    break

                input_set.append(min_idx)
                channel_set.remove(min_idx)
                last_error = min_error

            all_candidate.append(input_set)

            # 최종 모델 학습
            model = CustomGRU(
                num_channel=len(input_set),
                hidden_size=self.hidden_size,
                output_size=1,
                num_layers=self.num_layers,
                dropout=self.dropout
            ).to(device)

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), weight_decay=self.weight_decay)

            hist_res = train(model, train_loader, val_loader, criterion, optimizer,
                             num_epochs=self.num_epochs, patience=self.patience, delta=self.delta)
            hist_result.append(hist_res)

            # Granger Matrix 값 계산
            model.eval()
            with torch.no_grad():
                predictions = []
                true_values = []
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    predictions.append(outputs.cpu().numpy())
                    true_values.append(targets.cpu().numpy())

            predictions = np.concatenate(predictions, axis=0)
            true_values = np.concatenate(true_values, axis=0)

            var_denominator[0][k] = np.var(predictions - true_values, axis=0)

            for j in range(self.num_channel):
                if j not in input_set:
                    granger_matrix[j][k] = var_denominator[0][k]
                elif len(input_set) == 1:
                    granger_matrix[j][k] = np.var(predictions - true_values, axis=0)
                else:
                    tmp_x = inputs.clone()
                    channel_del_idx = input_set.index(j)
                    tmp_x[:, :, channel_del_idx] = 0  # 해당 채널의 값을 0으로 설정
                    outputs = model(tmp_x.to(device)).cpu().numpy()
                    granger_matrix[j][k] = np.var(outputs - true_values, axis=0)

            print(f'Trained model for output {k + 1}')

        # Granger Matrix 정규화 및 로그 변환
        granger_matrix = granger_matrix / var_denominator
        np.fill_diagonal(granger_matrix, 1)
        granger_matrix[granger_matrix < 1] = 1
        granger_matrix = np.log(granger_matrix)

        end_time = datetime.datetime.now()
        print(f'Training time: {(end_time - start_time).seconds} seconds')

        return granger_matrix

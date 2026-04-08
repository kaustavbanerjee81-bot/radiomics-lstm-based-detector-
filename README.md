# ================================
# Necessary Imports
# ================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# ================================
# Load Data
# ================================
data = pd.read_excel("/content/sample_data/patients_with_all_timepoints.xlsx")

# ================================
# Preprocessing
# ================================
data = data.dropna(how='all')

# Drop first 2 columns (non-feature columns)
dropped_data = data.drop(data.columns[0:2], axis=1)

# Normalize data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(dropped_data)


# ================================
# Sequence Construction
# ================================
data_list = []

rows, cols = normalized_data.shape
block_size = 4

for start_row in range(0, rows, block_size):
    end_row = min(start_row + block_size, rows)
    for col in range(cols):
        sublist = normalized_data[start_row:end_row, col].tolist()
        data_list.append(sublist)


# ================================
# Training Sequences
# ================================
training_sequence = []

for i in data_list:
    for j in range(1, len(i)):
        training_sequence.append(i[:j+1])

# Padding (front padding with zeros)
max_len = max(len(seq) for seq in training_sequence)

padded_sequences = []
for seq in training_sequence:
    pad_len = max_len - len(seq)
    padded_seq = [0]*pad_len + seq
    padded_sequences.append(padded_seq)

padded_training_sequence = torch.tensor(padded_sequences, dtype=torch.float32)


# ================================
# X and Y Formation
# ================================
X = padded_training_sequence[:, :-1]
Y = padded_training_sequence[:, -1]

X = X.unsqueeze(-1)


# ================================
# Custom Dataset
# ================================
class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X.float()
        self.Y = Y.float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


dataset = CustomDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)


# ================================
# LSTM Model
# ================================
class LSTMmodel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        output = self.fc(last_hidden)
        return output.squeeze(1)


model = LSTMmodel()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# ================================
# Training Setup
# ================================
epochs = 50
lr = 0.001

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# ================================
# Training Loop
# ================================
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch_X, batch_Y in dataloader:
        batch_X = batch_X.to(device)
        batch_Y = batch_Y.to(device)

        predictions = model(batch_X)
        loss = criterion(predictions, batch_Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss:.6f}")


# ================================
# Evaluation
# ================================
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for batch_X, batch_Y in dataloader:
        batch_X = batch_X.to(device)
        batch_Y = batch_Y.to(device)

        predictions = model(batch_X)

        all_preds.extend(predictions.cpu().numpy())
        all_targets.extend(batch_Y.cpu().numpy())

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

n_cols = scaler.n_features_in_
col_index = 0


def inverse_col(values, scaler, col_index, n_cols):
    dummy = np.zeros((len(values), n_cols))
    dummy[:, col_index] = values
    return scaler.inverse_transform(dummy)[:, col_index]


preds_original = inverse_col(all_preds, scaler, col_index, n_cols)
targets_original = inverse_col(all_targets, scaler, col_index, n_cols)

mse = mean_squared_error(targets_original, preds_original)
rmse = np.sqrt(mse)
mae = mean_absolute_error(targets_original, preds_original)
r2 = r2_score(targets_original, preds_original)

print("\nMODEL PERFORMANCE")
print(f"MSE  : {mse:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"MAE  : {mae:.6f}")
print(f"R2   : {r2:.6f}")

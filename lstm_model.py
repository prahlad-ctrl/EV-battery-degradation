import torch
from torch import nn

class BatteryLSTM(nn.Module):
    def __init__(self, input_size = 5, hidden_size1 = 128, hidden_size2 = 64, dense = 32, dropout = 0.2):
        super(BatteryLSTM, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size = input_size, hidden_size = hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        self.lstm2 = nn.LSTM(input_size = hidden_size1, hidden_size = hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(hidden_size2, dense)
        self.relu = nn.ReLU()
        
        self.fc2 = nn.Linear(dense, 1)
    
    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out = out[:, -1, :] # only pred
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out
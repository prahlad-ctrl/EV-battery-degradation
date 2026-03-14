import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from lstm_model import BatteryLSTM
import os

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_train = torch.load('data/processed_data/X_train.pt')
    y_train = torch.load('data/processed_data/y_train.pt')
    X_test = torch.load('data/processed_data/X_test.pt')
    y_test = torch.load('data/processed_data/y_test.pt')
    
    batch_size = 32
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = BatteryLSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 50
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss = train_loss/ len(train_loader.dataset)
        
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item() * X_batch.size(0)
            
        test_loss = test_loss/ len(test_loader.dataset)
        
        if (epoch+1) % 5 == 0:
            print(f'epoch: {epoch+1}/{epochs}, train loss: {train_loss:.4f}, test loss: {test_loss:.4f}')
        
    
    torch.save(model.state_dict(), 'models/battery_lstm.pth')
    
if __name__ == "__main__":
    train_model()
    

'''
got a pretty good model IMO (train loss: 0.0111, test loss: 0.0123)
'''
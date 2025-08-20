import pandas as pd
from sklearn import preprocessing
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import joblib
#from convlstm import ConvLSTM
from tqdm import tqdm
from models.convlstm import ConvLSTM, ConvLSTMCell, ConvLSTMTimeSeries, TimeSeries3DDataset

def main():
    weather_df = pd.read_csv('data/weather_data.csv', index_col=0, parse_dates=True)
    air_df = pd.read_csv('data/air_data.csv', index_col=0, parse_dates=True)
    air_df.drop(columns='aqi', inplace=True)

    weather_df["wind_x_component"] = np.cos(weather_df["wind_direction_10m"] / (180 / np.pi))
    weather_df["wind_y_component"] = np.sin(weather_df["wind_direction_10m"] / (180 / np.pi))
    weather_df.drop(columns='wind_direction_10m', inplace=True)

    air_train, air_test_and_val = air_df.loc[:'2023-12-31 23:00:00'], air_df.loc['2024-01-01 00:00:00':]
    weather_train, weather_test_and_val = weather_df.loc[:'2023-12-31 23:00:00'], weather_df.loc['2024-01-01 00:00:00':]

    indices = np.arange(len(air_test_and_val))
    half_way = int(len(indices) * 0.5)

    air_valid = air_test_and_val.iloc[:half_way]
    air_test = air_test_and_val.iloc[half_way:]

    weather_valid = weather_test_and_val.iloc[:half_way]
    weather_test = weather_test_and_val.iloc[half_way:]

    air_train = air_train.reset_index().sort_values(by=['province', 'time'])
    weather_train = weather_train.reset_index().sort_values(by=['province', 'time'])

    air_valid = air_valid.reset_index().sort_values(by=['province', 'time'])
    weather_valid = weather_valid.reset_index().sort_values(by=['province', 'time'])

    air_test = air_test.reset_index().sort_values(by=['province', 'time'])
    weather_test = weather_test.reset_index().sort_values(by=['province', 'time'])

    air_train.drop(columns=['province', 'time'], inplace=True)
    weather_train.drop(columns=['province', 'time'], inplace=True)

    air_valid.drop(columns=['province', 'time'], inplace=True)
    weather_valid.drop(columns=['province', 'time'], inplace=True)

    air_test.drop(columns=['province', 'time'], inplace=True)
    weather_test.drop(columns=['province', 'time'], inplace=True)

    n_provinces = 3
    window_size = 3

    train_dataset = TimeSeries3DDataset(air_train, weather_train, n_provinces, window_size)
    valid_dataset = TimeSeries3DDataset(air_valid, weather_valid, n_provinces, window_size)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers = 0, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers = 0, pin_memory=True)

    joblib.dump(train_dataset.features_scaler, 'train/weather_scaler.pickle')
    joblib.dump(train_dataset.target_scaler, 'train/air_scaler.pickle')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ConvLSTMTimeSeries(
                input_dim = n_provinces,
                hidden_dim = [256, 128, 64],
                input_width = train_dataset.features_length,
                output_width = train_dataset.target_length
                )

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    criterion = nn.MSELoss()
    num_epochs = 50
    val_patience = 20
    waited_epoch = 0
    best_val_loss = float('inf')

    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        train_loader = tqdm(train_dataloader, desc="Training", leave=False)
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_loader.set_postfix({"Loss": f"{total_loss / (train_loader.n + 1):.4f}"})
        
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Training Loss: {avg_train_loss:.4f}")

        model.eval()
        total_val_loss = 0
        valid_loader = tqdm(valid_dataloader, desc="Validation", leave=False)
        with torch.no_grad():
            for X, y in valid_loader:
                X, y = X.to(device), y.to(device)
                output = model(X)
                loss = criterion(output, y)
                total_val_loss += loss.item()
                valid_loader.set_postfix({"Val Loss": f"{total_val_loss / (valid_loader.n + 1):.4f}"})

        avg_val_loss = total_val_loss / len(valid_dataloader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            waited_epoch = 0
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "conv_lstm.pth")
        else:
            waited_epoch += 1
            if waited_epoch >= val_patience:
                print("Early stopping triggered.")
    return
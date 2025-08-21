import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import joblib

from models.convlstm import ConvLSTM, ConvLSTMCell, ConvLSTMTimeSeries, TimeSeries3DDataset
def root_mean_squared_error(y_true, y_pred, multioutput="uniform_average"):
    mse = mean_squared_error(y_true, y_pred, multioutput=multioutput)
    return np.sqrt(mse)

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

    test_dataset = TimeSeries3DDataset(air_test, weather_test, n_provinces, window_size)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers = 0, pin_memory=True)

    best_conv_lstm = ConvLSTMTimeSeries(
        input_dim = n_provinces,
        hidden_dim = [256, 128, 64],
        input_width = test_dataset.features_length,
        output_width = test_dataset.target_length
    )
    best_conv_lstm.load_state_dict(torch.load("./train/conv_lstm.pth", map_location="cpu", weights_only=True))
    target_scaler = joblib.load("train/air_scaler.pickle")
    with torch.no_grad():
        outputs = []
        true_values = []
        for X, y in test_dataloader:
            output = best_conv_lstm.predict(X, numpy_output=False).squeeze()
            output = output.view(n_provinces, 1, test_dataset.target_length)
            outputs.append(output)

            y = y.squeeze()
            y = y.view(n_provinces, 1, test_dataset.target_length)
            true_values.append(y)

        stacked_outputs = torch.cat(outputs, dim=1)
        original_outputs = stacked_outputs.view(-1, 6)
        original_outputs = target_scaler.inverse_transform(original_outputs)

        stacked_true = torch.cat(true_values, dim=1)
        original_true = stacked_true.view(-1, 6)
        original_true = target_scaler.inverse_transform(original_true)

    
    rmse = pd.Series(root_mean_squared_error(original_outputs, original_true, multioutput="raw_values"), 
          index=["co", "no2", "o3", "so2", "pm2_5", "pm10"])
    nrmse = pd.Series(rmse / np.mean(original_true, axis=0), index=["co", "no2", "o3", "so2", "pm2_5", "pm10"])
    print("RMSE:\n", rmse)
    print("NRMSE:\n", nrmse)
    return

if __name__ == "__main__":
    main()

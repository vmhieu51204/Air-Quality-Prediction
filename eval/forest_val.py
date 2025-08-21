import pandas as pd
import numpy as np
import random
from joblib import dump, load
from utils.utils import sliding_window

from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

def main():
    
    air = pd.read_csv("data/air_data.csv")
    weather = pd.read_csv("data/weather_data.csv")    

    weather_np = []
    air_np = []

    for city_id in ['Hà Nội', 'Hưng Yên', 'Bắc Ninh']:   
        air_df = air.loc[air['province'] == city_id].drop(columns=['province'])
        weather_df = weather.loc[weather['province'] == city_id].drop(columns=['province'])   
 
        air_df = air_df.loc[(air_df.iloc[:, 1:] >= 0).all(axis=1)]
        air_df.drop("aqi", axis=1, inplace=True)
        air_df.reset_index(drop=True, inplace=True)

        weather_df.dropna(axis=0, inplace=True)
        weather_df.reset_index(drop=True, inplace=True)

        X, y = sliding_window(weather_df, air_df, target_size="same")

        m = X.shape[0]
        X = X.reshape((m, -1))

        weather_np.append(X)
        air_np.append(y)
        
    weather_np = np.vstack(weather_np)
    air_np = np.vstack(air_np)
    air_np = air_np[:, -1]

    weather_np = weather_np.astype("float32")
    air_np = air_np.astype("float32")

    random.seed(42)
    idx = list(range(len(weather_np)))
    random.shuffle(idx)

    train_ratio = 0.8  
    split_point = int(train_ratio * len(X))

    train_idx = idx[:split_point]
    test_idx = idx[split_point:]
    X_train, X_test, y_train, y_test = weather_np[train_idx], weather_np[test_idx], air_np[train_idx], air_np[test_idx]

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('forest', RandomForestRegressor(n_estimators=30, n_jobs=-1))    
    ])

    model = TransformedTargetRegressor(
        regressor=pipeline,
        transformer=StandardScaler()
    )

    model = load("./train/random_forest.pkl")

    y_pred = model.predict(X_test)

    # RMSE
    rmse = pd.Series(root_mean_squared_error(model.predict(X_test), y_test, multioutput="raw_values"), 
            index=["co", "no2", "o3", "so2", "pm2_5", "pm10"])

    # nRMSE (normalized by mean of y_test)
    nrmse = rmse / np.mean(y_test)

    print(f"RMSE: {rmse:.4f}")
    print(f"nRMSE: {nrmse:.4f}")
    return

if __name__ == "__main__":
    main()
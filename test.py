import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import LSTM, Dense, Dropout,Input, MultiHeadAttention, LayerNormalization
from tensorflow.keras.optimizers import Adam,RMSprop
import matplotlib.pyplot as plt


# 读取训练和测试数据
train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("test_data.csv")

selected_columns = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit','temp', 'atemp', 'hum', 'windspeed']
train_data_pre = train_data[selected_columns].to_numpy()
test_data_pre = test_data[selected_columns].to_numpy()

scaler = StandardScaler()

train_data_scaled = scaler.fit_transform(train_data[['cnt']])
test_data_scaled = scaler.transform(test_data[ ['cnt']])

train_data_red = np.hstack((train_data_pre,train_data_scaled))
test_data_red = np.hstack((test_data_pre,test_data_scaled))
train_data_red = np.nan_to_num(train_data_red)  # 将NaN替换为0
test_data_red = np.nan_to_num(test_data_red) 


def create_sequences(data, seq_length, forecast_horizon):
    x, y = [], []
    for i in range(len(data) - seq_length - forecast_horizon+1):
        if i == len(data) - seq_length -forecast_horizon:
            print(data[i:i+seq_length,:12])
        x.append(data[i:i + seq_length,:12])
        y.append(data[i + seq_length:i + seq_length + forecast_horizon, -1])  # 预测'cnt'
    return np.array(x), np.array(y)

seq_length = 96  # 历史96小时数据
forecast_horizon_96 = 96  # 短期预测
forecast_horizon_240 = 240  # 长期预测

x_train_96, y_train_96 = create_sequences(train_data_red, seq_length, forecast_horizon_96)
x_train_240, y_train_240 = create_sequences(train_data_red, seq_length, forecast_horizon_240)

x_test_96,y_test_96 = create_sequences(test_data_red,seq_length,forecast_horizon_96)
x_test_240,y_test_240 = create_sequences(test_data_red,seq_length,forecast_horizon_240)

print(len(x_test_240[0][0]))
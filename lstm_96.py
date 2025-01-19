import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout,Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


def build_lstm_model(input_shape, forecast_horizon):
    model = Sequential()
    model.add(LSTM(units=64, activation='tanh', input_shape=input_shape, return_sequences=True))  # LSTM layer with sequences
    model.add(Dropout(0.2))
    
    # Adding more fully connected (Dense) layers
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.3))  # Additional dropout for regularization
    
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.3))  # Another dropout layer

    model.add(Flatten())
    model.add(Dense(units=forecast_horizon))
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


def create_sequences(data, seq_length, forecast_horizon):
    x, y = [], []
    for i in range(len(data) - seq_length - forecast_horizon+1):
        x.append(data[i:i + seq_length,:12])
        y.append(data[i + seq_length:i + seq_length + forecast_horizon, -1])  # 预测'cnt'
    return np.array(x), np.array(y)


# 读取训练和测试数据
train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("test_data.csv")

# 读取所需数据列
selected_columns = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit','temp', 'atemp', 'hum', 'windspeed']
train_data_pre = train_data[selected_columns].to_numpy()
test_data_pre = test_data[selected_columns].to_numpy()

# 对测试数据进行归一化
scaler = MinMaxScaler(feature_range=(0, 1))
train_data_scaled = scaler.fit_transform(train_data[['cnt']])
test_data_scaled = scaler.transform(test_data[ ['cnt']])

# 将数据与归一化后的数据合并
train_data_red = np.hstack((train_data_pre,train_data_scaled))
test_data_red = np.hstack((test_data_pre,test_data_scaled))
train_data_red = np.nan_to_num(train_data_red)  # 将NaN替换为0
test_data_red = np.nan_to_num(test_data_red) 


seq_length = 96  # 历史96小时数据
forecast_horizon_96 = 96  # 短期预测

x_train_96, y_train_96 = create_sequences(train_data_red, seq_length, forecast_horizon_96)
x_test_96,y_test_96 = create_sequences(test_data_red,seq_length,forecast_horizon_96)

# 构建LSTM模型
model_96 = build_lstm_model(x_train_96.shape[1:], forecast_horizon_96)

# 实际的cnt值
y_tr_96 = test_data['cnt'].to_numpy() 
y_true_96 = []
for i in range(len(x_test_96)):  
    y_true_96.append(y_tr_96[i+forecast_horizon_96:i+forecast_horizon_96+forecast_horizon_96]) 


mse_96_lstm_list = []
mae_96_lstm_list = []
for i in range(5):
    model_96.fit(x_train_96, y_train_96, epochs=20, batch_size=32)
    y_pred_96= model_96.predict(x_test_96)
    # 对测试数据进行逆归一化
    y_pred_96 = y_pred_96 * (scaler.data_max_[-1] - scaler.data_min_[-1]) + scaler.data_min_[-1]
    mse_96_lstm_list.append(mean_squared_error(y_true_96, y_pred_96))
    mae_96_lstm_list.append(mean_absolute_error(y_true_96, y_pred_96))

print(mse_96_lstm_list)
print(mae_96_lstm_list)
print(f"Average MSE: {np.mean(mse_96_lstm_list)}, Standard Deviation: {np.std(mse_96_lstm_list)}")
print(f"Average MAE: {np.mean(mae_96_lstm_list)}, Standard Deviation: {np.std(mae_96_lstm_list)}")



# 绘制真实值（Ground Truth）
plt.plot(y_true_96[1], label='Ground Truth (Actual)', color='blue')

# 绘制预测值
plt.plot(y_pred_96[1], label='Predicted', color='red',linestyle='--')

# 添加标题和标签
plt.title('Bike Rental Count Prediction vs Ground Truth', fontsize=16)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Bike Rental Count (cnt)', fontsize=14)

# 显示图表
plt.grid(True)
plt.show()
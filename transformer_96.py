import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import LSTM, Dense, Dropout,Input, MultiHeadAttention, LayerNormalization,Reshape
from tensorflow.keras.optimizers import Adam,RMSprop
import matplotlib.pyplot as plt


# 读取训练和测试数据
train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("test_data.csv")

selected_columns = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit','temp', 'atemp', 'hum', 'windspeed']
train_data_pre = train_data[selected_columns].to_numpy()
test_data_pre = test_data[selected_columns].to_numpy()

scaler = MinMaxScaler(feature_range=(0, 1))

train_data_scaled = scaler.fit_transform(train_data[['cnt']])
test_data_scaled = scaler.transform(test_data[ ['cnt']])

train_data_red = np.hstack((train_data_pre,train_data_scaled))
test_data_red = np.hstack((test_data_pre,test_data_scaled))
train_data_red = np.nan_to_num(train_data_red)  # 将NaN替换为0
test_data_red = np.nan_to_num(test_data_red) 


def create_sequences(data, seq_length, forecast_horizon):
    x, y = [], []
    for i in range(len(data) - seq_length - forecast_horizon+1):
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



def build_transformer_model(input_shape, forecast_horizon):
    # 输入层：96个时间步，每个时间步12个特征
    inputs = Input(shape=input_shape)
    
    # 多头自注意力层
    x = MultiHeadAttention(num_heads=8, key_dim=64)(inputs, inputs)
    x = Dropout(0.1)(x)
    x = LayerNormalization()(x)
    
    x = Dense(512, activation='relu')(x)
    # 全连接层
    x = Dense(64, activation='relu')(x)
    
    # 输出层：为每个时间步预测一个新的特征（未来96个时间步，每个时间步1个预测值）
    x = Dense(1)(x)  # 输出每个时间步1个预测值
    
    # 将输出的形状变为 (batch_size, forecast_horizon)，每个样本预测未来96个时间步的一个特征
    x = Reshape((forecast_horizon,))(x)
    
    # 构建模型
    model = Model(inputs=inputs, outputs=x)
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    return model


# 构建Transformer模型
transformer_96 = build_transformer_model(x_train_96.shape[1:], forecast_horizon_96)


# 训练Transformer模型
#transformer_96.fit(x_train_96, y_train_96, epochs=10, batch_size=32)


y_tr_96 = test_data['cnt'].to_numpy() 
#y_pred_96 = pred = transformer_96.predict(x_test_96)
#y_pred_96 = y_pred_96 * (scaler.data_max_[-1] - scaler.data_min_[-1]) + scaler.data_min_[-1]

y_true_96 = []
for i in range(len(x_test_96)):  # 滚动预测每96小时
    y_true_96.append(y_tr_96[i+forecast_horizon_96:i+forecast_horizon_96*2]) 




mse_96_lstm_list = []
mae_96_lstm_list = []
for i in range(5):
    transformer_96.fit(x_train_96, y_train_96, epochs=20, batch_size=32)
    y_pred_96= transformer_96.predict(x_test_96)
    # 对测试数据进行逆归一化
    y_pred_96 = y_pred_96 * (scaler.data_max_[-1] - scaler.data_min_[-1]) + scaler.data_min_[-1]
    mse_96_lstm_list.append(mean_squared_error(y_true_96, y_pred_96))
    mae_96_lstm_list.append(mean_absolute_error(y_true_96, y_pred_96))

print(mse_96_lstm_list)
print(mae_96_lstm_list)
print(f"Average MSE: {np.mean(mse_96_lstm_list)}, Standard Deviation: {np.std(mse_96_lstm_list)}")
print(f"Average MAE: {np.mean(mae_96_lstm_list)}, Standard Deviation: {np.std(mae_96_lstm_list)}")

# 绘制真实值（Ground Truth）
plt.plot(y_true_96[0], label='Ground Truth (Actual)', color='blue')

# 绘制预测值
plt.plot(y_pred_96[0], label='Predicted', color='red',linestyle='--')

# 添加标题和标签
plt.title('Bike Rental Count Prediction vs Ground Truth', fontsize=16)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Bike Rental Count (cnt)', fontsize=14)

# 添加图例


# 显示图表
plt.grid(True)
plt.show()
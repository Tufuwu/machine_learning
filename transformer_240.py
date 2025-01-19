import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import LSTM, Dense, Dropout,Input, MultiHeadAttention, LayerNormalization,Reshape,Flatten
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
    """
    构建一个基于Transformer的时间序列预测模型

    参数:
    - input_shape: 输入数据的形状 (时间步数, 特征数)，比如 (96, 12)
    - forecast_horizon: 预测的时间步数，这里是 240，代表预测未来240小时

    返回:
    - model: 构建好的Keras模型
    """
    
    inputs = Input(shape=input_shape)  # 输入形状 (96, 12)
    
    # 多头自注意力层
    x = MultiHeadAttention(num_heads=8, key_dim=64)(inputs, inputs)
    x = Dropout(0.1)(x)  # Dropout层，防止过拟合
    x = LayerNormalization()(x)  # 层归一化
    
    # 进一步提取特征
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    
    # 扁平化数据以便进行输出
    x = Flatten()(x)  # 将 (batch_size, 240, 1) 转换为 1D 形状
    
    # 输出层：预测未来240个时间步的一个标量
    x = Dense(forecast_horizon)(x)  # 输出 240 个值
    
    # 将输出 reshape 为 (batch_size, 240, 1)
    #x = Reshape((forecast_horizon,1))(x)
    
    # 构建模型
    model = Model(inputs=inputs, outputs=x)
    
    # 编译模型，使用 Adam 优化器和均方误差损失函数
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    return model


# 构建Transformer模型
#transformer_96 = build_transformer_model(x_train_96.shape[1:], forecast_horizon_96)
transformer_240 = build_transformer_model(x_train_240.shape[1:], forecast_horizon_240)

# 训练Transformer模型
#transformer_240.fit(x_train_240, y_train_240, epochs=10, batch_size=32)
#transformer_240.fit(x_train_240, y_train_240, epochs=10, batch_size=32)

y_tr_240 = test_data['cnt'].to_numpy() 
#y_pred_240 = pred = transformer_240.predict(x_test_240)
#y_pred_240 = y_pred_240 * (scaler.data_max_[-1] - scaler.data_min_[-1]) + scaler.data_min_[-1]
#y_pred = np.squeeze(y_pred_240)
#y_pred_96 = scaler.inverse_transform(y_pred_96.reshape(-1, 1))
y_true_240 = []
for i in range(len(x_test_240)):  # 滚动预测每96小时

    y_true_240.append(y_tr_240[i+forecast_horizon_96:i+forecast_horizon_240+forecast_horizon_96])  



#mse_240_lstm = mean_squared_error(y_true_240, y_pred_240)
#mae_240_lstm = mean_absolute_error(y_true_240, y_pred_240)
#print(mse_240_lstm)
#print(mae_240_lstm)
mse_240_lstm_list = []
mae_240_lstm_list = []
for i in range(5):
    transformer_240.fit(x_train_240, y_train_240, epochs=20, batch_size=32)
    y_pred_240= transformer_240.predict(x_test_240)
    # 对测试数据进行逆归一化
    y_pred_240 = y_pred_240 * (scaler.data_max_[-1] - scaler.data_min_[-1]) + scaler.data_min_[-1]
    y_pred = np.squeeze(y_pred_240)
    mse_240_lstm_list.append(mean_squared_error(y_true_240, y_pred_240))
    mae_240_lstm_list.append(mean_absolute_error(y_true_240, y_pred_240))

print(mse_240_lstm_list)
print(mae_240_lstm_list)
print(f"Average MSE: {np.mean(mse_240_lstm_list)}, Standard Deviation: {np.std(mse_240_lstm_list)}")
print(f"Average MAE: {np.mean(mae_240_lstm_list)}, Standard Deviation: {np.std(mae_240_lstm_list)}")




# 绘制真实值（Ground Truth）
plt.plot(y_true_240[0], label='Ground Truth (Actual)', color='blue')

# 绘制预测值
plt.plot(y_pred_240[0], label='Predicted', color='red',linestyle='--')

# 添加标题和标签
plt.title('Bike Rental Count Prediction vs Ground Truth', fontsize=16)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Bike Rental Count (cnt)', fontsize=14)

# 添加图例


# 显示图表
plt.grid(True)
plt.show()
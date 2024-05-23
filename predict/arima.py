'''
我想使用arima模型进行流量预测，一组流量数据是一个8*8的矩阵，目标是使用每10组的流量数据预测下1组的流量数据。
例如使用1到10组数据预测第11组的数据，使用2到11组数据预测第12组数据。
数据文件存放在features_1400 copy.csv中，一共3200行数据，即一共50组数据。
预测完成后画出真实值和预测值的对比曲线图，并且计算MSE, RMSE, MAE, WAPE
'''
import pandas as pd
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# data = pd.read_csv('features_1400 copy.csv', header=None)

# data = np.array(data)
# data = data / data.sum(axis=1, keepdims=True) # 归一化处理
# data = data.reshape(-1, 8, 8) # 将数据转换为三维矩阵

# 读出所有数据

def normalize(mx):
    # 该归一化方法是 每个数 除以 所在行的行和 [[1,1], [1,3]] -> [[0.5,0.5],[0.25,0.75]]
    rowsum = mx.sum(1).astype(np.float64)  
    r_inv = np.power(rowsum, -1).flatten()  # 尽量不要让数据有0，不然报运行时警告
    r_inv[np.isinf(r_inv)] = 0. # 在计算倒数的时候存在一个问题，如果原来的值为0，则其倒数为无穷大，因此需要对r_inv中无穷大的值进行修正，更改为0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

suffix = ["00_1", "15_1", "30_1", "45_1", "00_5", "15_5", "30_5", "45_5"]
csv_files = [] # 所有数据文件的路径
for index in range(0, 9):# data1000_1.csv
    for item in suffix:
        csv_files.append("/home/sinet/lt/flow_matrix/"+"data0%s%s.csv"%(index, item))
        
data = np.array([[0,0,0,0,0,0,0,0]])
for file in csv_files:
    one_file_data = np.array(pd.read_csv(file, header=None))
    data = np.concatenate((data, one_file_data), axis=0)

data = normalize(data)
data = data[1:,:].reshape(-1, 8, 8) 
# print(data.shape)   (58542, 8, 8)

true_values = []
predicted_values = []

for i in range(64):
    for j in range(1, 50):
        inputs = data[j-1, i // 8, i % 8]
        model = ARIMA([inputs], order=(1, 1, 1))
        result = model.fit()
        forecast = result.forecast(steps=1)[0] # 预测下一组数据
        true_value = data[j, i // 8, i % 8] # 真实值
        predicted_value = forecast # 预测值
        true_values.append(true_value)
        predicted_values.append(predicted_value)

true_values = np.array(true_values).reshape(-1)
predicted_values = np.array(predicted_values).reshape(-1)

# plt.plot(true_values, label='True Values')
# plt.plot(predicted_values, label='Predicted Values')
# plt.legend()
# plt.show()

mse = mean_squared_error(true_values, predicted_values)
rmse = np.sqrt(mse)
mae = mean_absolute_error(true_values, predicted_values)
wape = np.mean(np.abs(true_values - predicted_values) / np.abs(true_values))

print('MSE:', mse)
print('RMSE:', rmse)
print('MAE:', mae)
print('WAPE:', wape)

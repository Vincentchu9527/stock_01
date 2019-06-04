# -*- coding:utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from itertools import product
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 文件加载
filename = './shanghai_1990-12-19_to_2019-2-28.csv'
data = pd.read_csv(filename)
# print(data.head())
# print(data.info())

# 数据变换
data.Timestamp = pd.to_datetime(data.Timestamp)
data.set_index(['Timestamp'], inplace=True)
# print(data.head())
data_month = data.resample('M').mean()
# print(data_month)
train_data = data_month.Price

# 平稳性检测
n_diff = 0
adf = adfuller(train_data)
while adf[1] >= 0.05:
    n_diff += 1
    adf = adfuller(train_data.diff(n_diff).dropna())
print('原始序列经过%s阶差分后归于平稳, p值为: %s' % (n_diff, adf[1]))

# 白噪声检验
[[lb], [p]] = acorr_ljungbox(train_data.diff(n_diff).dropna(), lags=1)
print('白噪声检验p值为 %s' % p)
print('是否为非白噪声:', p < 0.05)

# 寻找最优ARIMA模型参数
# 1. 构造参数池
ps = range(0, 5)
qs = range(0, 5)
parameters_origin = product(ps, qs)
parameters = []
for i in parameters_origin:
    i = list(i)
    i.insert(1, 1)
    parameters.append(i)
# 2. 初始化最优参数
results = []
best_aic = float("inf")
best_param = (0, 1, 0)
# 3. 通过循环找出aic值最小的模型
best_model = ARIMA(train_data, order=best_param).fit()
for param in parameters:
    try:
        model = ARIMA(train_data, order=param).fit()
    except ValueError:
        print('参数错误:', param)
        continue
    aic = model.aic
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param
    results.append([param, model.aic])
# 4. 输出最优模型
result_table = pd.DataFrame(results)
result_table.columns = ['parameters', 'aic']
print(result_table)
print('最优模型: ', best_model.summary())

# 残差白噪声检验
lagnum = 12  # 设置残差延迟个数
# best_model = ARIMA(train_data, (4, 1, 2)).fit()
data_pred = best_model.predict(typ='levels')
pred_error = (data_pred - train_data).dropna()  # 计算残差
# print(pred_error)
lb, p = acorr_ljungbox(pred_error, lags=lagnum)
h = (p < 0.05).sum()  # 残差序列为白噪声则模型通过检验
print('h值为:', h)
if h > 0:
    print('模型ARIMA(4, 1, 2) 不符合残差白噪声检验')
else:
    print('模型ARIMA(4, 1, 2) 符合残差白噪声检验')

# 预测
# best_model = ARIMA(train_data, (4, 1, 2)).fit()
data2 = data_month.copy()
date_list = pd.date_range('2019-03-31', periods=10, freq='M')
future = pd.DataFrame(index=date_list, columns=data2.columns)
data2 = pd.concat([data2, future])
# print(data2)
# print(data2.info())
data2['forecast'] = best_model.predict(start=1, end=349, typ='levels')
print(data2)

# 绘图
# data2.Price.plot(label='实际')
# data2.forecast.plot(color='r', ls='--', label='预测')
data2.Price[-20:].plot(label='实际')
data2.forecast[-20:].plot(color='r', ls='--', label='预测')
plt.legend()
plt.title('股市指数(月)')
plt.xlabel('时间')
plt.ylabel('指数')
plt.show()

# 保存
data2.to_excel('./result_01.xls')

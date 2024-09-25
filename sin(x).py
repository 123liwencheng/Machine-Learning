import numpy as np
import matplotlib.pyplot as plt

# 数据点个数
data_count = 100
x = np.linspace(0, 2 * np.pi, data_count)

# 五次多项式的初始系数
param = [0, 1, 0, -1 / 6, 0, 1 / 120]

# 定义五次多项式
def function_sin(xx, param):
    return sum(p * xx ** i for i, p in enumerate(param))

# 定义损失函数
def loss_function(param):
    return np.mean((np.sin(x) - function_sin(x, param)) ** 2)

# 损失函数偏导数
def partial_derivative(j, param):
    return -2 * np.mean((np.sin(x) - function_sin(x, param)) * x ** j)

# 反向传播更新参数
learning_rate = 0.0000001  # 学习率
epochs = 9  # 训练轮数
times_per_epoch = [1000, 8000, 500, 5000, 6000, 1000, 3000, 3000, 1000]  # 每轮的迭代次数

for epoch in range(epochs):
    x = np.linspace(0, 2 * np.pi, data_count) if epoch == 0 else np.linspace(5, 6, data_count) if epoch == 1 else \
        np.linspace(0, 1, data_count) if epoch == 2 else np.linspace(3, 4, data_count) if epoch == 3 else \
        np.linspace(4, 2 * np.pi, data_count) if epoch == 4 else np.linspace(2, 3, data_count) if epoch == 5 else \
        np.linspace(4, 2 * np.pi, data_count) if epoch == 6 else np.linspace(2, 2 * np.pi, data_count) if epoch == 7 else \
        np.linspace(0, 2 * np.pi, data_count)

    min_loss = loss_function(param)
    param_min = param.copy()

    for i in range(times_per_epoch[epoch]):
        for j in range(len(param)):
            param[j] -= learning_rate * partial_derivative(j, param)

        current_loss = loss_function(param)
        if current_loss < min_loss:
            param_min = param.copy()
            min_loss = current_loss
            if min_loss - current_loss < 0.01:
                learning_rate *= 1.1  # 学习率增加
        else:
            param = param_min.copy()  # 恢复到之前的最优参数
            learning_rate *= 0.85  # 学习率减小

        print(f'第{i + 1}次迭代，参数值：{param}，损失函数值：{current_loss:.6f}')

# 绘制拟合曲线
y = np.sin(x)
plt.plot(x, y, 'o', label='data')
y_fit = function_sin(x, param)
plt.plot(x, y_fit, label='fit')
plt.legend()
plt.show()

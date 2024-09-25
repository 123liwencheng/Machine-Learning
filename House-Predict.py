import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 加载数据
data = pd.read_csv('House-Price.csv')

# 特征和目标
X = data.drop('MEDV', axis=1)  # 特征
y = data['MEDV']  # 目标变量

# 归一化特征
def normalize(x):
    x_min = x.min(0)
    x_max = x.max(0)
    return (x - x_min) / (x_max - x_min)

X = normalize(X)

# 添加偏置项
X = np.c_[np.ones(X.shape[0]), X]  # 使用 NumPy 添加偏置项

# 初始化参数
weights = np.zeros(X.shape[1])
bias = 0

# 损失函数
def compute_loss(X, y, weights, bias):
    m = len(y)
    predictions = X.dot(weights) + bias
    sqrErrors = np.square(predictions - y)
    return np.sum(sqrErrors) / (2 * m)

# 梯度下降
def gradient_descent(X, y, weights, bias, learning_rate, iterations):
    m = len(y)
    for i in range(iterations):
        predictions = X.dot(weights) + bias
        errors = predictions - y
        updates = (learning_rate / m) * (X.T.dot(errors))
        weights -= updates
        bias -= learning_rate * np.sum(errors) / m
    return weights, bias

# 预测
def predict(X, weights, bias):
    return X.dot(weights) + bias

# 训练模型
learning_rate = 0.01
iterations = 500
weights, bias = gradient_descent(X, y, weights, bias, learning_rate, iterations)

# 预测
predictions = predict(X, weights, bias)

# 可视化预测结果与实际结果
plt.scatter(y, predictions)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs Predicted Prices')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # 添加y=x参考线
plt.show()

# 输出模型参数
print("Weights: \n", weights[1:])  # 不包括偏置项
print("Bias: \n", bias)

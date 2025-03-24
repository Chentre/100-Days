# Simple Linear Regression

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Step 1: Data Preprocessing

dataset = pd.read_csv('100-Days\studentscores.csv')
print("\n数据集 dataset:")
print(dataset)

X = dataset.iloc[:, : 1].values
Y = dataset.iloc[:, 1].values

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=1/4, random_state=0)

# Step 2: Fitting Simple Linear Regression Model to the training set

regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

# Step 3: Predecting the Result
Y_pred = regressor.predict(X_test)

print("\n划分数据集后 X_train:")
print(X_train)
print("划分数据集后 X_test:")
print(X_test)
print("划分数据集后 Y_train:")
print(Y_train)
print("划分数据集后 Y_test:")
print(Y_test)

print("\n预测后预测结果 Y_pred:")
print(Y_pred)

# Step 4: Visualization
# Visualising the Training results

plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.show()

# Visualizing the test results
plt.scatter(X_test, Y_test, color='red')  # 绘制测试集散点
plt.plot(X_test, Y_pred, color='blue')  # 绘制线性模型
plt.show()

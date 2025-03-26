# Multiple Linear Regression

# Importing the libraries

from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd


# Step 1: Data Preprocessing

# Importing the dataset

dataset = pd.read_csv('100-Days\datasets\\50_Startups.csv')

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,  4].values

# Encoding Categorical data 编码分类数据

labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
print("\n对分类特征进行编码后 X:")
print(X)

# One Hot Encoding 独热编码
onehotencoder = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(onehotencoder.fit_transform(X))
print("\n独热编码后 X:")
print(X)

# # Avoiding Dummy Variable Trap 避免虚拟变量陷阱
# X = X[:, 1:]
# print("\n避免虚拟变量陷阱后 X:")
# print(X)

# Splitting the dataset into the Training set and Test set 划分训练集和测试集

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)

print("\n划分数据集后 X_train:")
print(X_train)
print("划分数据集后 X_test:")
print(X_test)
print("划分数据集后 Y_train:")
print(Y_train)
print("划分数据集后 Y_test:")
print(Y_test)

# Step 2: Fitting Multiple Linear Regression to the Training set 拟合多元线性回归模型
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


# Step 3: Predicting the Test set results 预测测试集结果

y_pred = regressor.predict(X_test)
print("\n预测测试集结果 y_pred:")
print(y_pred)

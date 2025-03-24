# Logistic Regression

# Step 1 | Data Pre-Processing

# Importing the Libraries
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('100-Days\datasets\Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

print("\n划分数据集后 X_train:")
print(X_train)
print("划分数据集后 X_test:")
print(X_test)
print("划分数据集后 y_train:")
print(y_train)
print("划分数据集后 y_test:")
print(y_test)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print("\n特征缩放后 X_train:")
print(X_train)
print("特征缩放后 X_test:")
print(X_test)
print("特征缩放后 y_train:")
print(y_train)
print("特征缩放后 y_test:")
print(y_test)

# Step 2 | Logistic Regression Model
# Fitting Logistic Regression to the Training set
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Step 3 | Predection
# Predicting the Test set results
y_pred = classifier.predict(X_test)
print("\n预测后预测结果 y_pred:")
print(y_pred)


# Step 4 | Evaluating The Predection
# Making the Confusion Matrix 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print("\n混淆矩阵 cm:")
print(cm)

# Visualization 
def plot_decision_boundary(X, y, title):
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.RdYlGn)  # 合适的红绿渐变cmap
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', label=0)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='green', label=1)
    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()

# 绘制训练集可视化
plot_decision_boundary(X_train, y_train, 'Logistic Regression (Training set)')
plt.show()

# 绘制测试集可视化
plot_decision_boundary(X_test, y_test, 'Logistic Regression (Test set)')
plt.show()
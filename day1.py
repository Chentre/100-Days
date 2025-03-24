# Step 1: Importing the libraries
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd

# Step 2: Importing dataset
dataset = pd.read_csv("100-Days\Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values
print("\n导入数据集后 X:")
print(X)
print("导入数据集后 Y:")
print(Y)

# Step 3: Handling the missing data
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print("\n处理缺失值后 X:")
print(X)

# Step 4: Encoding categorical data 编码分类数据
# 对分类特征进行编码
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
print("\n对分类特征进行编码后 X:")
print(X)

# 使用 OneHotEncoder 对分类特征进行独热编码
onehotencoder = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [0])],
                                  remainder='passthrough')
X = onehotencoder.fit_transform(X)
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
print("\n独热编码后 X:")
print(X)
print("独热编码后 Y:")
print(Y)


# Step 5: Splitting the datasets into training sets and Test sets 划分训练集和测试集
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

# Step 6: Feature Scaling 特征缩放
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

print("\n特征缩放后 X_train:")
print(X_train)
print("特征缩放后 X_test:")
print(X_test)

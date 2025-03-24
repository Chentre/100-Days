from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd

dataset = pd.read_csv("100-Days\Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

onehotencoder = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [0])],
                                  remainder='passthrough')

X = onehotencoder.fit_transform(X)

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# 输出处理后的数据
print("处理后的特征矩阵 X_train:")
print(X_train)
print("\n处理后的特征矩阵 X_test:")
print(X_test)
print("\n处理后的目标变量 Y_train:")
print(Y_train)
print("\n处理后的目标变量 Y_test:")
print(Y_test)

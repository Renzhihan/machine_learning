import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv('Data.csv')
X=dataset.iloc[:,:-1].values 
#取所有行数 除去最后一列
y=dataset.iloc[:,3].values

#taking care of  missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(add_indicator=False,copy=True,missing_values=np.nan,strategy='mean',verbose=0) #均值
X[:,1:3]=imputer.fit_transform(X[:,1:3])

#encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])

#dummy encoding 虚拟编码
onehotencoder=OneHotEncoder()
X_t=onehotencoder.fit_transform(X[:,0].reshape(-1, 1)).toarray()
X=np.concatenate((X_t,X[:,1:3]),axis=1)

labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)

#spliting the dataset into the training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y,train_size=0.8,random_state=0)
#random_state=0时，每次得到完全一样的训练集和测试集

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

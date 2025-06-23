import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline

# Import Linear Regression Machine learning libraries
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.metrics import r2_score

data = pd.read_csv("E:\Datasets\car-mpg.csv")
data.head()

data = data.drop(['car_name'],axis = 1)
data['origin'] = data['origin'].replace({1:'America',2:'Europe',3:'Asia'})
data = pd.get_dummies(data,columns = ['origin'],dtype = int)
data = data.replace('?',np.nan)

# data = data.apply(lambda x: x.fillna(x.median()),axis = 0)

data = data.apply(pd.to_numeric, errors='ignore')
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].apply(lambda x: x.fillna(x.median()))

x = data.drop(['mpg'],axis = 1)
y = data[['mpg']]


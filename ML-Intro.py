# Import Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"E:\Datasets\Data.csv")

# Split the data into X and Y
x = dataset.iloc[:,:-1].values

# Dependent variable 
y = dataset.iloc[:,3].values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer() # by default it will take mean --> parameter and median and mode are hypermeter

imputer  = imputer.fit(x[:,1:3])

x[:,1:3] = imputer.transform(x[:,1:3])


# Convert Categorical data into numerical data
from sklearn.preprocessing import LabelEncoder

labelencoder_x = LabelEncoder()

labelencoder_x.fit_transform(x[:,0])

x[:,0] = labelencoder_x.fit_transform(x[:,0])

labelencoder_y = LabelEncoder()

y = labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.8,random_state= 0)














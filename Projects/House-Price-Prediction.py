
# House Price Using Backward Elimination

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

%matplotlib inline

# Importing Dataset

data = pd.read_csv(r"E:\Datasets\House_data.csv")

# Checking missing values

data.isnull().any().sum()

data.dtypes

data = data.drop(['id','date'],axis = 1)

data.shape

with sns.plotting_context("notebook",font_scale = 2.5):
    g = sns.pairplot(data[['sqft_lot','sqft_above','price','sqft_living','bedrooms']],
                     hue = 'bedrooms',palette = 'tab20',height = 6)
    g.set(xticklabels = []);
    
# Seperating independent and Dependent Variables

x = data.iloc[:,1:].values
y = data.iloc[:,0].values


#splitting dataset into training and testing dataset
from sklearn.model_selection import train_test_split

x_train,x_test, y_train,y_test = train_test_split(x,y,test_size = 1/3,random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

# Training the model with train data
regressor.fit(x_train,y_train)

# Predicting the test set result

y_pred = regressor.predict(x_test)

# Backward Elimination

import statsmodels.api as sm

def backwardElimination(a,SL): # a is the indipendent variable(columns/features)
    numVars = len(a[0])
    temp = np.zeros((21613,19)).astype(int)
    for i in range(0,numVars):
        regressor_OLS = sm.OLS(y,a).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0,numVars - i):
                if (regressor_OLS.pvalues[j].astype(float)==maxVar):
                    temp[:,j] = a[:, j]
                    a = np.delete(a,j,1)
                    tmp_regressor = sm.OLS(y,a).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((a,temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback,j,1)
                        print(regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
                    
                    
    regressor_OLS.summary()
    return a

SL = 0.05

x_opt = x[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]]
x_Modeled = backwardElimination(x_opt,SL)




















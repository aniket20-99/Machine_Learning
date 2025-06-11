import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load the datasets
df = pd.read_csv(r"E:\Datasets\Salary_Data.csv")
missing_val = df.isnull().sum()

# Split the dataset into x and y
x = df.iloc[:,:-1] # To remove we use :-1

y = df.iloc[:,-1]


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.8,random_state = 0)

from sklearn.linear_model import LinearRegression # Algorithm
regressor = LinearRegression() # Regressor = model
regressor.fit(x_train,y_train) # x_train is question and y_train is answer

y_pred = regressor.predict(x_test) 
comparison = pd.DataFrame({'Actual': y_test,'Predicted': y_pred})
print(comparison)

plt.scatter(x_test,y_test,color = 'red')
plt.plot(x_train,regressor.predict(x_train),color = 'blue')
plt.title('Salary vs Experience')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.7,random_state = 0)

from sklearn.linear_model import LinearRegression # Algorithm
regressor = LinearRegression() # Regressor = model
regressor.fit(x_train,y_train) # x_train is question and y_train is answer

y_pred = regressor.predict(x_test) 


plt.scatter(x_test,y_test,color = 'blue')
plt.plot(x_train,regressor.predict(x_train),color = 'red')
plt.title('Salary vs Experience')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

# Training and testing phase always works on historical data(seen data) 
# Validation phase always works on future data(unseen data)

m_slope = regressor.coef_ # m - slope
print(m_slope)

c_intercept = regressor.intercept_ # c - Constant
print(c_intercept)

y_12 = m_slope * 12 + c_intercept # y^ = mx + c,Here 12 is x which is future exp data of an employee
print(y_12)
























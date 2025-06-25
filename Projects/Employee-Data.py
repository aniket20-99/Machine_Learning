import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv(r'E:\Datasets\emp_sal.csv')

X = data.iloc[:,1:2].values
y = data.iloc[:,2].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

plt.scatter(X,y,color = 'red')
plt.plot(X,lin_reg.predict(X),color = 'blue')
plt.title('Linear regression model (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


lin_model_pred = lin_reg.predict([[6.5]])
lin_model_pred

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures()

x_poly = poly_reg.fit_transform(X)
poly_reg.fit(x_poly,y)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)


plt.scatter(X,y,color = 'red')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)),color = 'blue')
plt.title('Poly model(Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

lin_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
lin_model_pred


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 6)

x_poly = poly_reg.fit_transform(X)
poly_reg.fit(x_poly,y)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)


plt.scatter(X,y,color = 'red')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)),color = 'blue')
plt.title('Poly model(Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

lin_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
lin_model_pred                                                                      










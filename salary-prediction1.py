import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Load the dataset
data = pd.read_csv(r"E:\Datasets\Salary_Data.csv")

data.isnull().sum() #  Cehcking Missing values

# Split the dataset into features and target variable
x = data.iloc[:,:-1]
y = data.iloc[:,-1]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 42)

from sklearn.linear_model import LinearRegression

# model name
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

comparison = pd.DataFrame({'Actual': y_test,'Predicted':y_pred})
print(comparison)

# Visualizing the results
plt.scatter(x_test,y_test,color = 'red')
plt.plot(x_train,regressor.predict(x_train),color = 'blue')
plt.title('Salary vs Experience')
plt.xlabel('Year of Experiences')
plt.ylabel('"Salary')
plt.show()

# Finding slope and intercept

m_slope = regressor.coef_
print(f'Slope: {m_slope}')

c_intercept = regressor.intercept_
print(f"Intercept(c):{c_intercept}")

y_future = m_slope * 15 + c_intercept
print(f"Predicted salary for 15 years experience is {y_future}")



# Assuming `regressor` is already trained and available

st.title("💼 Salary Prediction App (India)")
st.markdown("Enter your **Years of Experience** and we'll predict your expected salary based on historical data.")

# Input from user
experience = st.number_input("👨‍💻 How many years of experience do you have?", min_value=0.0, max_value=50.0, step=0.5)

# Prediction logic
if st.button("Predict My Salary 💰"):
    predicted_salary = regressor.predict(np.array([[experience]]))[0]
    st.success(f"✅ Based on your experience, your estimated salary is: ₹{predicted_salary:,.2f}")
    
    # Display slope and intercept for transparency
    st.subheader("📈 Model Evaluation Metrics")
    st.write(f"**Slope (m):** {m_slope[0]:.2f}")
    st.write(f"**Intercept (c):** ₹{c_intercept:,.2f}")
    
    # Predict for 15 years as static reference (optional)
    y_future = m_slope * (experience + 5) + c_intercept
    st.info(f"📊 For reference, someone with {experience + 5} years experience earns approximately ₹{y_future[0]:,.2f}")

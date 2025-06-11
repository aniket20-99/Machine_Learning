import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Title
st.set_page_config(page_title="Salary Predictor", layout="centered")
st.title("💼 Salary Prediction App (India)")
st.markdown("📊 This app predicts salary based on years of experience using a linear regression model.")

# Load the dataset
data = pd.read_csv(r"E:\Datasets\Salary_Data.csv")

# Check and clean
if data.isnull().sum().any():
    st.warning("Dataset contains missing values. Please clean your data.")
else:
    # Split the dataset
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Train the model
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    # Slope and Intercept
    m_slope = regressor.coef_
    c_intercept = regressor.intercept_

    # User input
    experience = st.number_input("👨‍💻 Enter your years of experience:", min_value=0.0, max_value=50.0, step=0.5)

    if st.button("Predict My Salary 💰"):
        predicted_salary = regressor.predict(np.array([[experience]]))[0]
        st.success(f"✅ Estimated Salary: ₹{predicted_salary:,.2f}")

        st.subheader("🔍 Model Insights")
        st.write(f"**📈 Slope (m):** {m_slope[0]:.2f}")
        st.write(f"**📉 Intercept (c):** ₹{c_intercept:,.2f}")

        future_exp = experience + 5
        y_future = regressor.predict(np.array([[future_exp]]))[0]
        st.info(f"📌 Projected salary after {future_exp} years experience: ₹{y_future:,.2f}")

        # Optional plot
        fig, ax = plt.subplots()
        ax.scatter(x_test, y_test, color='red', label='Actual')
        ax.plot(x_train, regressor.predict(x_train), color='blue', label='Prediction Line')
        ax.set_title("Salary vs Experience")
        ax.set_xlabel("Years of Experience")
        ax.set_ylabel("Salary (INR)")
        ax.legend()
        st.pyplot(fig)

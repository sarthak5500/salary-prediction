import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample data for training
data = {
    'YearsExperience': [1.1, 1.3, 1.5, 2.0, 2.2, 2.9, 3.0, 3.2, 3.2, 3.7, 3.9, 4.0,
                        4.5, 4.9, 5.1, 5.3, 5.9, 6.0, 6.8, 7.1],
    'Salary': [39343, 46205, 37731, 43525, 39891, 56642, 60150, 54445, 64445,
               57189, 63218, 55794, 56957, 57081, 61111, 67938, 66029, 83088, 81363, 93940]
}

df = pd.DataFrame(data)

# Train the model
X = df[['YearsExperience']]
y = df['Salary']
model = LinearRegression()
model.fit(X, y)

# Streamlit app
st.title("💼 Salary Predictor")
st.write("Enter your years of experience to predict your salary:")

# Input from user
experience = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, step=0.1)

# Predict salary
if st.button("Predict Salary"):
    predicted_salary = model.predict([[experience]])[0]
    st.success(f"Estimated Salary: Rs{predicted_salary:,.2f}")

# Optional: Show the data and model info
with st.expander("See training data"):
    st.dataframe(df)

with st.expander("Model Information"):
    st.write(f"Model Coefficient (slope): {model.coef_[0]:.2f}")
    st.write(f"Model Intercept: {model.intercept_:.2f}")

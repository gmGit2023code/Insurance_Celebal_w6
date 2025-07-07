import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import plotly.express as px

# Load data
def load_data():
    return pd.read_csv("insurance.csv")

# Train model
def train_model():
    df = load_data()
    X = df.drop("charges", axis=1)
    y = df["charges"]

    # Define categorical and numeric features
    categorical_features = ["sex", "smoker", "region"]
    numeric_features = ["age", "bmi", "children"]

    # Preprocessing
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first"), categorical_features)
    ], remainder='passthrough')

    # Pipeline
    model = Pipeline([
        ("preprocess", preprocessor),
        ("regressor", LinearRegression())
    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    return model, df

# Streamlit UI
def main():
    st.title("🩺 Insurance Charges Predictor")
    st.write("Enter your details to estimate medical insurance charges.")

    # Train model
    model, df = train_model()

    # User input
    age = st.slider("Age", 18, 100, 30)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    children = st.slider("Number of children", 0, 5, 0)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    # Predict button
    if st.button("Predict Charges"):
        input_data = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region
        }])
        prediction = model.predict(input_data)[0]

        st.success(f"💰 Estimated Insurance Charges: ${prediction:,.2f}")

        # Visualization
        fig = px.scatter(df, x="bmi", y="charges", color="smoker",
                         title="BMI vs Charges (Colored by Smoker)",
                         labels={"bmi": "BMI", "charges": "Charges"})
        fig.add_scatter(x=[bmi], y=[prediction], mode='markers',
                        marker=dict(size=12, color='red'),
                        name="Your Prediction")
        st.plotly_chart(fig)

    with st.expander("📊 View Dataset Sample"):
        st.dataframe(df.head())

if __name__ == '__main__':
    if 'get_ipython' in globals():
        get_ipython().system('streamlit run SQL_Week_6.py')
    else:
        main()
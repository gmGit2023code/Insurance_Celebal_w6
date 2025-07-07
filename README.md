Insurance Charges Predictor

A machine learning web app that predicts medical insurance costs based on user demographics and health factors.

![Streamlit App](https://img.shields.io/badge/Streamlit-Deployed-brightgreen)
![Python](https://img.shields.io/badge/Python-3.7%2B-blue)

Features:
- Predicts insurance costs using Linear Regression
- Interactive input sliders and dropdowns
- Visualizes predictions with interactive charts
- Shows dataset sample

Dataset:
The model is trained on insurance.csv (included in this repo) containing:
- 1,338 records
- 7 features: age, sex, bmi, children, smoker, region, charges

Requirements:
Dependencies are listed in requirements.txt:
streamlit
pandas
numpy
scikit-learn
plotly

Live Demo:
Deployed on Streamlit: 
[https://project.streamlit.app](https://insurancecelebalw6.streamlit.app/)

Run Locally:
1. Clone this repository
2. Install dependencies: pip install -r requirements.txt
3. Run the app: streamlit run app.py

How to Use:
1. Enter your details (age, BMI, smoking status, etc.)
2. Click "Predict Charges"
3. View your estimated insurance cost
4. See how your prediction compares to others in the dataset

Files:
- app.py - Main application code
- insurance.csv - Dataset
- requirements.txt - Dependency list

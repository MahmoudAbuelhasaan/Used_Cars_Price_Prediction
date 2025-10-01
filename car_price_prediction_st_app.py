import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Title
st.title("Car Price Prediction App")

# Load model & preprocessor
ml_model = pickle.load(open('model.pk', 'rb'))
preprocessor = pickle.load(open('preproccessor.pk', 'rb'))

# Load dataset (to fetch ranges and unique values for inputs)
X = pd.read_csv('cleaned_data.csv')

# User inputs
brand = st.selectbox('Brand', X['Brand'].unique())
car_model = st.selectbox('Model', X['Model'].unique())
kilometers_driven = st.number_input('Kilometers Driven', 
                                    min_value=int(X['Kilometers_Driven'].min()), 
                                    max_value=int(X['Kilometers_Driven'].max()))
fuel_type = st.selectbox('Fuel Type', X['Fuel_Type'].unique())
transmission = st.selectbox('Transmission', X['Transmission'].unique())
owner_type = st.selectbox('Owner Type', X['Owner_Type'].unique())
engine = st.number_input('Engine (CC)', 
                         min_value=float(X['Engine'].min()), 
                         max_value=float(X['Engine'].max()))
power = st.number_input('Power (bhp)', 
                        min_value=float(X['Power'].min()), 
                        max_value=float(X['Power'].max()))
seats = st.number_input('Seats', 
                        min_value=int(X['Seats'].min()), 
                        max_value=int(X['Seats'].max()))
location = st.selectbox('Location', X['Location'].unique())
mileage = st.number_input('Mileage (kmpl)', 
                          min_value=float(X['Mileage(kmpl)'].min()), 
                          max_value=float(X['Mileage(kmpl)'].max()))
age = st.number_input('Age of Car (years)', 
                      min_value=int(X['Age'].min()), 
                      max_value=int(X['Age'].max()))

# Build input dataframe
new_data = pd.DataFrame({
    'Location': [location],
    'Kilometers_Driven': [kilometers_driven],
    'Fuel_Type': [fuel_type],
    'Transmission': [transmission],
    'Owner_Type': [owner_type],
    'Engine': [engine],
    'Power': [power],
    'Seats': [seats],
    'Brand': [brand],
    'Model': [car_model],
    'Age': [age],
    'Mileage(kmpl)': [mileage]
})

# Preprocess and predict
if st.button("Predict Price"):
    try:
        data_prep = preprocessor.transform(new_data)
        pred = ml_model.predict(data_prep)
        st.success(f'The predicted price of the car is: {pred[0]:,.2f} Lakhs')
    except Exception as e:
        st.error(f"Error in prediction: {e}")

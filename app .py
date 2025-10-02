# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 10:14:20 2025

@author: Admin
"""

import numpy as np
import pickle
import pandas as pd
import streamlit as st

# Load the trained model
loaded_model = pickle.load(open('C:/Users/Admin/Desktop/deployed model/car_sales_data.pkl', 'rb'))

def car_price_prediction(engine_size, year_of_manufacture, mileage):
    # Create a dataframe for the model
    new_car = pd.DataFrame([{
        "Year of manufacture": year_of_manufacture,
        "Engine size": engine_size,
        "Mileage": mileage
    }])
    
    # Predict the price
    predicted_price = loaded_model.predict(new_car)
    return predicted_price[0]

def main():
    st.title('Car Price Prediction System')

    # Input fields
    engine_size = st.text_input('Enter the Engine size (e.g., 12)')
    year_of_manufacture = st.text_input('Enter the Year of Manufacture (e.g., 2020)')
    mileage = st.text_input('Enter the Mileage (e.g., 5000)')

    if st.button('Predict Price'):
        # Convert inputs to numeric types
        try:
            engine_size = float(engine_size)
            year_of_manufacture = int(year_of_manufacture)
            mileage = float(mileage)
            
            # Get prediction
            price = car_price_prediction(engine_size, year_of_manufacture, mileage)
            st.success(f'The predicted price for the car is: ${price:.2f}')
        except ValueError:
            st.error("Please enter valid numeric values for Engine size, Year, and Mileage.")

if __name__ == '__main__':
    main()

    
    
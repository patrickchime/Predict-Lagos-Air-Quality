# import packages
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error

import data # Modules

# Read in dataset
df = data.process_data("value_type", "timestamp","value", "P2")


app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Exploration', 'Prediction'])  # three pages

if app_mode == 'Home':
    st.title('Forecasting Particulate Matter (PM2.5) in Lagos')
    st.image('pm2.5.jfif')
    st.write("""Accurate monitoring of PM2.5 (Particulate Matter 2.5 micrometers or smaller in diameter) is indeed a crucial step towards
                pollution control, climate change control and overall environmental quality improvement in Lagos,Nigeria.
                The dataset used is sourced from [openAfrica](https://open.africa/dataset/sensorsafrica-airquality-archive-lagos).
            """)
    st.subheader('Preprocessed Dataset:')
    st.write(df.head())
    
    

elif app_mode == "Exploration":
    st.subheader("Exploratory Data Analysis")

    # Build a histogram chart
    st.markdown("Distribution chart of PM 2.5 in Lagos State")
    
    # Create and display the histogram
    fig = data.plot_hist(df)
    st.pyplot(fig)
    
    # Give space between charts
    st.markdown("---")
    # Build a Bar chart
    st.markdown("Hourly Trend of PM 2.5 in Lagos State")
    
    # Create and display the bar plot
    fig = data.hourly_trend(df)
    st.pyplot(fig)
    
    # Give space between charts
    st.markdown("---")
    # Build a Bar chart
    st.markdown("Weekly Trend of PM 2.5 in Lagos State")
    
    # Create and display the bar plot
    fig = data.weekly_trend(df)
    st.pyplot(fig)

    
else:
    st.subheader("Daily PM 2.5 Prediction in Lagos State")

    daily_data = data.resample_data()
        
 
    # Get the date input from the user
    date = st.date_input("Enter the date you want to get the PM 2.5 prediction for")
    
    # Calculate the difference between the entered date and the last date in the time series data
    last_date = daily_data.index[-1].date()  # Get the last date from the time series data
    date_difference = (date - last_date).days  # Calculate the difference in days


    
    # Instantiate make_prediction()
    fitted_model = data.make_prediction()
        
    # Text
    st.sidebar.title("WHO PM 2.5 Guideline")
    st.sidebar.write("1. The WHO guideline for the annual mean PM 2.5 concentration level is 5 µg/m3\n 2. Average exposures for 24-hour should not exceed 15µg/m3 more than 3-4 days per year")
    

    # forecasting
    forecast = fitted_model.get_forecast(steps = date_difference)
    
    # get forecasted values
    forecast_values = forecast.predicted_mean.round(2)
    
    # build dataframe of forecast values
    forecast_values.index = pd.date_range(start=daily_data.index[-1],
                                      periods=date_difference+1)[1:] # starts from the next day after last day in the dataset 

    # rename the index
    forecast_values.index.rename("future_date", inplace=True)

    # output prediction
    st.subheader(f'The predicted value of PM 2.5 in Lagos for "{date}" is:')
    st.header(f'{forecast_values[-1]} µg/m3')
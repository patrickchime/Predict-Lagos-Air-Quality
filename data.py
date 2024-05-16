import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error


# Define a function to read data
def read_data(pattern):
    # Read data

    # Get a list of filenames matching the pattern
    file_list = glob.glob(pattern)

    # Iterate over the files
    files = []
    for file in file_list:

        # Append file to datasets
        files.append(file)
        
    return files
    
    
# Define a function to process data    
def process_data(column1, column2, column3, value):
    """ This function accepts data from 
        read_data function and processess it
    """
    
    # Instantiate read_data()
    docs = read_data("datasets/lagos_air_quality*.csv")
    
    # Iterate over the datasets
    datasets = []
    for file in docs:
        df = pd.read_csv(file, sep=";")
        
        # Convert timestamp to datetime
        df[column2] = pd.to_datetime(df[column2])
        
        # Filter for pm2
        df = df[df[column1] == value]

        
        # feature engineering
        df["date"] = df[column2].dt.date
        df["weekday"] = df[column2].dt.weekday
        df["hour"] = df[column2].dt.hour
        
        # Convert date to datetime
        df["date"] = pd.to_datetime(df["date"])
        
        # Select features of interest
        df = df[["timestamp","weekday","hour",column3 ]]
        
        # Reset index
        df = df.reset_index(drop = True)
        
        # append all files together
        datasets.append(df)
    
    # Join all files together as one table
    pm2_df = pd.concat(datasets, ignore_index=True)
    return pm2_df


# Make a histogram chart of PM 2.5
def plot_hist(df):
    """
    Make a histogram chart of PM2.5
    """
    
    # Figure
    fig, ax = plt.subplots(figsize=(12,4))

    # Histogram of height in cm
    n, bin_edges, patches = ax.hist(df.value, rwidth=0.96, bins=15, color="#1E315C")

    # Get rid of top horizontal axis line and right vertical axis line
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["bottom","left"]].set_linewidth(0.2)

    # Axis labels
    ax.set_ylabel("Frequency", loc="top")
    ax.set_xlabel("Particulate Matter (PM2.5)", loc="left")
    ax.set_title("Distribution of PM2.5", loc="left");

    # the bin labels:
    bin_centers = 0.5 * np.diff(bin_edges) + bin_edges[:-1]

    labels = [f"< {bin_edges[1].astype(int)}"]  # The first label
    labels += [f"{bin_edges[i].astype(int)+1} - {bin_edges[i + 1].astype(int)}" for i in range(1, len(bin_edges) - 2)]
    labels += [f"{bin_edges[-2].astype(int)+1} +"] # The last label

    ax.set_xticks(bin_centers, labels, rotation=45)
    ax.bar_label(patches)
    
    return fig;


# Make bar chart showing hourly trend
def hourly_trend(df):
    
    hourly_trend = df.groupby("hour")["value"].mean()

    # Make a line plot of PM2.5
    fig, ax = plt.subplots(figsize=(12,5))

    # Bar plot
    hourly_trend.plot(kind="bar", ax=ax, color="#1E315C")

    # Add data labels
    mean_values = list(hourly_trend.values)

    for i, mean_value in enumerate(mean_values):
        plt.annotate(str(round(mean_value)), xy=(i, mean_value), ha = "center", va = "bottom")

    # Get rid of top horizontal axis line and right vertical axis line
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["bottom","left"]].set_linewidth(0.2)


    # Axis labels
    ax.set_ylabel("PM2.5 (micrometers)", loc="top")
    ax.set_xlabel("Hour", loc="left")
    ax.set_title("Hourly Trends in Particulate Matter 2.5", loc="left")
    
    return fig;

# Define a weekly trend function:
def weekly_trend(df):
    
    # Map the week numbers to their corresponding names
    week_names = {0 : "Monday",
               1 : "Tuesday",
               2 : "Wednesday",
               3 : "Thursday",
               4 : "Friday",
               5 : "Saturday",
               6 : "Sunday"}

    df["week_name"] = df["weekday"].map(week_names)

        # Calculate the mean PM2.5 value for each day of the week
    daily_trend = df.groupby("week_name")["value"].mean()

    # Define the desired order of weekdays
    week_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Reorder the index of the daily_trend Series
    daily_trend_sorted = daily_trend.reindex(week_order)

    # Make a bar plot of PM2.5
    fig, ax = plt.subplots(figsize=(12, 4))

    # Bar plot
    daily_trend_sorted.plot(kind="bar", ax=ax, color="#1E315C")

    # Add data labels
    mean_values = list(daily_trend_sorted.values)
    for i, mean_value in enumerate(mean_values):
        plt.annotate(str(round(mean_value, 2)), xy=(i, mean_value), ha="center", va="bottom")

    # Get rid of top horizontal axis line and right vertical axis line
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Set axis labels and title
    ax.set_ylabel("PM2.5 (micrometers)", loc="top")
    ax.set_xlabel("Weekday", loc="left")
    ax.set_title("Average Daily Trends in Particulate Matter 2.5", loc="left")

    # Rotate x-axis labels
    ax.tick_params(axis='x', rotation=45)
    
    return fig;


# Define a function to resample data
def resample_data():
    # Instantiate process_data()
    df = process_data("value_type", "timestamp","value", "P2")
    
    # Drop unnecesary features
    df = df.drop(columns=["weekday", "hour"],axis=1)
    df.set_index('timestamp', inplace=True)

    # Daily Resampling
    daily_data = df.resample('D').mean().fillna(method='ffill')  # fill missing values with forward fill
    
    # Squeeze
    daily_data = daily_data.squeeze()
    
    # Return daily_data
    return daily_data
    

   

@st.cache_data
def make_prediction():

    # Instantiate resample_data()
    daily_data = resample_data()
    
    # split data
    cut_off = int(len(daily_data) *0.8) # series data is split chronologically in a horizontal manner
    y_train = daily_data.iloc[:cut_off]
    y_test =  daily_data.iloc[cut_off:]

    
    # Define SARIMAX model
    order = (2, 1, 2)  # (p, d, q)
    seasonal_order = (1, 1, 1, 7)  # (P, D, Q, S)
    model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order)

    # Fit SARIMAX model
    fitted_model = model.fit()

    # Predict
    pred = fitted_model.forecast(steps=len(y_test))

    # Calculate the mean absolute error 
    mae = mean_absolute_error(y_test, pred)
    st.sidebar.title("Evaluation Metric")
    st.sidebar.write(f"Mean Absolute Error: {round(mae,2)} µg/m3")
    
    return fitted_model

    
    
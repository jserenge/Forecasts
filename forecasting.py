import streamlit as st
import pandas as pd
import itertools
import warnings
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def run_sarimax_model(df, date_column):
    # Remove the date column from the list of columns to forecast
    columns_to_forecast = df.columns.tolist()
    columns_to_forecast.remove(date_column)

    # Define the p, d, q parameters to take any value between 0 and 2
    p = d = q = range(0, 2)

    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    # Specify to ignore warning messages
    warnings.filterwarnings("ignore")

    # Create a DataFrame to store the forecasts
    forecast_df = pd.DataFrame()

    for column in columns_to_forecast:
        series = df[column]

        best_aic = np.inf
        best_pdq = None
        best_seasonal_pdq = None
        temp_model = None

        for param in pdq:
            for param_seasonal in seasonal_pdq:

                try:
                    temp_model = SARIMAX(series,
                                         order=param,
                                         seasonal_order=param_seasonal,
                                         enforce_stationarity=False,
                                         enforce_invertibility=False)
                    results = temp_model.fit()

                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_pdq = param
                        best_seasonal_pdq = param_seasonal
                except:
                    continue

        print(f"Best SARIMAX{best_pdq}x{best_seasonal_pdq}12 model for {column} - AIC:{best_aic}")

        # Fit the SARIMA model
        model = SARIMAX(series, 
                        order=best_pdq,
                        seasonal_order=best_seasonal_pdq)
        results = model.fit()

        # Get forecast 120 steps ahead in future (10 years)
        forecast = results.get_forecast(steps=120)

        # Get confidence intervals of forecasts
        forecast_ci = forecast.conf_int()

        # Store the forecast in the DataFrame
        forecast_df[column] = forecast.predicted_mean

    return forecast_df

def calculate_weighted_index(df, weights):
    weighted_index = (df * weights).sum(axis=1) / weights.sum()
    return weighted_index

st.title('Forecasting App')

uploaded_file = st.file_uploader("Upload your input CSV file", type="csv")

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    st.write(input_df)

    date_column = st.selectbox('Select the date column', input_df.columns.tolist())

    if st.button('Run Forecast'):
        with st.spinner('Running the SARIMAX model...'):
            forecast_df = run_sarimax_model(input_df, date_column)

        st.write(forecast_df)

        # Prompt the user to input weights for each variable
        weights = []
        for column in forecast_df.columns:
            weight = st.number_input(f"Enter weight for {column}", value=1.0)
            weights.append(weight)

        # Calculate and display the weighted average index
        weighted_index = calculate_weighted_index(forecast_df, np.array(weights))
        st.write("Weighted Average Index")
        st.write(weighted_index)

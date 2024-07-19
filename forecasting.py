import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools
import warnings

def run_sarimax_model(series):
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    warnings.filterwarnings("ignore")

    best_aic = np.inf
    best_pdq = None
    best_seasonal_pdq = None

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

    model = SARIMAX(series, 
                    order=best_pdq,
                    seasonal_order=best_seasonal_pdq)
    results = model.fit()
    forecast = results.get_forecast(steps=12)
    forecast_ci = forecast.conf_int()
    return forecast.predicted_mean, forecast_ci

st.title('Forecasting and Index Creation App')

uploaded_file = st.file_uploader("Upload your input CSV file", type="csv")

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    st.write(input_df)

    target_variable = st.selectbox('Select the target variable', input_df.columns.tolist())

    if st.button('Run Index-Based Forecast'):
        with st.spinner('Creating Index and Forecasting...'):
            # Define the predictors (excluding the target variable)
            predictors = input_df.drop(columns=[target_variable])
            # Run a linear regression model
            reg_model = LinearRegression()
            reg_model.fit(predictors, input_df[target_variable])
            coefficients = reg_model.coef_

            # Create the index (weighted average based on coefficients)
            index = np.dot(predictors, coefficients)

            # Forecast the index using SARIMAX
            forecasted_index, forecast_ci = run_sarimax_model(index)

            st.write("Forecasted Index:")
            st.write(forecasted_index)

    if st.button('Run Direct Forecast'):
        with st.spinner('Running SARIMAX model...'):
            forecasted_target, forecast_ci = run_sarimax_model(input_df[target_variable])
            st.write("Forecasted Target Variable:")
            st.write(forecasted_target)

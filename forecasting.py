import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go

# Function to normalize data
def normalize_data(data):
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

def calculate_weights(data):
    variances = np.var(data, axis=0)
    weights = variances / np.sum(variances)
    return weights

def calculate_weighted_sums(data, weights):
    normalized_data = normalize_data(data)
    weighted_sums = np.dot(normalized_data, weights)
    multipliers = weighted_sums / weighted_sums[0]
    return multipliers

def forecast_multipliers(multipliers, periods=10):
    model = ExponentialSmoothing(multipliers, trend='add', seasonal=None)
    fit = model.fit()
    forecast = fit.forecast(periods)
    return forecast

st.title('Cost Forecasting App')

uploaded_file = st.file_uploader("Upload your input CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded DataFrame:")
        st.write(df)
        
        cost_columns = st.multiselect('Select the cost components', df.columns[1:].tolist())
        
        if cost_columns:
            data = df[cost_columns].values
            st.write("Selected Data:")
            st.write(data)
            
            weights = calculate_weights(data)
            st.write("Calculated Weights:")
            st.write(weights)
            
            multipliers = calculate_weighted_sums(data, weights)
            st.write("Calculated Multipliers:")
            st.write(multipliers)
            
            forecast_periods = st.slider('Select number of periods to forecast', 1, 20, 10)
            forecast = forecast_multipliers(multipliers, forecast_periods)
            st.write("Forecasted Multipliers:")
            st.write(forecast)
            
            years = df['Year'].values
            future_years = np.arange(years[-1] + 1, years[-1] + 1 + len(forecast))
            all_years = np.concatenate([years, future_years])
            all_multipliers = np.concatenate([multipliers, forecast])
            
            st.write("All Years and Multipliers:")
            result_df = pd.DataFrame({'Year': all_years, 'Multiplier': all_multipliers})
            st.write(result_df)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=years, y=multipliers, mode='lines+markers', name='Historical'))
            fig.add_trace(go.Scatter(x=future_years, y=forecast, mode='lines+markers', name='Forecast'))
            fig.update_layout(title='Cost Multipliers Over Time', xaxis_title='Year', yaxis_title='Multiplier')
            st.plotly_chart(fig)
            
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="Download forecast as CSV",
                data=csv,
                file_name="forecast.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please upload a CSV file to begin.")

st.sidebar.header("About")
st.sidebar.info("This app calculates cost multipliers based on historical data and forecasts future values.")
st.sidebar.header("Instructions")
st.sidebar.info("1. Upload a CSV file with your cost data.\n2. Select the relevant cost columns.\n3. Adjust the forecast period if needed.\n4. View the results and download if desired.")

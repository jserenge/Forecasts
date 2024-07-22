import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
from io import BytesIO

# Function to normalize data
def normalize_data(data):
    normalized = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    st.write("Normalized Data (Debugging):")
    st.write(normalized)
    return normalized

# Function to calculate weights
def calculate_weights(data):
    variances = np.var(data, axis=0)
    st.write("Variances (Debugging):")
    st.write(variances)
    
    weights = variances / np.sum(variances)
    st.write("Weights (Debugging):")
    st.write(weights)
    return weights

# Function to calculate weighted sums
def calculate_weighted_sums(data, weights):
    normalized_data = normalize_data(data)
    weighted_sums = np.dot(normalized_data, weights)
    
    st.write("Normalized Data (Debugging):")
    st.write(normalized_data)
    
    st.write("Weighted Sums (Debugging):")
    st.write(weighted_sums)
    
    multipliers = weighted_sums / weighted_sums[0]
    st.write("Multipliers (Debugging):")
    st.write(multipliers)
    return multipliers

# Function to forecast multipliers
def forecast_multipliers(multipliers, periods=10):
    model = ExponentialSmoothing(multipliers, trend='add', seasonal=None)
    fit = model.fit()
    forecast = fit.forecast(periods)
    return forecast

# Function to convert DataFrame to Excel
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='openpyxl')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer._save()
    processed_data = output.getvalue()
    return processed_data

# Main app
st.title('Cost Forecasting App')

# File uploader
uploaded_file = st.file_uploader("Upload your input Excel file", type="xlsx")

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
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
            
            # Plotting the results
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=years, y=multipliers, mode='lines+markers', name='Historical'))
            fig.add_trace(go.Scatter(x=future_years, y=forecast, mode='lines+markers', name='Forecast'))
            fig.update_layout(title='Cost Multipliers Over Time', xaxis_title='Year', yaxis_title='Multiplier')
            st.plotly_chart(fig)
            
            # Convert DataFrame to Excel
            excel_data = to_excel(result_df)
            
            # Download button for Excel
            st.download_button(
                label="Download forecast as Excel",
                data=excel_data,
                file_name="forecast.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please upload an Excel file to begin.")

# Sidebar information
st.sidebar.header("About")
st.sidebar.info("This app calculates cost multipliers based on historical data and forecasts future values.")
st.sidebar.header("Instructions")
st.sidebar.info("1. Upload an Excel file with your cost data.\n2. Select the relevant cost columns.\n3. Adjust the forecast period if needed.\n4. View the results and download if desired.")

import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
from io import BytesIO

def normalize_data(data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    if np.any(max_vals - min_vals == 0):
        st.error("Normalization error: Some columns have zero variance.")
        return None
    normalized = (data - min_vals) / (max_vals - min_vals)
    st.write("Normalized Data (first 5 rows):")
    st.write(normalized[:5])
    return normalized

def calculate_weights(data):
    variances = np.var(data, axis=0)
    if np.any(variances == 0):
        st.error("Weight calculation error: Some columns have zero variance.")
        return None
    st.write("Variances:")
    st.write(variances)
    
    weights = variances / np.sum(variances)
    st.write("Weights:")
    st.write(weights)
    return weights

def calculate_weighted_sums(data, weights):
    st.write("Data shape:", data.shape)
    st.write("Weights shape:", weights.shape)
    
    normalized_data = normalize_data(data)
    if normalized_data is None:
        return None
    
    st.write("Normalized data shape:", normalized_data.shape)
    
    weighted_sums = np.dot(normalized_data, weights)
    st.write("Weighted sums shape:", weighted_sums.shape)
    st.write("Weighted sums (first 5 values):")
    st.write(weighted_sums[:5])
    
    if len(weighted_sums) > 1:
        if weighted_sums[1] == 0:
            st.error("Weighted sums calculation error: Initial value after skipping the first row is zero.")
            return None
        multipliers = weighted_sums[1:] / weighted_sums[1]
    else:
        st.error("Weighted sums array is empty")
        return None
    
    st.write("Multipliers shape:", multipliers.shape)
    st.write("Multipliers (first 5 values):")
    st.write(multipliers[:5])
    
    return multipliers

def forecast_multipliers(multipliers, periods=10):
    st.write("Input multipliers for forecasting (first 5 values):", multipliers[:5])
    st.write("Forecasting for periods:", periods)
    
    if len(multipliers) < 2:
        st.error("Not enough data points for forecasting")
        return None
    
    model = ExponentialSmoothing(multipliers, trend='add', seasonal=None)
    fit = model.fit()
    forecast = fit.forecast(periods)
    
    st.write("Forecast result (first 5 values):", forecast[:5])
    return forecast

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

st.title('Cost Forecasting App')

uploaded_file = st.file_uploader("Upload your input Excel file", type="xlsx")

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        st.write("Uploaded DataFrame (first 5 rows):")
        st.write(df.head())
        
        st.write("Data types of columns:")
        st.write(df.dtypes)
        
        df.rename(columns={df.columns[0]: 'Year'}, inplace=True)
        
        cost_columns = st.multiselect('Select the cost components', df.columns[1:].tolist())
        
        if cost_columns:
            data = df[cost_columns].values  # Do not skip the first row yet
            st.write("Selected Data (first 5 rows):")
            st.write(data[:5])
            
            weights = calculate_weights(data[1:])  # Skip the first row for weight calculation
            if weights is None:
                st.stop()
            
            multipliers = calculate_weighted_sums(data[1:], weights)  # Skip the first row for weighted sums calculation
            if multipliers is not None:
                st.write("Calculated Multipliers (first 5 values):")
                st.write(multipliers[:5])
            else:
                st.error("Failed to calculate multipliers")
                st.stop()
            
            forecast_periods = st.slider('Select number of periods to forecast', 1, 20, 10)
            forecast = forecast_multipliers(multipliers, forecast_periods)
            if forecast is not None:
                st.write("Forecasted Multipliers (first 5 values):")
                st.write(forecast[:5])
            else:
                st.error("Failed to forecast multipliers")
                st.stop()
            
            years = df['Year'].values[1:]  # Skip the first year
            future_years = np.arange(years[-1] + 1, years[-1] + 1 + len(forecast))
            all_years = np.concatenate([years, future_years])
            all_multipliers = np.concatenate([multipliers, forecast])
            
            st.write("All Years and Multipliers (first 5 rows):")
            result_df = pd.DataFrame({'Year': all_years, 'Multiplier': all_multipliers})
            st.write(result_df.head())
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=years, y=multipliers, mode='lines+markers', name='Historical'))
            fig.add_trace(go.Scatter(x=future_years, y=forecast, mode='lines+markers', name='Forecast'))
            fig.update_layout(title='Cost Multipliers Over Time', xaxis_title='Year', yaxis_title='Multiplier')
            st.plotly_chart(fig)
            
            excel_data = to_excel(result_df)
            
            st.download_button(
                label="Download forecast as Excel",
                data=excel_data,
                file_name="forecast.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("Please select at least one cost component.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error(f"Error details: {e.__class__.__name__}")
        import traceback
        st.error(traceback.format_exc())
else:
    st.info("Please upload an Excel file to begin.")

st.sidebar.header("About")
st.sidebar.info("This app calculates cost multipliers based on historical data and forecasts future values.")
st.sidebar.header("Instructions")
st.sidebar.info("1. Upload an Excel file with your cost data.\n2. Select the relevant cost columns.\n3. Adjust the forecast period if needed.\n4. View the results and download if desired.")

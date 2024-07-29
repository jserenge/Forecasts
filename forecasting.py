import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.decomposition import PCA
from io import BytesIO

def load_and_process_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        df.rename(columns={df.columns[0]: 'Year'}, inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def apply_pca(data):
    pca = PCA(n_components=1)
    principal_components = pca.fit_transform(data)
    return principal_components.flatten()

def forecast_multipliers(multipliers, periods):
    model = ExponentialSmoothing(multipliers, trend='add', seasonal=None)
    fit = model.fit()
    forecast = fit.forecast(periods)
    return forecast

def create_result_df(years, multipliers, forecast):
    all_years = np.concatenate([years, np.arange(years[-1] + 1, years[-1] + 1 + len(forecast))])
    all_multipliers = np.concatenate([multipliers, forecast])
    result_df = pd.DataFrame({'Year': all_years, 'Multiplier': all_multipliers})
    return result_df

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

def main():
    st.title('Cost Forecasting App')

    st.sidebar.header("About")
    st.sidebar.info("This app calculates cost multipliers based on historical data and forecasts future values.")

    st.sidebar.header("Instructions")
    st.sidebar.info("1. Upload an Excel file with your cost data.\n2. Select the relevant cost columns.\n3. View the results and download if desired.")

    uploaded_file = st.file_uploader("Upload your input Excel file", type="xlsx")

    if uploaded_file is not None:
        df = load_and_process_data(uploaded_file)
        if df is not None:
            st.write("Uploaded DataFrame (first 5 rows):")
            st.write(df.head())

            cost_columns = st.multiselect('Select the cost components', df.columns[1:].tolist())
            if cost_columns:
                data = df[cost_columns].values
                multipliers = apply_pca(data)
                forecast_periods = 10
                forecast = forecast_multipliers(multipliers, forecast_periods)

                years = df['Year'].values
                result_df = create_result_df(years, multipliers, forecast)
                st.write("Final DataFrame (first 10 rows):")
                st.write(result_df.head())

                excel_data = to_excel(result_df)
                st.download_button(
                    label="Download forecast as Excel",
                    data=excel_data,
                    file_name="forecast.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("Please select at least one cost component.")
        else:
            st.error("Error loading data")

if __name__ == "__main__":
    main()

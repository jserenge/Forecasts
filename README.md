**Cost Forecasting App**
---

**Overview**

The Cost Forecasting App is a web-based tool built with Streamlit that enables users to analyze historical cost data, extract key cost multipliers using PCA (Principal Component Analysis), and forecast future values using the SARIMAX (Seasonal Autoregressive Integrated Moving Average with Exogenous Regressors) model.

---

**Features**

Upload an Excel file containing historical cost data.

Select relevant cost components for analysis.

Apply PCA to reduce dimensionality and extract principal multipliers.

Generate lag features for time series forecasting.

Use SARIMAX to predict future cost multipliers.

Display and download the forecasted results as an Excel file.

---

**Installation**
Prerequisites
Ensure you have Python installed along with the following dependencies:

pip install streamlit pandas numpy statsmodels scikit-learn openpyxl

---

**How to Use**
Run the application:

streamlit run app.py

Upload an Excel file containing cost data.

Select cost components to analyze.

View the processed data and forecasted multipliers.

Download the results as an Excel file.

---
**File Format Requirements**
The uploaded file must be an Excel file (.xlsx).

The first column should contain the "Year".

The subsequent columns should contain cost components.

---

**Code Structure**

load_and_process_data(uploaded_file): Loads and processes the uploaded Excel file.

apply_pca(data): Extracts principal components using PCA.

create_lag_features(data, lags): Creates lag features for forecasting.

forecast_multipliers(multipliers, periods): Forecasts future multipliers using SARIMAX.

create_result_df(years, multipliers, forecast): Constructs a DataFrame with actual and forecasted multipliers.

to_excel(df): Converts the results into an Excel file.

main(): Streamlit UI logic to interact with users.

**Future Enhancements**

Allow users to adjust forecasting parameters.

Enable visualization of trends and predictions.

Support for additional forecasting models.

**License**

This project is open-source and available for use and modification.

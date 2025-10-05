import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import xgboost as xgb
from sklearn.model_selection import train_test_split
from dateutil import parser
import warnings

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(page_title="Retail Sales Forecasting", layout="wide")
st.title("📈 Retail Sales Forecasting Dashboard")
st.markdown("""
Easily forecast future sales using **XGBoost** (for product-level prediction)
and **Prophet** (for trend-based forecasting). Upload your data and start exploring!
""")

# ============================================================
# HELPER FUNCTION — Robust Date Parser
# ============================================================
def smart_parse_dates(date_series):
    parsed_dates = []
    for d in date_series:
        try:
            parsed = parser.parse(str(d), dayfirst=True)
            parsed_dates.append(parsed)
        except Exception:
            parsed_dates.append(pd.NaT)
    return pd.to_datetime(parsed_dates, errors='coerce')


# ============================================================
#  UPLOAD DATA
# ============================================================
uploaded_file = st.file_uploader("📤 Upload your sales CSV file", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'date' not in df.columns:
        st.error(" The file must contain a 'date' column.")
        st.stop()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df['date'] = smart_parse_dates(df['date'])

    invalid_count = df['date'].isna().sum()
    if invalid_count > 0:
        st.warning(f"{invalid_count} rows had invalid or missing dates and were removed.")
        df = df.dropna(subset=['date'])

    st.success("Data loaded successfully!")
    st.write("### Sample Data")
    st.dataframe(df.head())

    # ============================================================
    # 2️⃣ FEATURE ENGINEERING
    # ============================================================
    df = df.sort_values(['store', 'product_id', 'date']).reset_index(drop=True)
    df['promotion_flag'] = (df['discount'] > 0).astype(int)
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    df['lag_7'] = df.groupby(['store', 'product_id'])['units_sold'].shift(7)
    df['rolling_7'] = df.groupby(['store', 'product_id'])['units_sold'].transform(lambda x: x.shift(1).rolling(7).mean())
    df_ml = df.dropna(subset=['lag_7', 'rolling_7'])

    features = ['day_of_week', 'month', 'is_weekend', 'promotion_flag', 'price', 'discount', 'lag_7', 'rolling_7']
    X = df_ml[features]
    y = df_ml['units_sold']

    # ============================================================
    # 3️⃣ TRAIN XGBOOST MODEL
    # ============================================================
    model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X, y)

    # ============================================================
    # 4️⃣ USER INPUT: FORECAST SETTINGS
    # ============================================================
    st.header("Forecast Configuration")

    col1, col2, col3 = st.columns(3)
    with col1:
        store_selected = st.selectbox("Select Store", df['store'].unique())
    with col2:
        product_selected = st.selectbox("Select Product", df[df['store'] == store_selected]['product_id'].unique())
    with col3:
        forecast_days = st.slider("Forecast Days", min_value=7, max_value=90, value=30)

    st.markdown("### Promotion Settings")
    promo_choice = st.radio(
        "Promotion Plan:",
        ["No promotions", "All days on promotion", "Custom date range"]
    )

    if promo_choice == "Custom date range":
        promo_start = st.date_input("Promotion Start Date")
        promo_end = st.date_input("Promotion End Date")

    # ============================================================
    # 5️⃣ GENERATE FUTURE DATA FOR FORECAST
    # ============================================================
    last_row = df_ml[(df_ml['store'] == store_selected) & (df_ml['product_id'] == product_selected)].tail(1)
    future_dates = pd.date_range(start=last_row['date'].iloc[0] + pd.Timedelta(days=1), periods=forecast_days)

    future_df = pd.DataFrame({
        'date': future_dates,
        'store': store_selected,
        'product_id': product_selected,
        'price': last_row['price'].iloc[0],
        'discount': last_row['discount'].iloc[0],
        'day_of_week': future_dates.dayofweek,
        'month': future_dates.month,
        'is_weekend': future_dates.dayofweek.isin([5, 6]).astype(int),
        'lag_7': last_row['lag_7'].iloc[0],
        'rolling_7': last_row['rolling_7'].iloc[0]
    })

    # Apply promotion logic
    if promo_choice == "No promotions":
        future_df['promotion_flag'] = 0
    elif promo_choice == "All days on promotion":
        future_df['promotion_flag'] = 1
    else:
        future_df['promotion_flag'] = future_df['date'].between(pd.to_datetime(promo_start), pd.to_datetime(promo_end)).astype(int)

    # ============================================================
    # 6️⃣ MAKE PREDICTIONS (XGBOOST)
    # ============================================================
    X_future = future_df[features]
    future_df['predicted_sales_xgb'] = model.predict(X_future)

    st.subheader(f"XGBoost Forecast — {store_selected} | {product_selected}")
    st.line_chart(future_df.set_index('date')['predicted_sales_xgb'])
    st.dataframe(future_df[['date', 'predicted_sales_xgb']].head(20))

    # ============================================================
    # 7️⃣ PROPHET FORECAST (TOTAL SALES TREND)
    # ============================================================
    st.header("Prophet Forecast (Total Daily Sales)")

    df_prophet = df.groupby('date').agg({'units_sold': 'sum', 'promotion_flag': 'max'}).reset_index()
    df_prophet.columns = ['ds', 'y', 'promotion_flag']

    m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    m.add_regressor('promotion_flag')
    m.fit(df_prophet)

    future_prophet = m.make_future_dataframe(periods=forecast_days)
    future_prophet = future_prophet.merge(df_prophet[['ds', 'promotion_flag']], on='ds', how='left').fillna(0)
    forecast = m.predict(future_prophet)

    st.line_chart(forecast.set_index('ds')['yhat'])
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days))

    # ============================================================
    # 8️⃣ DOWNLOAD FORECAST
    # ============================================================
    st.download_button(
        "Download XGBoost Forecast CSV",
        data=future_df.to_csv(index=False),
        file_name=f"{store_selected}_{product_selected}_forecast.csv",
        mime="text/csv"
    )

    st.success("Forecast generated successfully!")
else:
    st.info("Upload a CSV file to begin forecasting.")
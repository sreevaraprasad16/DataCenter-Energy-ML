import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

import streamlit as st
from streamlit_autorefresh import st_autorefresh
from supabase import create_client


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Data Center Energy Monitor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROFESSIONAL PAGE STYLE
# ============================================================

st.markdown(
    """
    <style>

    html, body, [class*="css"] {
        font-family: "Segoe UI", Arial, sans-serif;
    }

    .main-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 16px;
        color: #666666;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 650;
        margin-top: 20px;
        margin-bottom: 12px;
    }

    .info-text {
        font-size: 14px;
        color: #666666;
    }

    section[data-testid="stSidebar"] {
        padding-top: 1.5rem;
    }

    div[data-testid="stMetric"] {
        border-radius: 10px;
        padding: 12px;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SUPABASE CONNECTION
# ============================================================

supabase = create_client(
    st.secrets["SUPABASE_URL"],
    st.secrets["SUPABASE_KEY"]
)


# ============================================================
# AUTOMATIC REFRESH
# ============================================================

st_autorefresh(
    interval=30000,
    key="data_center_refresh"
)


# ============================================================
# LOAD MACHINE LEARNING MODEL
# ============================================================

model = joblib.load(
    "random_forest_clean_model.pkl"
)


# ============================================================
# LOAD ANOMALY DETECTION MODEL
# ============================================================

anomaly_model = joblib.load(
    "anomaly_detection_model.pkl"
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

model_r2 = 0.9061
model_mae = 0.0197
model_rmse = 0.0253


# ============================================================
# LOAD MONITORING DATA
# ============================================================

monitoring_columns = [
    "Device_ID",
    "Timestamp",
    "Temperature_C",
    "Humidity_Percent",
    "IT_Load_kW",
    "Cooling_Power_kW",
    "Total_Power_kW",
    "PUE"
]


response = (
    supabase
    .table("monitoring_data")
    .select(",".join(monitoring_columns))
    .order("Timestamp", desc=True)
    .limit(100)
    .execute()
)


monitoring_df = pd.DataFrame(
    response.data
)


# ============================================================
# CHECK DATA
# ============================================================

if monitoring_df.empty:

    st.error(
        "No monitoring data is available."
    )

    st.info(
        "Make sure the MQTT simulator and ingestion service "
        "are running and that Supabase contains monitoring records."
    )

    st.stop()


# ============================================================
# DATA CLEANING
# ============================================================

monitoring_df["Timestamp"] = pd.to_datetime(
    monitoring_df["Timestamp"],
    errors="coerce"
)


numeric_columns = [
    "Temperature_C",
    "Humidity_Percent",
    "IT_Load_kW",
    "Cooling_Power_kW",
    "Total_Power_kW",
    "PUE"
]


for column in numeric_columns:

    monitoring_df[column] = pd.to_numeric(
        monitoring_df[column],
        errors="coerce"
    )


monitoring_df = (
    monitoring_df
    .dropna(
        subset=[
            "Timestamp",
            "Temperature_C",
            "Humidity_Percent",
            "IT_Load_kW",
            "Cooling_Power_kW",
            "PUE"
        ]
    )
    .sort_values("Timestamp")
    .reset_index(drop=True)
)


# ============================================================
# GENERATE ML PREDICTIONS
# ============================================================
# IMPORTANT:
# Predicted_PUE is calculated by the Random Forest locally.
# It is NOT required to exist in Supabase.

prediction_features = [
    "Temperature_C",
    "Humidity_Percent",
    "IT_Load_kW",
    "Cooling_Power_kW"
]


prediction_input = monitoring_df[
    prediction_features
].copy()


monitoring_df["Predicted_PUE"] = (
    model.predict(prediction_input)
)


monitoring_df["Predicted_PUE"] = (
    monitoring_df["Predicted_PUE"]
    .round(3)
)


# ============================================================
# PUE FORECASTING
# ============================================================

forecast_data = monitoring_df[
    ["Timestamp", "PUE"]
].copy()


forecast_data["PUE"] = pd.to_numeric(
    forecast_data["PUE"],
    errors="coerce"
)


forecast_data = (
    forecast_data
    .dropna(subset=["PUE"])
    .reset_index(drop=True)
)


forecast_data["Time_Index"] = range(
    len(forecast_data)
)


forecast_model = LinearRegression()


forecast_model.fit(
    forecast_data[["Time_Index"]],
    forecast_data["PUE"]
)


forecast_steps = 6


last_index = forecast_data[
    "Time_Index"
].iloc[-1]


future_indexes = range(
    last_index + 1,
    last_index + forecast_steps + 1
)


future_forecast = pd.DataFrame(
    {
        "Time_Index": future_indexes
    }
)


future_forecast["Forecasted_PUE"] = (
    forecast_model.predict(
        future_forecast[["Time_Index"]]
    )
)


future_forecast["Forecasted_PUE"] = (
    future_forecast["Forecasted_PUE"]
    .round(3)
)


# ============================================================
# LATEST READING
# ============================================================

latest = monitoring_df.iloc[-1]


current_temperature = float(
    latest["Temperature_C"]
)


current_humidity = float(
    latest["Humidity_Percent"]
)


current_it_load = float(
    latest["IT_Load_kW"]
)


current_cooling_power = float(
    latest["Cooling_Power_kW"]
)


current_total_power = float(
    latest["Total_Power_kW"]
)


current_pue = float(
    latest["PUE"]
)


# ============================================================
# CURRENT LIVE ML PREDICTION
# ============================================================

current_prediction_input = pd.DataFrame(
    [
        {
            "Temperature_C": current_temperature,
            "Humidity_Percent": current_humidity,
            "IT_Load_kW": current_it_load,
            "Cooling_Power_kW": current_cooling_power
        }
    ]
)


current_predicted_pue = float(
    model.predict(
        current_prediction_input
    )[0]
)


current_predicted_pue = round(
    current_predicted_pue,
    3
)


# ============================================================
# PREDICTION ERROR
# ============================================================

prediction_error = abs(
    current_pue - current_predicted_pue
)


# ============================================================
# ML ANOMALY PREDICTION
# ============================================================

anomaly_input = pd.DataFrame(
    [
        {
            "Temperature_C": current_temperature,
            "Humidity_Percent": current_humidity,
            "IT_Load_kW": current_it_load,
            "Cooling_Power_kW": current_cooling_power,
            "PUE": current_pue
        }
    ]
)


ml_anomaly_prediction = anomaly_model.predict(
    anomaly_input
)[0]


ml_anomaly_score = anomaly_model.decision_function(
    anomaly_input
)[0]


ml_anomaly_score = round(
    float(ml_anomaly_score),
    4
)


if ml_anomaly_prediction == -1:

    ml_anomaly_status = "Anomaly Detected"

else:

    ml_anomaly_status = "Normal"


# ============================================================
# RULE-BASED ANOMALY DETECTION
# ============================================================

cooling_ratio = (
    current_cooling_power /
    current_it_load
)


anomaly_reasons = []


if current_pue > 1.50:

    anomaly_reasons.append(
        "High PUE"
    )


if current_temperature > 28:

    anomaly_reasons.append(
        "High Temperature"
    )


if cooling_ratio > 0.30:

    anomaly_reasons.append(
        "High Cooling Ratio"
    )


if prediction_error > 0.10:

    anomaly_reasons.append(
        "High Prediction Error"
    )


if anomaly_reasons:

    anomaly_status = "Anomaly Detected"

else:

    anomaly_status = "Normal"


# ============================================================
# PUE ALERT
# ============================================================

if current_pue <= 1.30:

    pue_status = "GOOD"

    pue_alert = (
        "PUE is within the efficient range."
    )

elif current_pue <= 1.50:

    pue_status = "WARNING"

    pue_alert = (
        "PUE is moderately high. "
        "Monitor energy usage."
    )

else:

    pue_status = "CRITICAL"

    pue_alert = (
        "PUE is high. Immediate energy-efficiency "
        "attention is recommended."
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## Data Center Monitor"
    )

    st.caption(
        "Energy efficiency and PUE monitoring"
    )

    st.divider()

    st.markdown(
        "### System"
    )

    st.write(
        "MQTT Telemetry"
    )

    st.write(
        "Supabase PostgreSQL"
    )

    st.write(
        "Random Forest ML"
    )

    st.write(
        "Streamlit Dashboard"
    )

    st.divider()

    st.markdown(
        "### ML Inputs"
    )

    st.write(
        "Temperature"
    )

    st.write(
        "Humidity"
    )

    st.write(
        "IT Load"
    )

    st.write(
        "Cooling Power"
    )

    st.divider()

    st.markdown(
        "### Navigation"
    )

    st.markdown(
        "[Overview](#overview)"
    )

    st.markdown(
        "[Live Monitoring](#live-monitoring)"
    )

    st.markdown(
        "[PUE Prediction](#pue-prediction)"
    )

    st.markdown(
        "[Model Performance](#model-performance)"
    )

    st.markdown(
        "[Energy Analysis](#energy-analysis)"
    )

    st.markdown(
        "[Monitoring Data](#monitoring-data)"
    )


# ============================================================
# PAGE HEADER
# ============================================================

st.markdown(
    '<div id="overview"></div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="main-title">'
    'Data Center Energy Efficiency Monitor'
    '</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="subtitle">'
    'Real-time energy monitoring and machine learning based PUE prediction'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# TOP STATUS
# ============================================================

status_col1, status_col2, status_col3 = st.columns(3)


with status_col1:

    st.metric(
        "Monitoring Status",
        "Active"
    )


with status_col2:

    st.metric(
        "ML Model",
        "Random Forest"
    )


with status_col3:

    st.metric(
        "Database",
        "Supabase"
    )


st.divider()


# ============================================================
# LIVE MONITORING
# ============================================================

st.markdown(
    '<div id="live-monitoring"></div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="section-title">Live Monitoring</div>',
    unsafe_allow_html=True
)


live_col1, live_col2, live_col3, live_col4 = st.columns(4)


with live_col1:

    st.metric(
        "Temperature",
        f"{current_temperature:.2f} °C"
    )


with live_col2:

    st.metric(
        "Humidity",
        f"{current_humidity:.2f} %"
    )


with live_col3:

    st.metric(
        "IT Load",
        f"{current_it_load:.2f} kW"
    )


with live_col4:

    st.metric(
        "Cooling Power",
        f"{current_cooling_power:.2f} kW"
    )


power_col1, power_col2, power_col3, power_col4, power_col5 = st.columns(5)


with power_col1:

    st.metric(
        "Total Power",
        f"{current_total_power:.2f} kW"
    )


with power_col2:

    st.metric(
        "Actual PUE",
        f"{current_pue:.3f}"
    )


with power_col3:

    st.metric(
        "Predicted PUE",
        f"{current_predicted_pue:.3f}"
    )


with power_col4:

    st.metric(
        "Last Reading",
        latest["Timestamp"].strftime("%H:%M:%S")
    )


with power_col5:

    st.metric(
        "Prediction Error",
        f"{prediction_error:.3f}"
    )


# ============================================================
# PUE STATUS
# ============================================================

if current_pue <= 1.30:

    pue_status = "Good"

    pue_message = (
        "Current energy efficiency is good."
    )

elif current_pue <= 1.50:

    pue_status = "Moderate"

    pue_message = (
        "Energy efficiency should be monitored."
    )

else:

    pue_status = "Poor"

    pue_message = (
        "Energy efficiency requires attention."
    )


if pue_status == "Good":

    st.success(
        f"PUE Status: {pue_status}. {pue_message}"
    )

elif pue_status == "Moderate":

    st.warning(
        f"PUE Status: {pue_status}. {pue_message}"
    )

else:

    st.error(
        f"PUE Status: {pue_status}. {pue_message}"
    )


# ============================================================
# PUE TREND
# ============================================================

st.markdown(
    '<div class="section-title">PUE Trend</div>',
    unsafe_allow_html=True
)


pue_chart_data = (
    monitoring_df[
        [
            "Timestamp",
            "PUE",
            "Predicted_PUE"
        ]
    ]
    .dropna(
        subset=[
            "PUE",
            "Predicted_PUE"
        ]
    )
    .set_index("Timestamp")
)


pue_chart_data = pue_chart_data.rename(
    columns={
        "PUE": "Actual PUE",
        "Predicted_PUE": "Predicted PUE"
    }
)


st.line_chart(
    pue_chart_data,
    use_container_width=True
)


# ============================================================
# PUE FORECAST
# ============================================================

st.markdown(
    '<div class="section-title">PUE Forecast</div>',
    unsafe_allow_html=True
)


forecast_chart_data = future_forecast[
    ["Forecasted_PUE"]
].copy()


forecast_chart_data.index = [
    f"Future {i}"
    for i in range(
        1,
        len(forecast_chart_data) + 1
    )
]


st.line_chart(
    forecast_chart_data,
    use_container_width=True
)


st.caption(
    "Forecast based on historical PUE trends using Linear Regression."
)


st.dataframe(
    future_forecast[
        ["Forecasted_PUE"]
    ],
    use_container_width=True
)


# ============================================================
# PUE PREDICTION
# ============================================================

st.markdown(
    '<div id="pue-prediction"></div>',
    unsafe_allow_html=True
)


st.divider()


st.markdown(
    '<div class="section-title">PUE Prediction</div>',
    unsafe_allow_html=True
)


st.write(
    "Enter operating parameters to generate a PUE prediction "
    "using the trained Random Forest model."
)


input_col1, input_col2 = st.columns(2)


with input_col1:

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=18.0,
        max_value=35.0,
        value=25.0,
        step=0.1
    )


    humidity = st.number_input(
        "Humidity (%)",
        min_value=20.0,
        max_value=80.0,
        value=50.0,
        step=0.1
    )


    it_load = st.number_input(
        "IT Load (kW)",
        min_value=50.0,
        max_value=500.0,
        value=150.0,
        step=1.0
    )


with input_col2:

    cooling_power = st.number_input(
        "Cooling Power (kW)",
        min_value=5.0,
        max_value=150.0,
        value=35.0,
        step=0.1
    )


    total_power = st.number_input(
        "Total Power (kW)",
        min_value=50.0,
        max_value=600.0,
        value=195.0,
        step=1.0
    )


st.caption(
    "Total Power is displayed for context and is not used "
    "as an input to the Random Forest model."
)


if st.button(
    "Predict PUE",
    use_container_width=True
):

    input_data = pd.DataFrame(
        [
            {
                "Temperature_C": temperature,
                "Humidity_Percent": humidity,
                "IT_Load_kW": it_load,
                "Cooling_Power_kW": cooling_power
            }
        ]
    )


    prediction = float(
        model.predict(input_data)[0]
    )


    st.subheader(
        "Prediction Result"
    )


    result_col1, result_col2 = st.columns(2)


    with result_col1:

        st.metric(
            "Predicted PUE",
            f"{prediction:.3f}"
        )


    with result_col2:

        if prediction <= 1.30:

            st.success(
                "Good efficiency"
            )

        elif prediction <= 1.50:

            st.warning(
                "Moderate efficiency"
            )

        else:

            st.error(
                "Poor efficiency"
            )


    if prediction <= 1.30:

        recommendation = (
            "The predicted operating condition is energy efficient."
        )

    elif prediction <= 1.50:

        recommendation = (
            "Consider optimizing cooling and IT load."
        )

    else:

        recommendation = (
            "Investigate cooling and power consumption "
            "to reduce energy wastage."
        )


    st.info(
        recommendation
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    '<div id="model-performance"></div>',
    unsafe_allow_html=True
)


st.divider()


st.markdown(
    '<div class="section-title">'
    'Machine Learning Model Performance'
    '</div>',
    unsafe_allow_html=True
)


model_col1, model_col2, model_col3, model_col4 = st.columns(4)


with model_col1:

    st.metric(
        "Model",
        "Random Forest"
    )


with model_col2:

    st.metric(
        "R² Score",
        f"{model_r2:.4f}"
    )


with model_col3:

    st.metric(
        "MAE",
        f"{model_mae:.4f}"
    )


with model_col4:

    st.metric(
        "RMSE",
        f"{model_rmse:.4f}"
    )


comparison = pd.DataFrame(
    {
        "Model": [
            "Linear Regression",
            "Decision Tree",
            "Random Forest"
        ],
        "MAE": [
            0.0242,
            0.0250,
            0.0197
        ],
        "RMSE": [
            0.0310,
            0.0321,
            0.0253
        ],
        "R²": [
            0.8587,
            0.8484,
            0.9061
        ]
    }
)


st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.markdown(
    '<div class="section-title">Feature Importance</div>',
    unsafe_allow_html=True
)


importance_df = pd.DataFrame(
    {
        "Feature": model.feature_names_in_,
        "Importance": model.feature_importances_
    }
).sort_values(
    "Importance",
    ascending=True
)


fig, ax = plt.subplots(
    figsize=(8, 4)
)


ax.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)


ax.set_xlabel(
    "Importance"
)


ax.set_ylabel(
    "Feature"
)


ax.set_title(
    "Random Forest Feature Importance"
)


plt.tight_layout()


st.pyplot(
    fig,
    use_container_width=True
)


plt.close(fig)


# ============================================================
# ENERGY ANALYSIS
# ============================================================

st.markdown(
    '<div id="energy-analysis"></div>',
    unsafe_allow_html=True
)


st.divider()


st.markdown(
    '<div class="section-title">Energy Analysis</div>',
    unsafe_allow_html=True
)


energy_col1, energy_col2, energy_col3, energy_col4 = st.columns(4)


with energy_col1:

    st.metric(
        "Average PUE",
        f"{monitoring_df['PUE'].mean():.3f}"
    )


with energy_col2:

    st.metric(
        "Best PUE",
        f"{monitoring_df['PUE'].min():.3f}"
    )


with energy_col3:

    st.metric(
        "Highest PUE",
        f"{monitoring_df['PUE'].max():.3f}"
    )


with energy_col4:

    st.metric(
        "Average IT Load",
        f"{monitoring_df['IT_Load_kW'].mean():.2f} kW"
    )


# ============================================================
# POWER TREND
# ============================================================

st.subheader(
    "Power Consumption Trend"
)


power_data = (
    monitoring_df
    .set_index("Timestamp")
    [
        [
            "IT_Load_kW",
            "Cooling_Power_kW",
            "Total_Power_kW"
        ]
    ]
)


power_data = power_data.rename(
    columns={
        "IT_Load_kW": "IT Load",
        "Cooling_Power_kW": "Cooling Power",
        "Total_Power_kW": "Total Power"
    }
)


st.line_chart(
    power_data,
    use_container_width=True
)


# ============================================================
# PUE VS IT LOAD
# ============================================================

st.subheader(
    "PUE vs IT Load"
)


st.scatter_chart(
    monitoring_df[
        [
            "IT_Load_kW",
            "PUE"
        ]
    ],
    x="IT_Load_kW",
    y="PUE",
    use_container_width=True
)


# ============================================================
# COOLING POWER VS PUE
# ============================================================

st.subheader(
    "Cooling Power vs PUE"
)


st.scatter_chart(
    monitoring_df[
        [
            "Cooling_Power_kW",
            "PUE"
        ]
    ],
    x="Cooling_Power_kW",
    y="PUE",
    use_container_width=True
)


# ============================================================
# TEMPERATURE VS PUE
# ============================================================

st.subheader(
    "Temperature vs PUE"
)


st.scatter_chart(
    monitoring_df[
        [
            "Temperature_C",
            "PUE"
        ]
    ],
    x="Temperature_C",
    y="PUE",
    use_container_width=True
)


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

st.subheader(
    "Correlation with PUE"
)


correlation = (
    monitoring_df[
        [
            "Temperature_C",
            "Humidity_Percent",
            "IT_Load_kW",
            "Cooling_Power_kW",
            "Total_Power_kW",
            "PUE"
        ]
    ]
    .corr()["PUE"]
    .drop("PUE")
)


correlation_df = (
    correlation
    .reset_index()
)


correlation_df.columns = [
    "Parameter",
    "Correlation"
]


st.bar_chart(
    correlation_df.set_index(
        "Parameter"
    ),
    use_container_width=True
)


# ============================================================
# EFFICIENCY DISTRIBUTION
# ============================================================

st.subheader(
    "PUE Efficiency Distribution"
)


good_count = (
    monitoring_df["PUE"] <= 1.30
).sum()


moderate_count = (
    (
        monitoring_df["PUE"] > 1.30
    )
    &
    (
        monitoring_df["PUE"] <= 1.50
    )
).sum()


poor_count = (
    monitoring_df["PUE"] > 1.50
).sum()


status_df = pd.DataFrame(
    {
        "Status": [
            "Good",
            "Moderate",
            "Poor"
        ],
        "Readings": [
            good_count,
            moderate_count,
            poor_count
        ]
    }
)


st.bar_chart(
    status_df.set_index(
        "Status"
    ),
    use_container_width=True
)


# ============================================================
# SYSTEM HEALTH
# ============================================================

st.divider()


st.markdown(
    '<div class="section-title">System Health</div>',
    unsafe_allow_html=True
)


health_score = 100


if current_pue > 1.50:

    health_score -= 30

elif current_pue > 1.30:

    health_score -= 15


if current_temperature > 28:

    health_score -= 30

elif current_temperature > 26:

    health_score -= 15


if current_total_power > 280:

    health_score -= 30

elif current_total_power > 250:

    health_score -= 15


health_score = max(
    health_score,
    0
)


health_col1, health_col2 = st.columns(2)


with health_col1:

    st.metric(
        "System Health Score",
        f"{health_score}/100"
    )


with health_col2:

    if health_score >= 80:

        st.success(
            "System condition is healthy."
        )

    elif health_score >= 50:

        st.warning(
            "System requires monitoring."
        )

    else:

        st.error(
            "System requires attention."
        )


st.progress(
    health_score / 100
)


# ============================================================
# ANOMALY DETECTION
# ============================================================

st.subheader(
    "Anomaly Detection"
)


if ml_anomaly_status == "Anomaly Detected":

    st.error(
        "⚠️ ML Anomaly Detected"
    )

    st.write(
        "Isolation Forest detected an unusual operating pattern."
    )

elif anomaly_status == "Anomaly Detected":

    st.warning(
        "⚠️ Rule-Based Anomaly Detected"
    )

    st.write(
        "Detected conditions:",
        ", ".join(anomaly_reasons)
    )

else:

    st.success(
        "✅ System Normal"
    )

    st.write(
        "No unusual operating conditions detected."
    )


st.caption(
    f"Isolation Forest Anomaly Score: {ml_anomaly_score:.4f}"
)


# ============================================================
# ALERTS
# ============================================================

st.subheader(
    "Operational Alerts"
)


alert_col1, alert_col2, alert_col3, alert_col4 = st.columns(4)


with alert_col1:

    if current_pue > 1.50:

        st.error(
            f"High PUE: {current_pue:.3f}"
        )

    elif current_pue > 1.30:

        st.warning(
            f"Moderate PUE: {current_pue:.3f}"
        )

    else:

        st.success(
            f"PUE Normal: {current_pue:.3f}"
        )


with alert_col2:

    if current_temperature > 28:

        st.error(
            f"High Temperature: {current_temperature:.2f} °C"
        )

    elif current_temperature > 26:

        st.warning(
            f"Temperature Warning: {current_temperature:.2f} °C"
        )

    else:

        st.success(
            f"Temperature Normal: {current_temperature:.2f} °C"
        )


with alert_col3:

    if current_total_power > 280:

        st.error(
            f"High Power: {current_total_power:.2f} kW"
        )

    elif current_total_power > 250:

        st.warning(
            f"Power Warning: {current_total_power:.2f} kW"
        )

    else:

        st.success(
            f"Power Normal: {current_total_power:.2f} kW"
        )


with alert_col4:

    if prediction_error > 0.10:

        st.error(
            f"High Prediction Error: {prediction_error:.3f}"
        )

    elif prediction_error > 0.05:

        st.warning(
            f"Prediction Error Warning: {prediction_error:.3f}"
        )

    else:

        st.success(
            f"Prediction Error Normal: {prediction_error:.3f}"
        )


# ============================================================
# MONITORING DATA
# ============================================================

st.markdown(
    '<div id="monitoring-data"></div>',
    unsafe_allow_html=True
)


st.divider()


st.markdown(
    '<div class="section-title">Monitoring Data</div>',
    unsafe_allow_html=True
)


display_df = monitoring_df.copy()


display_df["Timestamp"] = (
    display_df["Timestamp"]
    .dt.strftime("%Y-%m-%d %H:%M:%S")
)


st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DOWNLOAD DATA
# ============================================================

csv_data = monitoring_df.to_csv(
    index=False
)


st.download_button(
    label="Download Monitoring Data",
    data=csv_data,
    file_name="data_center_monitoring_data.csv",
    mime="text/csv",
    use_container_width=True
)


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()


st.markdown(
    '<div class="section-title">Project Information</div>',
    unsafe_allow_html=True
)


st.write(
    "This system monitors data center energy consumption "
    "and predicts Power Usage Effectiveness using Machine Learning."
)


st.write(
    "The Random Forest model uses Temperature, Humidity, "
    "IT Load, and Cooling Power as prediction features."
)


st.write(
    "Total Power is monitored for energy analysis but is "
    "not used as an input feature for the Random Forest model."
)


st.write(
    "Telemetry is transmitted through MQTT and stored "
    "persistently in Supabase PostgreSQL."
)


# ============================================================
# LAST UPDATED
# ============================================================

st.divider()


st.caption(
    "Last monitoring reading: "
    + latest["Timestamp"].strftime(
        "%Y-%m-%d %H:%M:%S"
    )
)


st.caption(
    "Automatic dashboard refresh interval: 30 seconds"
)


st.caption(
    "Data Center Energy Efficiency Monitoring and "
    "PUE Prediction Using Machine Learning"
)
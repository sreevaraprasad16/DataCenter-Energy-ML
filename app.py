import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from streamlit_autorefresh import st_autorefresh


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Data Center Energy Efficiency Monitor",
    page_icon="⚡",
    layout="wide"
)


# =========================================================
# AUTOMATIC REFRESH
# =========================================================

st_autorefresh(
    interval=30000,
    key="data_center_refresh"
)


# =========================================================
# PREDICTION HISTORY SESSION STATE
# =========================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# =========================================================
# GENERATE LIVE MONITORING READING
# =========================================================

def generate_live_reading():

    temperature = np.random.uniform(
        20,
        28
    )

    humidity = np.random.uniform(
        40,
        60
    )

    it_load = np.random.uniform(
        100,
        220
    )

    cooling_power = (
        15
        + (temperature - 18) * 1.5
        + it_load * 0.08
        + np.random.normal(0, 2)
    )

    total_power = (
        it_load
        + cooling_power
        + np.random.uniform(5, 15)
    )

    pue = total_power / it_load

    return {
        "Timestamp": pd.Timestamp.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "Temperature_C": round(
            temperature,
            2
        ),

        "Humidity_Percent": round(
            humidity,
            2
        ),

        "IT_Load_kW": round(
            it_load,
            2
        ),

        "Cooling_Power_kW": round(
            cooling_power,
            2
        ),

        "Total_Power_kW": round(
            total_power,
            2
        ),

        "PUE": round(
            pue,
            3
        )
    }


# =========================================================
# LOAD CLEAN MACHINE LEARNING MODEL
# =========================================================

model = joblib.load(
    "random_forest_clean_model.pkl"
)


# =========================================================
# CLEAN MODEL PERFORMANCE
# =========================================================

model_r2 = 0.9061
model_performance = model_r2 * 100

model_mae = 0.0197
model_rmse = 0.0253


# =========================================================
# LOAD MONITORING DATA
# =========================================================

monitoring_df = pd.read_csv(
    "data/monitoring_data.csv"
)


# =========================================================
# GENERATE NEW LIVE READING
# =========================================================

new_reading = generate_live_reading()


# =========================================================
# ADD NEW READING
# =========================================================

monitoring_df = pd.concat(
    [
        monitoring_df,
        pd.DataFrame([new_reading])
    ],
    ignore_index=True
)


# =========================================================
# KEEP LATEST 100 READINGS
# =========================================================

monitoring_df = monitoring_df.tail(
    100
).reset_index(
    drop=True
)


# =========================================================
# SAVE UPDATED MONITORING DATA
# =========================================================

monitoring_df.to_csv(
    "data/monitoring_data.csv",
    index=False
)


# =========================================================
# LATEST READING
# =========================================================

latest = monitoring_df.iloc[-1]


# =========================================================
# CURRENT VALUES
# =========================================================

current_pue = float(
    latest["PUE"]
)

current_temperature = float(
    latest["Temperature_C"]
)

current_total_power = float(
    latest["Total_Power_kW"]
)


# =========================================================
# PUE EFFICIENCY STATUS
# =========================================================

if current_pue <= 1.30:

    pue_status = "Good Efficiency"

elif current_pue <= 1.50:

    pue_status = "Moderate Efficiency"

else:

    pue_status = "Poor Efficiency"


# =========================================================
# ENERGY EFFICIENCY RECOMMENDATION
# =========================================================

if current_pue <= 1.30:

    recommendation = (
        "Maintain current operating conditions. "
        "The data center is operating efficiently."
    )

elif current_pue <= 1.50:

    recommendation = (
        "Consider optimizing cooling systems and IT load "
        "to improve energy efficiency."
    )

else:

    recommendation = (
        "Investigate cooling and power consumption "
        "immediately to reduce energy wastage."
    )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "⚡ Data Center Monitor"
)

st.sidebar.write(
    "### Project"
)

st.sidebar.write(
    "Data Center Energy Efficiency Monitoring and "
    "PUE Prediction Using Machine Learning"
)

st.sidebar.divider()


# =========================================================
# SIDEBAR ML INFORMATION
# =========================================================

st.sidebar.write(
    "### ML Model"
)

st.sidebar.write(
    "Random Forest Regressor"
)

st.sidebar.write(
    "### ML Input Features"
)

st.sidebar.write(
    "🌡️ Temperature"
)

st.sidebar.write(
    "💧 Humidity"
)

st.sidebar.write(
    "💻 IT Load"
)

st.sidebar.write(
    "❄️ Cooling Power"
)

st.sidebar.write(
    "### Target"
)

st.sidebar.write(
    "📊 PUE Prediction"
)


# =========================================================
# DASHBOARD NAVIGATION
# =========================================================

st.sidebar.divider()

st.sidebar.subheader(
    "🧭 Dashboard Navigation"
)

st.sidebar.markdown(
    '<a href="#quick-overview">📊 Quick Overview</a>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<a href="#key-performance-indicators">📈 Key Performance Indicators</a>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<a href="#machine-learning-model">🤖 Machine Learning Model</a>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<a href="#pue-prediction">🎯 PUE Prediction</a>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<a href="#prediction-history">📋 Prediction History</a>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<a href="#feature-importance">🌲 Feature Importance</a>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<a href="#model-comparison">📈 Model Comparison</a>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<a href="#real-time-monitoring">📡 Real-Time Monitoring</a>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<a href="#energy-analysis">⚡ Energy Analysis</a>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<a href="#correlation-analysis">📊 Correlation Analysis</a>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<a href="#monitoring-data">🗃️ Monitoring Data</a>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<a href="#project-information">📘 Project Information</a>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<a href="#system-health">💚 System Health</a>',
    unsafe_allow_html=True
)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div id="quick-overview"></div>',
    unsafe_allow_html=True
)

st.title(
    "⚡ Data Center Energy Efficiency Monitor"
)

st.write(
    "Machine Learning based PUE prediction and "
    "energy efficiency monitoring"
)

st.divider()


# =========================================================
# QUICK OVERVIEW
# =========================================================

st.subheader(
    "📊 Quick Overview"
)

overview_col1, overview_col2, overview_col3 = st.columns(3)


with overview_col1:

    st.metric(
        "🤖 ML Model",
        "Random Forest"
    )


with overview_col2:

    st.metric(
        "🎯 Target",
        "PUE Prediction"
    )


with overview_col3:

    st.metric(
        "📡 Monitoring",
        "Active"
    )


# =========================================================
# KEY PERFORMANCE INDICATORS
# =========================================================

st.markdown(
    '<div id="key-performance-indicators"></div>',
    unsafe_allow_html=True
)

st.subheader(
    "📊 Key Performance Indicators"
)

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)


with kpi_col1:

    st.metric(
        "📊 Average PUE",
        f"{monitoring_df['PUE'].mean():.3f}"
    )


with kpi_col2:

    st.metric(
        "⚡ Average Total Power",
        f"{monitoring_df['Total_Power_kW'].mean():.2f} kW"
    )


with kpi_col3:

    st.metric(
        "❄️ Average Cooling Power",
        f"{monitoring_df['Cooling_Power_kW'].mean():.2f} kW"
    )


with kpi_col4:

    st.metric(
        "💻 Average IT Load",
        f"{monitoring_df['IT_Load_kW'].mean():.2f} kW"
    )


# =========================================================
# MACHINE LEARNING MODEL
# =========================================================

st.markdown(
    '<div id="machine-learning-model"></div>',
    unsafe_allow_html=True
)

st.subheader(
    "🏆 Final Machine Learning Model"
)

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Selected Model",
        "Random Forest"
    )


with col2:

    st.metric(
        "R² Score",
        f"{model_r2:.4f}",
        f"{model_performance:.2f}% R²"
    )


with col3:

    st.metric(
        "MAE",
        f"{model_mae:.4f}"
    )


with col4:

    st.metric(
        "RMSE",
        f"{model_rmse:.4f}"
    )


st.success(
    "Final Model: Random Forest Regressor"
)

st.info(
    "Random Forest was selected because it achieved "
    "the highest R² score and the lowest MAE among "
    "the three evaluated models."
)

st.caption(
    "Note: The 90.61% value represents the R² score "
    "expressed as a percentage. It is not prediction "
    "confidence or probability."
)

st.info(
    "ML input features: Temperature, Humidity, IT Load, "
    "and Cooling Power. Total Power is retained for "
    "monitoring and analysis but is NOT used as an ML input."
)


# =========================================================
# PUE PREDICTION
# =========================================================

st.markdown(
    '<div id="pue-prediction"></div>',
    unsafe_allow_html=True
)

st.divider()

st.subheader(
    "🔢 Enter Data Center Parameters"
)

col1, col2 = st.columns(2)


with col1:

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=18.0,
        max_value=30.0,
        value=25.0,
        step=0.1
    )

    humidity = st.number_input(
        "Humidity (%)",
        min_value=35.0,
        max_value=65.0,
        value=50.0,
        step=0.1
    )

    it_load = st.number_input(
        "IT Load (kW)",
        min_value=80.0,
        max_value=250.0,
        value=150.0,
        step=1.0
    )


with col2:

    total_power = st.number_input(
        "Total Power (kW)",
        min_value=100.0,
        max_value=320.0,
        value=195.0,
        step=1.0
    )

    cooling_power = st.number_input(
        "Cooling Power (kW)",
        min_value=15.0,
        max_value=55.0,
        value=35.0,
        step=0.1
    )


st.caption(
    "Total Power is displayed for monitoring/context only. "
    "It is not passed to the Random Forest model."
)

st.divider()


# =========================================================
# PREDICTION
# =========================================================

if st.button(
    "🚀 Predict PUE",
    use_container_width=True
):

    # -----------------------------------------------------
    # CLEAN ML INPUT
    # -----------------------------------------------------
    # Total_Power_kW is intentionally NOT included.

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


    # -----------------------------------------------------
    # PREDICT PUE
    # -----------------------------------------------------

    prediction = model.predict(
        input_data
    )[0]


    # -----------------------------------------------------
    # PREDICTED PUE STATUS
    # -----------------------------------------------------

    if prediction <= 1.30:

        prediction_status = (
            "Good Predicted Efficiency"
        )

        prediction_recommendation = (
            "The predicted operating condition is "
            "energy efficient."
        )

    elif prediction <= 1.50:

        prediction_status = (
            "Moderate Predicted Efficiency"
        )

        prediction_recommendation = (
            "Consider optimizing cooling systems "
            "and IT load."
        )

    else:

        prediction_status = (
            "Poor Predicted Efficiency"
        )

        prediction_recommendation = (
            "Investigate power and cooling consumption "
            "to reduce energy wastage."
        )


    # =====================================================
    # PREDICTION RESULT
    # =====================================================

    st.subheader(
        "🎯 Prediction Result"
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
                "🟢 Data Center Efficiency: GOOD"
            )

        elif prediction <= 1.50:

            st.warning(
                "🟡 Data Center Efficiency: MODERATE"
            )

        else:

            st.error(
                "🔴 Data Center Efficiency: POOR"
            )


    # =====================================================
    # PREDICTION STATUS
    # =====================================================

    st.subheader(
        "📊 Predicted Efficiency Status"
    )


    if prediction <= 1.30:

        st.success(
            f"🟢 {prediction_status}"
        )

    elif prediction <= 1.50:

        st.warning(
            f"🟡 {prediction_status}"
        )

    else:

        st.error(
            f"🔴 {prediction_status}"
        )


    # =====================================================
    # PREDICTION RECOMMENDATION
    # =====================================================

    st.subheader(
        "💡 Prediction Recommendation"
    )

    st.info(
        prediction_recommendation
    )


    # =====================================================
    # SAVE PREDICTION TO HISTORY
    # =====================================================

    st.session_state.prediction_history.append(
        {
            "Temperature (°C)": temperature,
            "Humidity (%)": humidity,
            "IT Load (kW)": it_load,
            "Total Power (kW)": total_power,
            "Cooling Power (kW)": cooling_power,
            "Predicted PUE": round(
                prediction,
                3
            )
        }
    )


    # =====================================================
    # ACTUAL VS PREDICTED PUE
    # =====================================================

    st.subheader(
        "📈 Actual PUE vs Predicted PUE"
    )

    actual_values = monitoring_df.tail(
        10
    ).copy()


    actual_values["Predicted PUE"] = model.predict(
        actual_values[
            [
                "Temperature_C",
                "Humidity_Percent",
                "IT_Load_kW",
                "Cooling_Power_kW"
            ]
        ]
    )


    actual_predicted_df = actual_values[
        [
            "Timestamp",
            "PUE",
            "Predicted PUE"
        ]
    ].copy()


    actual_predicted_df = actual_predicted_df.rename(
        columns={
            "PUE": "Actual PUE"
        }
    )


    st.line_chart(
        actual_predicted_df.set_index(
            "Timestamp"
        )
    )


    # =====================================================
    # INPUT SUMMARY
    # =====================================================

    st.subheader(
        "📋 Input Parameters"
    )

    summary = pd.DataFrame(
        {
            "Parameter": [
                "Temperature",
                "Humidity",
                "IT Load",
                "Total Power",
                "Cooling Power"
            ],

            "Value": [
                f"{temperature} °C",
                f"{humidity} %",
                f"{it_load} kW",
                f"{total_power} kW",
                f"{cooling_power} kW"
            ]
        }
    )

    st.table(
        summary
    )


# =========================================================
# PREDICTION HISTORY
# =========================================================

st.markdown(
    '<div id="prediction-history"></div>',
    unsafe_allow_html=True
)

st.divider()

st.subheader(
    "📋 Prediction History"
)


if st.session_state.prediction_history:

    prediction_history = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(
        prediction_history,
        use_container_width=True,
        hide_index=True
    )


    if st.button(
        "🗑️ Clear Prediction History",
        use_container_width=True
    ):

        st.session_state.prediction_history = []

        st.rerun()


else:

    st.info(
        "No predictions yet. Click 🚀 Predict PUE "
        "to create a prediction."
    )


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

st.markdown(
    '<div id="feature-importance"></div>',
    unsafe_allow_html=True
)

st.divider()

st.subheader(
    "🌲 Random Forest Feature Importance"
)


# IMPORTANT:
# Exactly 4 features because the clean model
# was trained using exactly 4 features.

features = [
    "Temperature_C",
    "Humidity_Percent",
    "IT_Load_kW",
    "Cooling_Power_kW"
]


importance = model.feature_importances_


importance_df = pd.DataFrame(
    {
        "Feature": features,
        "Importance": importance
    }
)


importance_df = importance_df.sort_values(
    "Importance",
    ascending=True
)


fig, ax = plt.subplots()


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
    fig
)

plt.close(fig)


st.caption(
    "Higher importance indicates that the feature contributed "
    "more to the Random Forest model's predictions on this dataset."
)


# =========================================================
# MODEL COMPARISON
# =========================================================

st.markdown(
    '<div id="model-comparison"></div>',
    unsafe_allow_html=True
)

st.divider()

st.subheader(
    "📈 Model Performance Comparison"
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


st.success(
    "Final Model: Random Forest Regressor"
)


# =========================================================
# R² SCORE COMPARISON
# =========================================================

st.subheader(
    "📊 R² Score Comparison"
)


r2_chart = comparison[
    [
        "Model",
        "R²"
    ]
].set_index(
    "Model"
)


st.bar_chart(
    r2_chart
)


st.info(
    "Random Forest achieved the highest R² score of 0.9061 "
    "among the evaluated models. This indicates that it "
    "provided the best overall predictive performance on "
    "the dataset used in this project."
)


# =========================================================
# REAL-TIME MONITORING
# =========================================================

st.markdown(
    '<div id="real-time-monitoring"></div>',
    unsafe_allow_html=True
)

st.divider()

st.subheader(
    "📡 Data Center Real-Time Monitoring"
)


st.write(
    "Simulated sensor readings from the data center. "
    "The dashboard automatically refreshes every 30 seconds."
)


# =========================================================
# ENERGY EFFICIENCY SUMMARY
# =========================================================

st.subheader(
    "📊 Energy Efficiency Summary"
)


average_pue = monitoring_df[
    "PUE"
].mean()


best_pue = monitoring_df[
    "PUE"
].min()


highest_pue = monitoring_df[
    "PUE"
].max()


average_it_load = monitoring_df[
    "IT_Load_kW"
].mean()


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "📊 Average PUE",
        f"{average_pue:.3f}"
    )


with col2:

    st.metric(
        "🟢 Best PUE",
        f"{best_pue:.3f}"
    )


with col3:

    st.metric(
        "🔴 Highest PUE",
        f"{highest_pue:.3f}"
    )


with col4:

    st.metric(
        "💻 Average IT Load",
        f"{average_it_load:.2f} kW"
    )


# =========================================================
# PUE EFFICIENCY DISTRIBUTION
# =========================================================

st.subheader(
    "📊 PUE Efficiency Distribution"
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
    )
)


# =========================================================
# LATEST MONITORING READINGS
# =========================================================

st.subheader(
    "📡 Latest Monitoring Readings"
)


latest = monitoring_df.iloc[-1]


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "🌡️ Temperature",
        f"{latest['Temperature_C']:.2f} °C"
    )


with col2:

    st.metric(
        "💻 IT Load",
        f"{latest['IT_Load_kW']:.2f} kW"
    )


with col3:

    st.metric(
        "❄️ Cooling Power",
        f"{latest['Cooling_Power_kW']:.2f} kW"
    )


with col4:

    st.metric(
        "📊 PUE",
        f"{latest['PUE']:.3f}"
    )


# =========================================================
# CURRENT EFFICIENCY STATUS
# =========================================================

st.subheader(
    "🚦 Current Efficiency Status"
)


st.info(
    f"Current PUE: {current_pue:.3f} — {pue_status}"
)


if current_pue <= 1.30:

    st.success(
        f"🟢 GOOD — Current PUE is {current_pue:.3f}"
    )

elif current_pue <= 1.50:

    st.warning(
        f"🟡 MODERATE — Current PUE is {current_pue:.3f}"
    )

else:

    st.error(
        f"🔴 POOR — Current PUE is {current_pue:.3f}"
    )


# =========================================================
# ENERGY EFFICIENCY RECOMMENDATION
# =========================================================

st.subheader(
    "💡 Energy Efficiency Recommendation"
)


st.info(
    recommendation
)


# =========================================================
# ENERGY EFFICIENCY ALERT
# =========================================================

st.subheader(
    "🚨 Energy Efficiency Alert"
)


if current_pue > 1.50:

    st.error(
        f"🔴 ALERT: PUE is high ({current_pue:.3f}). "
        "Energy efficiency requires immediate attention."
    )

elif current_pue > 1.30:

    st.warning(
        f"🟡 WARNING: PUE is {current_pue:.3f}. "
        "Consider checking cooling and power consumption."
    )

else:

    st.success(
        f"🟢 NORMAL: PUE is {current_pue:.3f}. "
        "Energy efficiency is currently good."
    )


# =========================================================
# TEMPERATURE ALERT
# =========================================================

st.subheader(
    "🌡️ Temperature Alert"
)


if current_temperature > 28:

    st.error(
        f"🔴 HIGH TEMPERATURE: "
        f"{current_temperature:.2f} °C. "
        "Cooling system requires immediate attention."
    )

elif current_temperature > 26:

    st.warning(
        f"🟡 TEMPERATURE WARNING: "
        f"{current_temperature:.2f} °C. "
        "Monitor the cooling system."
    )

else:

    st.success(
        f"🟢 NORMAL: Temperature is "
        f"{current_temperature:.2f} °C."
    )


# =========================================================
# TOTAL POWER ALERT
# =========================================================

st.subheader(
    "⚡ Total Power Alert"
)


if current_total_power > 280:

    st.error(
        f"🔴 CRITICAL POWER: "
        f"{current_total_power:.2f} kW. "
        "Total power consumption is very high."
    )

elif current_total_power > 250:

    st.warning(
        f"🟡 POWER WARNING: "
        f"{current_total_power:.2f} kW. "
        "Monitor energy consumption."
    )

else:

    st.success(
        f"🟢 NORMAL: Total power is "
        f"{current_total_power:.2f} kW."
    )


# =========================================================
# ENERGY CONSUMPTION ANALYSIS
# =========================================================

st.markdown(
    '<div id="energy-analysis"></div>',
    unsafe_allow_html=True
)

st.divider()

st.subheader(
    "⚡ Energy Consumption Analysis"
)


average_total_power = (
    monitoring_df[
        "Total_Power_kW"
    ].mean()
)


average_cooling_power = (
    monitoring_df[
        "Cooling_Power_kW"
    ].mean()
)


col1, col2 = st.columns(2)


with col1:

    st.metric(
        "⚡ Average Total Power",
        f"{average_total_power:.2f} kW"
    )


with col2:

    st.metric(
        "❄️ Average Cooling Power",
        f"{average_cooling_power:.2f} kW"
    )


# =========================================================
# POWER CONSUMPTION TREND
# =========================================================

st.subheader(
    "⚡ Power Consumption Trend"
)


power_data = monitoring_df.set_index(
    "Timestamp"
)[
    [
        "IT_Load_kW",
        "Cooling_Power_kW",
        "Total_Power_kW"
    ]
]


st.line_chart(
    power_data
)


# =========================================================
# PUE VS IT LOAD
# =========================================================

st.subheader(
    "📊 PUE vs IT Load"
)


pue_load_data = monitoring_df[
    [
        "IT_Load_kW",
        "PUE"
    ]
]


st.scatter_chart(
    pue_load_data,
    x="IT_Load_kW",
    y="PUE"
)


# =========================================================
# COOLING POWER VS PUE
# =========================================================

st.subheader(
    "❄️ Cooling Power vs PUE"
)


cooling_pue_data = monitoring_df[
    [
        "Cooling_Power_kW",
        "PUE"
    ]
]


st.scatter_chart(
    cooling_pue_data,
    x="Cooling_Power_kW",
    y="PUE"
)


# =========================================================
# TEMPERATURE VS PUE
# =========================================================

st.subheader(
    "🌡️ Temperature vs PUE"
)


temperature_pue_data = monitoring_df[
    [
        "Temperature_C",
        "PUE"
    ]
]


st.scatter_chart(
    temperature_pue_data,
    x="Temperature_C",
    y="PUE"
)


# =========================================================
# CORRELATION ANALYSIS
# =========================================================

st.markdown(
    '<div id="correlation-analysis"></div>',
    unsafe_allow_html=True
)

st.divider()

st.subheader(
    "📊 Correlation with PUE"
)


correlation = monitoring_df[
    [
        "Temperature_C",
        "Humidity_Percent",
        "IT_Load_kW",
        "Cooling_Power_kW",
        "Total_Power_kW",
        "PUE"
    ]
].corr()["PUE"].drop(
    "PUE"
)


correlation_df = correlation.reset_index()


correlation_df.columns = [
    "Parameter",
    "Correlation"
]


# =========================================================
# CORRELATION STRENGTH
# =========================================================

st.subheader(
    "📈 Correlation Strength"
)


correlation_chart = (
    correlation_df.set_index(
        "Parameter"
    )
)


st.bar_chart(
    correlation_chart
)


st.dataframe(
    correlation_df,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# PUE TREND
# =========================================================

st.subheader(
    "📈 PUE Trend"
)


st.line_chart(
    monitoring_df.set_index(
        "Timestamp"
    )["PUE"]
)


# =========================================================
# IT LOAD TREND
# =========================================================

st.subheader(
    "💻 IT Load Trend"
)


st.line_chart(
    monitoring_df.set_index(
        "Timestamp"
    )["IT_Load_kW"]
)


# =========================================================
# TEMPERATURE TREND
# =========================================================

st.subheader(
    "🌡️ Temperature Trend"
)


st.line_chart(
    monitoring_df.set_index(
        "Timestamp"
    )["Temperature_C"]
)


# =========================================================
# MONITORING DATA
# =========================================================

st.markdown(
    '<div id="monitoring-data"></div>',
    unsafe_allow_html=True
)

st.divider()

st.subheader(
    "📋 Monitoring Data"
)


st.dataframe(
    monitoring_df,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# DOWNLOAD MONITORING DATA
# =========================================================

st.subheader(
    "⬇️ Download Monitoring Data"
)


csv_data = monitoring_df.to_csv(
    index=False
)


st.download_button(
    label="📥 Download CSV",
    data=csv_data,
    file_name="data_center_monitoring_data.csv",
    mime="text/csv",
    use_container_width=True
)


# =========================================================
# DASHBOARD CONTROL
# =========================================================

st.subheader(
    "🔄 Dashboard Control"
)


if st.button(
    "🔄 Refresh Monitoring Data",
    use_container_width=True
):

    st.rerun()


# =========================================================
# PROJECT INFORMATION
# =========================================================

st.markdown(
    '<div id="project-information"></div>',
    unsafe_allow_html=True
)

st.divider()

st.subheader(
    "📘 About This Project"
)


st.write(
    "This project monitors data center energy consumption "
    "and predicts Power Usage Effectiveness (PUE) using "
    "Machine Learning."
)


st.write(
    "The Random Forest Regressor predicts PUE using "
    "temperature, humidity, IT load, and cooling power."
)


st.write(
    "Total Power is monitored and analyzed by the dashboard, "
    "but it is intentionally excluded from the ML input features."
)


st.info(
    "🎯 Main Goal: Improve data center energy efficiency "
    "by monitoring power consumption and identifying "
    "inefficient operating conditions."
)


# =========================================================
# LAST UPDATED TIME
# =========================================================

st.divider()


latest_timestamp = (
    monitoring_df[
        "Timestamp"
    ].iloc[-1]
)


st.caption(
    f"🕒 Last monitoring reading: {latest_timestamp}"
)


# =========================================================
# PUE EFFICIENCY GAUGE
# =========================================================

st.subheader(
    "🎯 Current PUE Efficiency Gauge"
)


if current_pue <= 1.30:

    st.success(
        f"🟢 GOOD EFFICIENCY — PUE: "
        f"{current_pue:.3f}"
    )

elif current_pue <= 1.50:

    st.warning(
        f"🟡 MODERATE EFFICIENCY — PUE: "
        f"{current_pue:.3f}"
    )

else:

    st.error(
        f"🔴 POOR EFFICIENCY — PUE: "
        f"{current_pue:.3f}"
    )


st.progress(
    min(
        current_pue / 2.0,
        1.0
    )
)


# =========================================================
# SYSTEM HEALTH SCORE
# =========================================================

st.markdown(
    '<div id="system-health"></div>',
    unsafe_allow_html=True
)

st.subheader(
    "💚 System Health Score"
)


health_score = 100


# ---------------------------------------------------------
# PUE PENALTY
# ---------------------------------------------------------

if current_pue > 1.50:

    health_score -= 30

elif current_pue > 1.30:

    health_score -= 15


# ---------------------------------------------------------
# TEMPERATURE PENALTY
# ---------------------------------------------------------

if current_temperature > 28:

    health_score -= 30

elif current_temperature > 26:

    health_score -= 15


# ---------------------------------------------------------
# POWER PENALTY
# ---------------------------------------------------------

if current_total_power > 280:

    health_score -= 30

elif current_total_power > 250:

    health_score -= 15


health_score = max(
    0,
    health_score
)


st.metric(
    "🏥 Overall Health Score",
    f"{health_score}/100"
)


st.progress(
    health_score / 100
)


if health_score >= 80:

    st.success(
        "🟢 Excellent system health"
    )

elif health_score >= 50:

    st.warning(
        "🟡 System needs monitoring"
    )

else:

    st.error(
        "🔴 System requires attention"
    )


# =========================================================
# MONITORING STATUS SUMMARY
# =========================================================

st.subheader(
    "📡 Monitoring Status Summary"
)


status_col1, status_col2, status_col3 = st.columns(3)


# =========================================================
# TEMPERATURE STATUS
# =========================================================

with status_col1:

    st.write(
        "🌡️ **Temperature Status**"
    )


    if current_temperature <= 26:

        st.success(
            "🟢 NORMAL"
        )

    elif current_temperature <= 28:

        st.warning(
            "🟡 WARNING"
        )

    else:

        st.error(
            "🔴 CRITICAL"
        )


# =========================================================
# POWER STATUS
# =========================================================

with status_col2:

    st.write(
        "⚡ **Power Status**"
    )


    if current_total_power <= 250:

        st.success(
            "🟢 NORMAL"
        )

    elif current_total_power <= 280:

        st.warning(
            "🟡 WARNING"
        )

    else:

        st.error(
            "🔴 CRITICAL"
        )


# =========================================================
# PUE STATUS
# =========================================================

with status_col3:

    st.write(
        "📊 **PUE Status**"
    )


    if current_pue <= 1.30:

        st.success(
            "🟢 GOOD"
        )

    elif current_pue <= 1.50:

        st.warning(
            "🟡 MODERATE"
        )

    else:

        st.error(
            "🔴 POOR"
        )


# =========================================================
# DATA CENTER HEALTH SUMMARY
# =========================================================

st.subheader(
    "🏥 Data Center Health Summary"
)


if (
    current_pue <= 1.30
    and current_temperature <= 26
    and current_total_power <= 250
):

    st.success(
        "🟢 HEALTH STATUS: EXCELLENT"
    )

    st.write(
        "The data center is operating efficiently "
        "under the current conditions."
    )


elif (
    current_pue <= 1.50
    and current_temperature <= 28
    and current_total_power <= 280
):

    st.warning(
        "🟡 HEALTH STATUS: NEEDS MONITORING"
    )

    st.write(
        "The data center is operating normally, "
        "but some parameters should be monitored."
    )


else:

    st.error(
        "🔴 HEALTH STATUS: ATTENTION REQUIRED"
    )

    st.write(
        "One or more operating parameters indicate "
        "a potentially inefficient condition."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()


st.caption(
    "⚡ Data Center Energy Efficiency Monitoring and "
    "PUE Prediction Using Machine Learning"
)
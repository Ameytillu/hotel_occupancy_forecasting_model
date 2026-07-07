import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Hotel Occupancy Forecasting",
    page_icon="🏨",
    layout="wide"
)

# --------------------------------------------------
# Paths
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.big-title {
    font-size: 42px;
    font-weight: 800;
    color: #1f4e79;
}
.sub-title {
    font-size: 18px;
    color: #555;
}
.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown('<div class="big-title">Hotel Occupancy Forecasting App</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Interactive machine learning app for predicting hotel occupancy using revenue management features.</div>',
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# Load Available Models
# --------------------------------------------------
model_files = list(MODEL_DIR.glob("*.pkl"))
model_files = [file for file in model_files if "feature" not in file.name.lower()]

if not model_files:
    st.error("No model files found in the models folder.")
    st.stop()

selected_model_file = st.sidebar.selectbox(
    "Select Model for Prediction",
    model_files,
    format_func=lambda x: x.name
)

model = joblib.load(selected_model_file)

feature_path = MODEL_DIR / "selected_feature_columns.pkl"

if feature_path.exists():
    feature_columns = joblib.load(feature_path)
else:
    st.error("selected_feature_columns.pkl not found in models folder.")
    st.stop()

st.sidebar.success(f"Loaded Model: {selected_model_file.name}")

# --------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------
page = st.sidebar.radio(
    "Choose Prediction Type",
    ["Single Prediction", "Bulk CSV Prediction", "Model Info"]
)

# --------------------------------------------------
# Helper Function
# --------------------------------------------------
def create_input_form(feature_columns):
    input_data = {}

    st.subheader("Enter Forecasting Inputs")

    col1, col2, col3 = st.columns(3)

    for i, feature in enumerate(feature_columns):
        current_col = [col1, col2, col3][i % 3]

        with current_col:
            feature_lower = feature.lower()

            if "date" in feature_lower:
                input_data[feature] = st.date_input(feature)

            elif feature_lower in ["season"]:
                input_data[feature] = st.selectbox(
                    feature,
                    ["Winter", "Spring", "Summer", "Fall"]
                )

            elif "dayofweek" in feature_lower:
                input_data[feature] = st.selectbox(
                    feature,
                    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                )

            elif (
                "flag" in feature_lower
                or "is" in feature_lower
                or "running" in feature_lower
                or "closed" in feature_lower
                or "cta" in feature_lower
                or "ctd" in feature_lower
                or "cruise" in feature_lower
                or "holiday" in feature_lower
                or "event" in feature_lower
            ):
                input_data[feature] = st.selectbox(feature, [0, 1])

            elif "adr" in feature_lower:
                input_data[feature] = st.number_input(
                    feature,
                    min_value=0.0,
                    max_value=1000.0,
                    value=220.0,
                    step=5.0
                )

            elif "occupancy" in feature_lower or "occ" in feature_lower:
                input_data[feature] = st.number_input(
                    feature,
                    min_value=0.0,
                    max_value=100.0,
                    value=70.0,
                    step=1.0
                )

            elif "rooms" in feature_lower:
                input_data[feature] = st.number_input(
                    feature,
                    min_value=0.0,
                    max_value=433.0,
                    value=100.0,
                    step=1.0
                )

            elif "pct" in feature_lower or "discount" in feature_lower or "wash" in feature_lower:
                input_data[feature] = st.number_input(
                    feature,
                    min_value=0.0,
                    max_value=100.0,
                    value=10.0,
                    step=1.0
                )

            elif "temp" in feature_lower:
                input_data[feature] = st.number_input(
                    feature,
                    min_value=20.0,
                    max_value=120.0,
                    value=80.0,
                    step=1.0
                )

            elif "month" in feature_lower:
                input_data[feature] = st.slider(feature, 1, 12, 7)

            elif "quarter" in feature_lower:
                input_data[feature] = st.slider(feature, 1, 4, 3)

            elif "week" in feature_lower:
                input_data[feature] = st.slider(feature, 1, 53, 28)

            else:
                input_data[feature] = st.number_input(
                    feature,
                    value=0.0,
                    step=1.0
                )

    return pd.DataFrame([input_data])


# --------------------------------------------------
# Single Prediction Page
# --------------------------------------------------
if page == "Single Prediction":

    st.markdown("### Single Date Occupancy Prediction")

    st.info(
        "Enter the expected values for the selected features. "
        "The model will predict the occupancy percentage for that scenario."
    )

    input_df = create_input_form(feature_columns)

    st.divider()

    if st.button("Predict Occupancy", use_container_width=True):
        prediction = model.predict(input_df)[0]
        prediction = max(0, min(100, prediction))

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Predicted Occupancy", f"{prediction:.2f}%")

        with col2:
            estimated_rooms = round((prediction / 100) * 433)
            st.metric("Estimated Rooms Sold", estimated_rooms)

        with col3:
            remaining_rooms = 433 - estimated_rooms
            st.metric("Estimated Rooms Left", remaining_rooms)

        st.success("Prediction completed successfully.")

        fig_df = pd.DataFrame({
            "Metric": ["Predicted Occupancy", "Remaining Capacity"],
            "Value": [prediction, 100 - prediction]
        })

        fig = px.pie(
            fig_df,
            names="Metric",
            values="Value",
            title="Predicted Occupancy Share"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Input Data Used for Prediction")
        st.dataframe(input_df, use_container_width=True)


# --------------------------------------------------
# Bulk CSV Prediction Page
# --------------------------------------------------
elif page == "Bulk CSV Prediction":

    st.markdown("### Bulk Occupancy Prediction using CSV")

    st.info(
        "Upload a CSV file with the same feature columns used during training. "
        "The app will generate occupancy predictions for all rows."
    )

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        bulk_df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data Preview")
        st.dataframe(bulk_df.head(), use_container_width=True)

        missing_cols = [col for col in feature_columns if col not in bulk_df.columns]

        if missing_cols:
            st.error("The uploaded CSV is missing these required columns:")
            st.write(missing_cols)
        else:
            prediction_df = bulk_df[feature_columns].copy()

            if st.button("Run Bulk Prediction", use_container_width=True):
                predictions = model.predict(prediction_df)
                predictions = np.clip(predictions, 0, 100)

                bulk_df["Predicted_Occupancy_Pct"] = predictions
                bulk_df["Estimated_Rooms_Sold"] = round((bulk_df["Predicted_Occupancy_Pct"] / 100) * 433)
                bulk_df["Estimated_Rooms_Left"] = 433 - bulk_df["Estimated_Rooms_Sold"]

                st.success("Bulk prediction completed.")

                st.subheader("Prediction Results")
                st.dataframe(bulk_df, use_container_width=True)

                fig = px.line(
                    bulk_df,
                    y="Predicted_Occupancy_Pct",
                    title="Predicted Occupancy Trend"
                )

                st.plotly_chart(fig, use_container_width=True)

                csv = bulk_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download Prediction Results",
                    data=csv,
                    file_name="occupancy_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )


# --------------------------------------------------
# Model Info Page
# --------------------------------------------------
elif page == "Model Info":

    st.markdown("### Model Information")

    st.write("Selected Model:")
    st.code(selected_model_file.name)

    st.write("Model Folder:")
    st.code(str(MODEL_DIR))

    st.write("Features Used by the Model:")
    st.dataframe(pd.DataFrame({"Feature": feature_columns}), use_container_width=True)

    metrics_path = MODEL_DIR / "model_evaluation_metrics.csv"

    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)

        st.subheader("Model Evaluation Metrics")
        st.dataframe(metrics_df, use_container_width=True)

        fig = px.bar(
            metrics_df,
            x="Model",
            y="MAE",
            title="Model Comparison by MAE"
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("model_evaluation_metrics.csv not found.")

st.divider()

st.caption(
    "Built for hotel revenue management forecasting. "
    "This app supports occupancy prediction, inventory planning, pricing decisions, and demand analysis."
)

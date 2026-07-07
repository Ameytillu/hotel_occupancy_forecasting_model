# Hotel Occupancy Forecasting using Machine Learning

## Project Overview

Forecasting hotel occupancy accurately is one of the most important responsibilities of a Revenue Management team. Occupancy forecasts influence pricing strategies, staffing decisions, inventory management, marketing campaigns, and overall hotel profitability.

The objective of this project is to build a Machine Learning model capable of predicting future hotel occupancy using historical operational data instead of relying only on traditional forecasting techniques or historical averages.

This project simulates a real-world hotel revenue management forecasting workflow by incorporating booking pace, group demand, pricing strategy, market conditions, competitor performance, seasonality, weather, holidays, and special events.

The project is designed to demonstrate both Machine Learning skills and practical Revenue Management knowledge.

---

# Business Problem

Revenue managers make pricing and inventory decisions every day based on expected demand.

An inaccurate forecast may result in:

* Selling rooms too cheaply during high-demand periods.
* Pricing rooms too aggressively during low-demand periods.
* Poor staffing decisions.
* Lost revenue opportunities.
* Inefficient inventory allocation.

Machine Learning allows us to identify complex demand patterns that are difficult to capture using traditional forecasting methods.

---

# Project Objectives

The primary objectives of this project are:

* Predict future hotel occupancy percentage.
* Analyze historical booking behavior.
* Understand which factors influence occupancy the most.
* Compare multiple Machine Learning algorithms.
* Select the best performing forecasting model.
* Save the trained model for future occupancy predictions.
* Create a reusable forecasting pipeline.

---

# Dataset

The dataset contains approximately five years of daily hotel operational data.

Total Records

* 1,827 Daily Observations

This first version of the model was created using synthetic hotel occupancy data. The synthetic dataset is useful for building the initial forecasting workflow, testing feature engineering, comparing algorithms, and validating the end-to-end model pipeline. The next model run will use messier data designed to be closer to real-world hotel data, where missing values, inconsistent patterns, operational noise, and changing demand behavior may produce different insights than the patterns observed in the synthetic dataset.

The original dataset contains more than 140 variables covering multiple aspects of hotel operations.

Instead of using every available variable, this project intentionally selects only the most meaningful forecasting features. Reducing unnecessary variables improves model interpretability, reduces overfitting, minimizes data leakage, and better reflects how professional forecasting models are developed.

---

# Selected Features

The final model uses carefully selected features from the following categories:

### Calendar Features

* Month
* Quarter
* Week of Year
* Day of Week
* Weekend Flag
* Season

### Holiday & Event Features

* Holiday Flag
* Special Event Flag
* Convention Indicator
* Cruise Ship Activity
* Days Until Next Holiday

### Weather Features

* Temperature
* Rain Indicator
* Storm Impact
* Hurricane Season

### Inventory Features

* Available Rooms
* Out of Order Rooms
* Maintenance Rooms
* Blocked Rooms

### Pricing Features

* ADR
* BAR Level
* Promotions
* Discount Percentage
* Length of Stay Restrictions

### Group Business

* Group Rooms
* Group Wash Percentage
* Number of Groups

### Booking Pace

* 14-Day On-The-Books Rooms
* 14-Day Occupancy
* Booking Pace Index

### Competitor Information

* Average Competitor ADR
* Competitor Occupancy

### Target Variable

* Occupancy Percentage

---

# Data Cleaning

The following preprocessing steps were performed before model training:

* Converted Date column into datetime format.
* Sorted observations chronologically.
* Removed unnecessary columns.
* Checked missing values.
* Checked duplicate records.
* Selected only meaningful forecasting variables.
* Created preprocessing pipelines.
* Encoded categorical variables.
* Imputed missing values.
* Scaled numerical variables where appropriate.

---

# Exploratory Data Analysis

The notebook includes detailed exploratory data analysis including:

* Occupancy Trend Over Time
* Monthly Occupancy Analysis
* Weekend vs Weekday Occupancy
* Holiday Demand Analysis
* Special Event Analysis
* Occupancy vs ADR
* Correlation Heatmap
* Feature Correlation Analysis
* Summary Statistics
* Missing Value Analysis

Each visualization includes a business interpretation from a Revenue Management perspective.

---

# Machine Learning Models

The following regression algorithms were trained and evaluated:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor
* Extra Trees Regressor

The models were compared using multiple evaluation metrics before selecting the best-performing model.

---

# Model Evaluation Metrics

Each model was evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Percentage Error (MAPE)
* R² Score

These metrics help evaluate both prediction accuracy and overall model performance.

---

# Visualizations

The notebook also includes:

* Actual vs Predicted Occupancy
* Residual Plot
* Feature Importance
* Correlation Heatmap
* Occupancy Trends
* Business Performance Charts

---

# Feature Importance

One of the key objectives of this project is not only to make accurate predictions but also to understand why the model makes those predictions.

Feature importance analysis helps identify which business factors contribute most to occupancy forecasting.

Examples include:

* Booking Pace
* ADR
* Group Rooms
* Weekend Demand
* Competitor Occupancy
* Holidays
* Special Events

This makes the model more explainable for Revenue Managers.

---

# Model Saving

The final trained model is saved using Joblib, allowing it to be reused without retraining.

Saved artifacts include:

* Trained Machine Learning Model
* Selected Feature List
* Model Evaluation Metrics

---

# Technologies Used

Programming Language

* Python

Data Analysis

* Pandas
* NumPy

Visualization

* Matplotlib
* Seaborn

Machine Learning

* Scikit-learn

Model Persistence

* Joblib

Development Environment

* Jupyter Notebook

---

# Revenue Management Applications

This project can support several hotel revenue management functions, including:

* Occupancy Forecasting
* Dynamic Pricing
* Revenue Optimization
* Group Business Evaluation
* Staffing Planning
* Inventory Management
* Event Impact Analysis
* Booking Pace Monitoring
* Market Demand Analysis

---

# Future Improvements

Some potential enhancements include:

* Integrating live PMS data.
* Using real STR market reports.
* Incorporating OTA booking trends.
* Adding airline and airport demand indicators.
* Including Google Trends search data.
* Building separate 30-day, 14-day, and 7-day forecasting models.
* Deploying the model as a Streamlit web application.
* Creating an interactive dashboard for Revenue Managers.
* Automating daily forecasting using scheduled data pipelines.
* Running the next model iteration on messier, more realistic hotel data to compare how demand patterns differ from the synthetic dataset.

---

# Repository Structure

```text
Hotel-Occupancy-Forecasting/
│
├── data/
│   └── moody_gardens_5yr_occupancy_dataset_v3.csv
│
├── notebooks/
│   └── Hotel_Occupancy_Forecasting.ipynb
│
├── models/
│   ├── occupancy_forecasting_model.pkl
│   ├── selected_feature_columns.pkl
│   └── model_evaluation_metrics.csv
│
├── README.md
│
└── requirements.txt
```

---

# Key Takeaways

This project combines practical Revenue Management concepts with Machine Learning techniques to build a realistic hotel occupancy forecasting model.

Rather than focusing solely on prediction accuracy, the project emphasizes business understanding, explainability, and real-world applicability. By combining booking pace, pricing strategy, seasonality, events, market conditions, and operational data, the model demonstrates how data-driven forecasting can support more informed revenue management decisions.

---


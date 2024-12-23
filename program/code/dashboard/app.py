import os
import pandas as pd
import streamlit as st
from utils import date_time_range_selection
from timeline import timeline
from constants import load_threshold_values, threshold_inputs

st.set_page_config(layout="wide")
st.title("Alerting System Hyperparameters")

outlier_tolerance_input_default = 5
outlier_tolerance_input = st.text_input("Outlier Tolerance", value=outlier_tolerance_input_default)
outlier_tolerance = int(outlier_tolerance_input) if outlier_tolerance_input else outlier_tolerance_input_default
st.slider("Outlier Tolerance", 0, 60, outlier_tolerance)

# Slider with manual input for grouping_time_window
grouping_time_window_input_default = 200
grouping_time_window_input = st.text_input("Grouping Time Window (seconds)", value=grouping_time_window_input_default)
grouping_time_window = int(grouping_time_window_input) if grouping_time_window_input else grouping_time_window_input_default
st.slider("Grouping Time Window (seconds)", 0, 10000, grouping_time_window)

# Slider with manual input for anomaly_threshold
anomaly_threshold_input_default = 6
anomaly_threshold_input = st.text_input("3-Sigma Anomaly Threshold", value=anomaly_threshold_input_default)
anomaly_threshold = int(anomaly_threshold_input) if anomaly_threshold_input else anomaly_threshold_input_default
st.slider("3-Sigma Anomaly Threshold", 1, 10, anomaly_threshold)

start_datetime, end_datetime = date_time_range_selection()

from summary_tab import summary_tab
# from cache_management import cache_management
from alerting_system import alerting_system
from constants import paths, folders

# TODO:
# - have loading/or progression better indicator whenever something is loading
# - make sure all HTOL dashboards are working and it is easy to switch between them
# - see if there are any ways to improve performance of the code
# - restructure the code (follow the DRY principle) and make it more modular
# - see if the statistical method can be integrated in any way into this
# - see if there are any other ways to improve this dashboard

# IMPORTANT TODO:
# - add the stats dots as a separate category (check that the stat point is outside the normal operation region before confirming it)
# - when grouping, the severity selected for the ONE ALERT representing the group should be the item of highest frequency (if draw, then highest severity amongsst the highest frequencies ones)
# - make sure we're able to change the dynamic values and everything gets refreshed to reflect it

# - selection range of dates to show the alerts (in all the tabs)
# - caching of alerts summary (assuming hyperparameters are fixed) even when program is restarted (to make sure only new rows in the csv files get processed in the background process)
# - a new tab with summarised alerts data for all HTOL machines

aggregated_counts = {'low': 0, 'medium': 0, 'high': 0, '3-sigma': 0}
total_data = {}

# Load available variables from the first HTOL folder
htol_folder = "HTOL-09"  # Assuming all HTOL folders have the same data structure
file_path = os.path.join(htol_folder, os.listdir(htol_folder)[0])
df = pd.read_csv(file_path, skiprows=3)
available_variables = df.columns.tolist()

# Dropdown to select variable
selected_variable = st.selectbox("Select Variable", available_variables, key="select_variable", index=available_variables.index("ChlPrs"))

threshold_inputs(selected_variable)
bands = load_threshold_values(selected_variable)

def HTOL_09_content():
    HTOL = "HTOL-09"
    alert_data, alert_counts = alerting_system(HTOL, outlier_tolerance, grouping_time_window, anomaly_threshold, start_datetime, end_datetime, selected_variable, bands)
    total_data[HTOL] = { "alert_data": alert_data, "alert_counts": alert_counts}
    for key in aggregated_counts.keys():
        aggregated_counts[key] += alert_counts[key]

def HTOL_10_content():
    HTOL = "HTOL-10"
    alert_data, alert_counts = alerting_system(HTOL, outlier_tolerance, grouping_time_window, anomaly_threshold, start_datetime, end_datetime, selected_variable, bands)
    total_data[HTOL] = { "alert_data": alert_data, "alert_counts": alert_counts}
    for key in aggregated_counts.keys():
        aggregated_counts[key] += alert_counts[key]

def HTOL_11_content():
    HTOL = "HTOL-11"
    alert_data, alert_counts = alerting_system(HTOL, outlier_tolerance, grouping_time_window, anomaly_threshold, start_datetime, end_datetime, selected_variable, bands)
    total_data[HTOL] = { "alert_data": alert_data, "alert_counts": alert_counts}
    for key in aggregated_counts.keys():
        aggregated_counts[key] += alert_counts[key]

def HTOL_12_content():
    HTOL = "HTOL-12"
    alert_data, alert_counts = alerting_system(HTOL, outlier_tolerance, grouping_time_window, anomaly_threshold, start_datetime, end_datetime, selected_variable, bands)
    total_data[HTOL] = { "alert_data": alert_data, "alert_counts": alert_counts}
    for key in aggregated_counts.keys():
        aggregated_counts[key] += alert_counts[key]

def HTOL_13_content():
    HTOL = "HTOL-13"
    alert_data, alert_counts = alerting_system(HTOL, outlier_tolerance, grouping_time_window, anomaly_threshold, start_datetime, end_datetime, selected_variable, bands)
    total_data[HTOL] = { "alert_data": alert_data, "alert_counts": alert_counts}
    for key in aggregated_counts.keys():
        aggregated_counts[key] += alert_counts[key]

def HTOL_14_content():
    HTOL = "HTOL-14"
    alert_data, alert_counts = alerting_system(HTOL, outlier_tolerance, grouping_time_window, anomaly_threshold, start_datetime, end_datetime, selected_variable, bands)
    total_data[HTOL] = { "alert_data": alert_data, "alert_counts": alert_counts}
    for key in aggregated_counts.keys():
        aggregated_counts[key] += alert_counts[key]

def HTOL_15_content():
    HTOL = "HTOL-15"
    alert_data, alert_counts = alerting_system(HTOL, outlier_tolerance, grouping_time_window, anomaly_threshold, start_datetime, end_datetime, selected_variable, bands)
    total_data[HTOL] = { "alert_data": alert_data, "alert_counts": alert_counts}
    for key in aggregated_counts.keys():
        aggregated_counts[key] += alert_counts[key]

# Create the tabs
HTOL_09, HTOL_10, HTOL_11, HTOL_12, HTOL_13, HTOL_14, HTOL_15, Combined_HTOLs, High_Severity_Timeline, Medium_Severity_Timeline, Low_Severity_Timeline, Sigma_Severity_Timeline  = st.tabs(folders + ["Combined_HTOLs",  "High Severity Timeline", "Medium Severity Timeline", "Low Severity Timeline", "3 Sigma Severity_Timeline"])

# Display content within each tab
with HTOL_09:
    HTOL_09_content()

with HTOL_10:
    HTOL_10_content()

with HTOL_11:
    HTOL_11_content()

with HTOL_12:
    HTOL_12_content()

with HTOL_13:
    HTOL_13_content()

with HTOL_14:
    HTOL_14_content()

with HTOL_15:
    HTOL_15_content()

with Combined_HTOLs:
    summary_tab(total_data, aggregated_counts)

with High_Severity_Timeline:
    severity = "high"
    timeline(total_data, severity, selected_variable, bands)

with Medium_Severity_Timeline:
    severity = "medium"
    timeline(total_data, severity, selected_variable, bands)

with Low_Severity_Timeline:
    severity = "low"
    timeline(total_data, severity, selected_variable, bands)

with Sigma_Severity_Timeline:
    severity = "3-sigma"
    timeline(total_data, severity, selected_variable, bands)

# with Cache_Management:
#     cache_management()
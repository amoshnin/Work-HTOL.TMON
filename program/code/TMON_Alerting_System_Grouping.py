import streamlit as st
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# Hyperparameters
st.set_page_config(layout="wide")
st.title("Alerting System Hyperparameters")
machine_state = "idle"  # Or get it from your DataFrame if it changes

def detect_sensor_anomalies(df, variable_title, idle_bands, run_bands, outlier_tolerance=3):
    """
    Detects anomalies in sensor data based on machine state and severity levels,
    with consecutive outlier tolerance.

    Args:
        df: A pandas DataFrame with columns 'time', variable_title, and 'machine_state'.
        idle_threshold: The normal operation threshold for the idle state.
        run_threshold: The normal operation threshold for the running state.
        tolerance: The number of consecutive outliers required to trigger an alert.

    Returns:
        A pandas DataFrame with alert indices and severity levels.
    """

    # Initialize consecutive outlier counter
    consecutive_outliers = 0

    # Initialize alert list
    alerts = []

    # Iterate through data and detect anomalies
    for i, row in df.iterrows():
        value = row[variable_title]
        bands = idle_bands if machine_state == 'idle' else run_bands

        # Check if value is outside normal range
        if (bands['low'][0] <= value <= bands['low'][1]) or (bands['low'][2] <= value <= bands['low'][3]):
            severity = 'low'
            consecutive_outliers += 1
        elif (bands['medium'][0] <= value <= bands['medium'][1]) or (bands['medium'][2] <= value <= bands['medium'][3]):
            severity = 'medium'
            consecutive_outliers += 1
        elif (bands['high'][0] <= value <= bands['high'][1]) or (bands['high'][2] <= value <= bands['high'][3]):
            severity = 'high'
            consecutive_outliers += 1
        else:  # Value is within normal range
            consecutive_outliers = 0
            continue  # Skip to next iteration if no anomaly

        if consecutive_outliers >= outlier_tolerance:
            alerts.append((i, severity))

    # Create DataFrame from alert list
    alert_df = pd.DataFrame(alerts, columns=['alert_index', 'severity'])

    return alert_df

Stat_Outlier_Detector_Sensitivity = st.slider("Stat Outlier Detector Sensitivity", 1, 15, 6)

def anomaly_detection_3_sigma_rule(data):
    chiller_pressure_title = "ChlPrs"

    # Calculate mean and standard deviation of the chiller_pressure_title column
    mean_pressure = np.mean(data[chiller_pressure_title])
    std_pressure = np.std(data[chiller_pressure_title])

    # Define the threshold for anomaly detection (3-sigma)
    anomaly_threshold = Stat_Outlier_Detector_Sensitivity

    # Create a new column to flag anomalies
    data['anomaly'] = 0  # Initialize all values to 0 (not an anomaly)

    # Flag anomalies based on the 3-sigma rule
    data.loc[data[chiller_pressure_title] > mean_pressure + anomaly_threshold * std_pressure, 'anomaly'] = 1
    data.loc[data[chiller_pressure_title] < mean_pressure - anomaly_threshold * std_pressure, 'anomaly'] = 1

@st.cache_data(show_spinner=False)
def visualise_time_series(df, variable_title, idle_bands, run_bands, alerts_indices, file_name):
    # Determine threshold and bands based on machine_state
    bands = idle_bands if machine_state == 'idle' else run_bands

    # Create Plotly figure
    fig = go.Figure()

    opacity = 0.2
    line_width = 0.5

    # Add background color bands
    fig.add_hrect(y0=bands['low'][1], y1=bands['low'][2], line_width=line_width, fillcolor="green", opacity=opacity, name='Normal')
    fig.add_hrect(
                y0=bands['low'][1],
                y1=bands['low'][0],
                line_width=line_width,
                fillcolor="yellow",
                opacity=opacity,
                name="LOW"
            )
    fig.add_hrect(
                y0=bands['low'][2],
                y1=bands['low'][3],
                line_width=line_width,
                fillcolor="yellow",
                opacity=opacity,
                name="LOW"
            )
    fig.add_hrect(
                y0=bands['medium'][1],
                y1=bands['medium'][0],
                line_width=line_width,
                fillcolor="orange",
                opacity=opacity,
                name="MEDIUM"
            )
    fig.add_hrect(
                y0=bands['medium'][2],
                y1=bands['medium'][3],
                line_width=line_width,
                fillcolor="orange",
                opacity=opacity,
                name="MEDIUM"
            )
    fig.add_hrect(
                y0=bands['high'][1],
                y1=bands['high'][1] - 30,
                line_width=line_width,
                fillcolor="red",
                opacity=opacity,
                name="HIGH"
            )
    fig.add_hrect(
                y0=bands['high'][2],
                y1=bands['high'][2] + 30,
                line_width=line_width,
                fillcolor="red",
                opacity=opacity,
                name="HIGH"
            )


    size = 8

    # Add sensor value line plot
    fig.add_trace(go.Scatter(x=df['Time'], y=df[variable_title], mode='lines', name='Sensor Value', line=dict(color='blue')))

    # Filter and add alert scatter plot (separate traces for each severity)
    for severity in ['low', 'medium', 'high', "3-sigma"]:  # Iterate through all severities
        severity_alerts = alerts_indices[alerts_indices['severity'] == severity]
        fig.add_trace(go.Scatter(
            x=df.loc[severity_alerts['alert_index'], 'Time'],
            y=df.loc[severity_alerts['alert_index'], variable_title],
            mode='markers',
            marker=dict(
                color='yellow' if severity == 'low' else 'orange' if severity == 'medium' else 'red' if severity == 'high' else 'purple',
                size=8,
                line=dict(width=2, color='black') if severity == '3-sigma' else None
            ),
            name=f'Alert ({severity})',
            showlegend=True  # Always show in legend
        ))

    highest_severity_band = bands['high']
    lower_bound, upper_bound = highest_severity_band[1] - 2, highest_severity_band[2] + 2

    # Set layout and display
    fig.update_layout(
        title=f'Sensor Value Over Time for {file_name}',
        xaxis_title='Time',
        yaxis_title=variable_title,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        yaxis_range=[lower_bound, upper_bound]  # Set default zoom
    )

    st.plotly_chart(fig)

def group_alerts(df, alerts_indices, grouping_time_window):
    """
    Groups alerts based on time proximity.

    Args:
        df: The original DataFrame containing the 'Time' column.
        alerts_indices: The DataFrame with 'alert_index' and 'severity' columns.
        grouping_time_window: The time window (in seconds) for grouping alerts.

    Returns:
        A DataFrame with grouped alert indices and severity levels.

    group_alerts function:
        - Efficiently groups alerts based on the time window.
        - Selects the middle alert within each group as the representative alert.
        - Assigns the highest severity among the grouped alerts to the representative alert.
    """

    # Convert 'Time' column in df to datetime format
    df['Time'] = pd.to_datetime(df['Time'])

    # Merge alerts_indices with df to get the timestamps of the alerts
    alerts_with_time = pd.merge(alerts_indices, df[['Time']], left_on='alert_index', right_index=True)

    # Sort alerts by time
    alerts_with_time = alerts_with_time.sort_values('Time')

    # Initialize variables for grouping
    grouped_alerts = []
    current_group = []

    for _, row in alerts_with_time.iterrows():
        if not current_group:
            current_group.append(row)
        else:
            time_diff = (row['Time'] - current_group[-1]['Time']).total_seconds()
            if time_diff <= grouping_time_window:
                current_group.append(row)
            else:
                highest_sensor_alert = max(current_group, key=lambda x: df.loc[x['alert_index'], chiller_pressure_title])
                grouped_alerts.append(highest_sensor_alert)
                current_group = [row]

    if current_group:
        highest_sensor_alert = max(current_group, key=lambda x: df.loc[x['alert_index'], chiller_pressure_title])
        grouped_alerts.append(highest_sensor_alert)

    # Convert the grouped alerts back to a DataFrame
    grouped_alerts_df = pd.DataFrame(grouped_alerts, columns=alerts_indices.columns)

    return grouped_alerts_df

data_directory = "../"

outlier_tolerance = st.slider("Outlier Tolerance", 0, 60, 5)
grouping_time_window = st.slider("Grouping Time Window (seconds)", 0, 3000, 200)

idle_threshold = 32
run_threshold = 32

# Define severity bands based on machine state
idle_bands = {
    'low': (idle_threshold - 3, idle_threshold - 1) + (idle_threshold + 1, idle_threshold + 3),
    'medium': (idle_threshold - 5, idle_threshold - 3) + (idle_threshold + 3, idle_threshold + 5),
    'high': (float('-inf'), idle_threshold - 5) + (idle_threshold + 5, float('inf'))
}

run_bands = {
    'low': (run_threshold - 5, run_threshold - 3) + (run_threshold + 3, run_threshold + 5),
    'medium': (run_threshold - 7, run_threshold - 5) + (run_threshold + 5, run_threshold + 7),
    'high': (float('-inf'), run_threshold - 7) + (run_threshold + 7, float('inf'))
}

chiller_pressure_title = "ChlPrs"
folders = ["HTOL-09", "HTOL-10", "HTOL-11", "HTOL-12", "HTOL-13", "HTOL-14", "HTOL-15"]
paths = [os.path.join(data_directory, folder_path) for folder_path in folders]

def extract_date(text):
    match = re.search(r'# START:(\d{1,2}/\d{1,2}/\d{4})', text)
    if match:
        return pd.to_datetime(match.group(1), format='%m/%d/%Y')
    else:
        return None

def alerting_system(HTOL_name):
    for file_name in os.listdir(HTOL_name):
        if "HTOL" in file_name:
            file_path = os.path.join(HTOL_name, file_name)

            # Read the first row to get the event date
            with open(file_path, 'r') as f:
                first_row = f.readline()
            event_date = extract_date(first_row)

            # Read the CSV, skipping the first 3 rows (including the header with the event date)
            df = pd.read_csv(file_path, skiprows=3)[['Time', chiller_pressure_title]]

            # Convert the 'Time' column to datetime, assuming it contains only time information
            df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time

            # Combine the event date with the time information
            if event_date:
                df['Time'] = df['Time'].apply(lambda x: datetime.combine(event_date, x))
            else:
                print("Warning: Could not extract event date from the first row.")

            alerts_indices = detect_sensor_anomalies(df, chiller_pressure_title, idle_bands, run_bands, outlier_tolerance=outlier_tolerance)
            grouped_alerts_indices = group_alerts(df, alerts_indices, grouping_time_window)

            col1, col2 = st.columns(2)

            with col1:
                st.header("Original Alerts (ungrouped)")
                visualise_time_series(df, chiller_pressure_title, idle_bands, run_bands, alerts_indices, file_name)

            with col2:
                st.header("Grouped Alerts")
                visualise_time_series(df, chiller_pressure_title, idle_bands, run_bands, grouped_alerts_indices, file_name)

# file_name = "combined_data.csv"
def HTOL_09_content():
    HTOL = "HTOL-09"
    alerting_system(HTOL)

def HTOL_10_content():
    HTOL = "HTOL-10"
    alerting_system(HTOL)

def HTOL_11_content():
    HTOL = "HTOL-11"
    alerting_system(HTOL)

def HTOL_12_content():
    HTOL = "HTOL-12"
    alerting_system(HTOL)

def HTOL_13_content():
    HTOL = "HTOL-13"
    alerting_system(HTOL)

def HTOL_14_content():
    HTOL = "HTOL-14"
    alerting_system(HTOL)

def HTOL_15_content():
    HTOL = "HTOL-15"
    alerting_system(HTOL)

# Create the tabs
HTOL_09, HTOL_10, HTOL_11, HTOL_12, HTOL_13, HTOL_14, HTOL_15 = st.tabs(folders)

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

def join_csvs_by_date(folder_path):
    """
    This function sorts CSV files in a folder by their filename (formatted as HTOL-09-20240314095049.csv),
    then joins them all into one CSV and saves it in the same folder.

    Args:
        folder_path (str): The path to the folder containing the CSV files.
    """


    # Get a list of all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Read each CSV into a DataFrame and concatenate them
    dfs = [pd.read_csv(os.path.join(folder_path, f), skiprows=3) for f in csv_files]
    combined_df = pd.concat(dfs, ignore_index=True)

    # Save the combined DataFrame to a new CSV file in the same folder
    combined_filename = 'combined_data.csv'
    combined_filepath = os.path.join(folder_path, combined_filename)
    combined_df.to_csv(combined_filepath, index=False)

    print(f"CSVs combined and saved as {combined_filename} in {folder_path}")

# for directory in os.listdir(data_directory):
#     if directory.startswith("HTOL-"):
#         machine_folder_path = os.path.join(data_directory, directory)
#         join_csvs_by_date(machine_folder_path)

        # print(machine_folder_path)
        # for file_name in os.listdir(machine_folder_path):
        #     file_path = os.path.join(machine_folder_path, file_name)
        #     df = pd.read_csv(file_path, skiprows=3)[['Time', chiller_pressure_title]]

        #     alerts_indices = detect_sensor_anomalies(df, chiller_pressure_title, idle_bands, run_bands, outlier_tolerance=outlier_tolerance)
        #     visualise_time_series(df, chiller_pressure_title, idle_threshold, run_threshold, idle_bands, run_bands, alerts_indices, file_name)
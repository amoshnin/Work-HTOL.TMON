import pandas as pd
import numpy as np
from constants import machine_state

def group_alerts(df, alerts_indices, grouping_time_window, selected_variable):
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
                highest_sensor_alert = max(current_group, key=lambda x: df.loc[x['alert_index'], selected_variable])
                grouped_alerts.append(highest_sensor_alert)
                current_group = [row]

    if current_group:
        highest_sensor_alert = max(current_group, key=lambda x: df.loc[x['alert_index'], selected_variable])
        grouped_alerts.append(highest_sensor_alert)

    # Convert the grouped alerts back to a DataFrame
    grouped_alerts_df = pd.DataFrame(grouped_alerts, columns=alerts_indices.columns)

    return grouped_alerts_df

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

def anomaly_detection_3_sigma_rule(data, idle_bands, run_bands, anomaly_threshold, selected_variable):
    """
    Detects anomalies using the 3-sigma rule and filters based on normal operation bands.

    Args:
        data: A pandas DataFrame with columns 'Time', 'ChlPrs' (chiller pressure), and potentially 'machine_state'.
        idle_bands: Dictionary defining normal operation bands for the 'idle' machine state.
        run_bands: Dictionary defining normal operation bands for the 'run' machine state.
        anomaly_threshold: The threshold (in number of standard deviations) for anomaly detection.

    Returns:
        A pandas DataFrame with alert indices and severity levels.
    """

    # Calculate mean and standard deviation
    mean_pressure = np.mean(data[selected_variable])
    std_pressure = np.std(data[selected_variable])

    # Flag potential anomalies based on 3-sigma rule
    potential_anomalies = (data[selected_variable] > mean_pressure + anomaly_threshold * std_pressure) | \
                          (data[selected_variable] < mean_pressure - anomaly_threshold * std_pressure)

    # Get machine state (assuming it's in the DataFrame, otherwise set it appropriately)
    machine_state = data['machine_state'].iloc[0] if 'machine_state' in data.columns else 'idle'
    bands = idle_bands if machine_state == 'idle' else run_bands

    # Filter potential anomalies based on normal operation bands
    is_within_normal_bands = (
        (data[selected_variable] >= bands['low'][0]) & (data[selected_variable] <= bands['low'][1])
    ) | (
        (data[selected_variable] >= bands['low'][2]) & (data[selected_variable] <= bands['low'][3])
    )

    # Combine conditions to get final anomalies
    anomalies = potential_anomalies & ~is_within_normal_bands

    # Create the alert DataFrame
    alert_df = pd.DataFrame({
        'alert_index': anomalies[anomalies].index,
        'severity': '3-sigma'
    })

    return alert_df
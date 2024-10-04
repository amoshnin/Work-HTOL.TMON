import streamlit as st
import pandas as pd
import os
from data_processing import process_HTOL_data
from visualization import visualise_time_series
from constants import idle_bands, run_bands

def compute_alert_counts(grouped_alerts_indices):
    """
    Computes the count of alerts for the specific severity and other severities.

    Args:
        grouped_alerts_indices: The DataFrame with 'alert_index' and 'severity' columns.

    Returns:
        A tuple containing the count of specific severity alerts, medium alerts, and high alerts.
    """
    low_count = (grouped_alerts_indices['severity'] == "low").sum()
    medium_count = (grouped_alerts_indices['severity'] == 'medium').sum()
    high_count = (grouped_alerts_indices['severity'] == 'high').sum()
    sigma_count = (grouped_alerts_indices['severity'] == '3-sigma').sum()
    return low_count, medium_count, high_count, sigma_count

def graph_button_title(button_key, severity):
    isVisible = st.session_state[button_key]
    return f"{"Show" if isVisible else "Hide"} {severity} Severity Graphs"

def graph_visible_title(HTOL_name, severity):
    return f"graph_visible_{HTOL_name}_{severity}"

def save_alert_data_to_csv(alert_data, HTOL_name, outlier_tolerance, grouping_time_window, anomaly_threshold, start_datetime, end_datetime):
    """
    Saves the alert data for a specific HTOL machine to a CSV file,
    including all rows from the original data and filling blanks for no alerts.
    """
    all_alerts = []

    for file_name, data in alert_data.items():
        df = data['df']
        grouped_alerts_indices = data['grouped_alerts_indices']

        # Map severity to "ALERT" column values
        grouped_alerts_indices['ALERT'] = grouped_alerts_indices['severity'].map({
            'low': 'LOW',
            'medium': 'MEDIUM',
            'high': 'HIGH',
            '3-sigma': 'SIGMA'
        })

        # Perform a left join to include all rows from the original DataFrame
        merged_df = pd.merge(df, grouped_alerts_indices[['alert_index', 'ALERT']], left_index=True, right_on='alert_index', how='left')

        # Add file name
        merged_df['file_name'] = file_name

        # Fill NaN values in 'ALERT' with blanks
        merged_df['ALERT'] = merged_df['ALERT'].fillna('')

        all_alerts.append(merged_df)

    # Combine all alerts
    combined_df = pd.concat(all_alerts)

    # Sort by time
    combined_df = combined_df.sort_values('Time')

    # Construct a unique folder name based on hyperparameters
    folder_name = f"outlier_tolerance={outlier_tolerance}_grouping_time_window={grouping_time_window}_anomaly_threshold={anomaly_threshold}_start_date={start_datetime.strftime('%Y-%m-%d')}_end_date={end_datetime.strftime('%Y-%m-%d')}"

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Save to CSV inside the folder
    csv_file_name = os.path.join(folder_name, f"{HTOL_name}_alerts.csv")
    combined_df.to_csv(csv_file_name, index=False)

    print(f"Alert data saved to {csv_file_name}")

# Helper function to get sort key based on selected option and severity
def get_sort_key(data, severity, sort_option):
    if sort_option.startswith("Date"):
        return data['event_date'].timestamp() if sort_option == "Date (Oldest First)" else -data['event_date'].timestamp()
    else:
        alert_count = (data['grouped_alerts_indices']['severity'] == severity).sum()
        return -alert_count if sort_option == "Alert Count (Highest First)" else alert_count

def display_alert_charts(alert_data, severity, sort_by, HTOL_name, selected_variable):
    """
    Displays the alert charts for a specific severity type with sorting options.
    """
    spinner_text = f"Loading {severity} severity alerts..."

    with st.spinner(spinner_text):
        filtered_alert_data = {k: v for k, v in alert_data.items() if k.startswith(HTOL_name)}

    for file_name, data in sorted(filtered_alert_data.items(), key=lambda item: get_sort_key(item[1], severity, sort_by)):
            low_count, medium_count, high_count, sigma_count = compute_alert_counts(data['grouped_alerts_indices'])
            if (data['grouped_alerts_indices']['severity'] == severity).sum() == 0:
                continue
            st.subheader(f"{file_name} (Low: {low_count}, Medium: {medium_count}, High: {high_count}, 3-Sigma: {sigma_count})")
            alerts_indices = data['grouped_alerts_indices']
            visualise_time_series(data['df'], selected_variable, idle_bands, run_bands, alerts_indices, file_name)

def alerting_system(HTOL_name, outlier_tolerance, grouping_time_window, anomaly_threshold, start_datetime, end_datetime, selected_variable):
    alert_counts, alert_data = process_HTOL_data(HTOL_name, outlier_tolerance, grouping_time_window, anomaly_threshold, start_datetime, end_datetime, selected_variable)
    # Display alert summary
    st.subheader(f"Alert Summary for {HTOL_name}")
    col1, col2, col3, col4 = st.columns(4)

    # Sorting options for each column
    for severity in ['low', 'medium', 'high', "3-sigma"]:
        button_title = graph_visible_title(HTOL_name, severity)
        if button_title not in st.session_state:
            st.session_state[button_title] = False

    with col1:
        severity = "low"
        sort_by_low = st.selectbox(f"Sort {severity} Severity by:", ["Date (Newest First)", "Date (Oldest First)", "Alert Count (Highest First)", "Alert Count (Lowest First)"], key=f"sort_by_{severity}_{HTOL_name}")
        st.metric(f"{severity} Severity Alerts", alert_counts[severity], delta_color="off", label_visibility="collapsed")
        st.markdown(f'<span style="color:yellow; font-size:20px">{"⬤"}</span>', unsafe_allow_html=True)
        button_key = graph_visible_title(HTOL_name, severity)
        if alert_counts[severity] > 0:
            if st.button("Toggle Visibility of Graphs", key=f"button_{button_key}"):
                st.session_state[button_key] = not st.session_state[button_key]
    with col2:
        severity = "medium"
        sort_by_medium = st.selectbox(f"Sort {severity} Severity by:", ["Date (Newest First)", "Date (Oldest First)", "Alert Count (Highest First)", "Alert Count (Lowest First)"], key=f"sort_by_{severity}_{HTOL_name}")
        st.metric(f"{severity} Severity Alerts", alert_counts[severity], delta_color="off", label_visibility="collapsed")
        st.markdown(f'<span style="color:orange; font-size:20px">{"⬤"}</span>', unsafe_allow_html=True)
        button_key = graph_visible_title(HTOL_name, severity)
        if alert_counts[severity] > 0:
            if st.button("Toggle Visibility of Graphs", key=f"button_{button_key}"):
                st.session_state[button_key] = not st.session_state[button_key]
    with col3:
        severity = "high"
        sort_by_high = st.selectbox(f"Sort {severity} Severity by:", ["Date (Newest First)", "Date (Oldest First)", "Alert Count (Highest First)", "Alert Count (Lowest First)"], key=f"sort_by_{severity}_{HTOL_name}")
        st.metric(f"{severity} Severity Alerts", alert_counts[severity], delta_color="off", label_visibility="collapsed")
        st.markdown(f'<span style="color:red; font-size:20px">{"⬤"}</span>', unsafe_allow_html=True)
        button_key = graph_visible_title(HTOL_name, severity)
        if alert_counts[severity] > 0:
            if st.button("Toggle Visibility of Graphs", key=f"button_{button_key}"):
                st.session_state[button_key] = not st.session_state[button_key]
    with col4:
        severity = "3-sigma"
        sort_by_high = st.selectbox(f"Sort {severity} Severity by:", ["Date (Newest First)", "Date (Oldest First)", "Alert Count (Highest First)", "Alert Count (Lowest First)"], key=f"sort_by_{severity}_{HTOL_name}")
        st.metric(f"{severity} Severity Alerts", alert_counts[severity], delta_color="off", label_visibility="collapsed")
        st.markdown(f'<span style="color:purple; font-size:20px">{"⬤"}</span>', unsafe_allow_html=True)
        button_key = graph_visible_title(HTOL_name, severity)
        if alert_counts[severity] > 0:
            if st.button("Toggle Visibility of Graphs", key=f"button_{button_key}"):
                st.session_state[button_key] = not st.session_state[button_key]

    with col1:
        severity = "low"
        if st.session_state[graph_visible_title(HTOL_name, severity)]:
            display_alert_charts(alert_data, severity, sort_by_low, HTOL_name, selected_variable)
    with col2:
        severity = "medium"
        if st.session_state[graph_visible_title(HTOL_name,severity)]:
            display_alert_charts(alert_data, severity, sort_by_medium, HTOL_name, selected_variable)
    with col3:
        severity = "high"
        if st.session_state[graph_visible_title(HTOL_name, severity)]:
            display_alert_charts(alert_data, severity, sort_by_high, HTOL_name, selected_variable)
    with col4:
        severity = "3-sigma"
        if st.session_state[graph_visible_title(HTOL_name, severity)]:
            display_alert_charts(alert_data, severity, sort_by_high, HTOL_name, selected_variable)

    # save_alert_data_to_csv(alert_data, HTOL_name, outlier_tolerance, grouping_time_window, anomaly_threshold, start_datetime, end_datetime)

    return alert_data, alert_counts
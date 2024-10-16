import re
import hashlib
import pandas as pd
import streamlit as st
import json
from datetime import time, datetime

def extract_date(text):
    match = re.search(r'# START:(\d{1,2}/\d{1,2}/\d{4})', text)
    if match:
        return pd.to_datetime(match.group(1), format='%m/%d/%Y')
    else:
        return None

def hash_hyperparameters(outlier_tolerance, grouping_time_window, anomaly_threshold, selected_variable, bands):
    combined_str = f"{outlier_tolerance}_{type(outlier_tolerance).__name__}_{grouping_time_window}_{type(grouping_time_window).__name__}_{anomaly_threshold}_{type(anomaly_threshold).__name__}_{selected_variable}_{type(selected_variable).__name__}_{json.dumps(bands)}"
    return hashlib.md5(combined_str.encode()).hexdigest()

def date_time_range_selection():
    """
    Provides a Streamlit UI for selecting a date and time range.

    Returns:
        A tuple containing the start and end datetime objects based on the user's selection.
    """
    # Set the default start and end dates
    default_start_date = datetime(year=2022, month=1, day=1)
    default_end_date = datetime(year=2026, month=1, day=1)

    # Initialize session state variables to store the selected date range
    if "start_datetime" not in st.session_state:
        st.session_state.start_datetime = datetime.combine(default_start_date, datetime.min.time())
    if "end_datetime" not in st.session_state:
        st.session_state.end_datetime = datetime.combine(default_end_date, datetime.min.time())

    result = st.date_input(
        "Select Date Range",
        [default_start_date, default_end_date],
        key="date_range_"
    )

    # Check if both start and end dates have been selected
    if len(result) == 2:
        start_date, end_date = result[0], result[1]
        # Update session state with the new selection
        st.session_state.start_datetime = datetime.combine(start_date, datetime.min.time())
        st.session_state.end_datetime = datetime.combine(end_date, datetime.min.time())

    return st.session_state.start_datetime, st.session_state.end_datetime
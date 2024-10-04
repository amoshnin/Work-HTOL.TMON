import os
import streamlit as st
import json

CACHE_DIR = ".cache"
data_directory = "./"

idle_threshold = 32
run_threshold = 32

def define_bands(threshold, low_range, medium_range, high_range):
    """
    Defines the bands based on the provided threshold and ranges.
    """
    bands = {
        "low": (threshold - medium_range, threshold - low_range) + (threshold + low_range, threshold + medium_range),
        "medium": (threshold - high_range, threshold - medium_range) + (threshold + medium_range, threshold + high_range),
        "high": (float("-inf"), threshold - high_range) + (threshold + high_range, float("inf")),
    }
    return bands

# Initialize the bands with default values
idle_bands = define_bands(32, 1, 3, 5)  # Assuming default values
run_bands = define_bands(32, 3, 5, 7)  # Assuming default values

def load_threshold_values(selected_variable):
    """Loads threshold values from cache or uses default values."""
    cache_file = os.path.join(CACHE_DIR, f"thresholds_{selected_variable}.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            try:
                data = json.load(f)
                return {
                    "idle_bands": define_bands(
                        data["idle_threshold"],
                        data["idle_low_band_range"],
                        data["idle_medium_band_range"],
                        data["idle_high_band_range"],
                    ),
                    "run_bands": define_bands(
                        data["run_threshold"],
                        data["run_low_band_range"],
                        data["run_medium_band_range"],
                        data["run_high_band_range"],
                    )
                }
            except ValueError:
                return {
                    "idle_bands":idle_bands,
                    "run_bands": run_bands
                }
    else:
        return {
            "idle_bands":idle_bands,
            "run_bands": run_bands
        }

def save_threshold_values(
    selected_variable,
    idle_threshold,
    run_threshold,
    #
    idle_low_band_range,
    run_low_band_range,
    #
    idle_medium_band_range,
    run_medium_band_range,
    #
    idle_high_band_range,
    run_high_band_range
):
    """Saves threshold values to a cache file."""
    cache_file = os.path.join(CACHE_DIR, f"thresholds_{selected_variable}.json")
    data = {
        "idle_threshold": idle_threshold,
        "run_threshold": run_threshold,
        #
        "idle_low_band_range": idle_low_band_range,
        "run_low_band_range": run_low_band_range,
        #
        "idle_medium_band_range": idle_medium_band_range,
        "run_medium_band_range": run_medium_band_range,
        #
        "idle_high_band_range": idle_high_band_range,
        "run_high_band_range": run_high_band_range
    }
    with open(cache_file, "w") as f:
        json.dump(data, f)

def threshold_inputs(selected_variable):
    """Displays input fields for threshold parameters."""
    st.subheader("Threshold Configuration")
    col1, col2 = st.columns(2)
    with col1:
        idle_threshold = st.number_input(
            "Idle Threshold", min_value=0, value=32, key=f"idle_{selected_variable}"
        )
    with col2:
        run_threshold = st.number_input(
            "Run Threshold", min_value=0, value=32, key=f"run_{selected_variable}"
        )
    col3, col4, col5 = st.columns(3)
    with col3:
        idle_low_band_range = st.number_input(
            "Idle Low Band Range (+/-)",
            min_value=1,
            value=1,
            key=f"idle_low_{selected_variable}",
        )
        run_low_band_range = st.number_input(
            "Run Low Band Range (+/-)",
            min_value=1,
            value=3,
            key=f"run_low_{selected_variable}",
        )
    with col4:
        idle_medium_band_range = st.number_input(
            "Idle Medium Band Range (+/-)",
            min_value=1,
            value=3,
            key=f"idle_medium_{selected_variable}",
        )
        run_medium_band_range = st.number_input(
            "Run Medium Band Range (+/-)",
            min_value=1,
            value=5,
            key=f"run_medium_{selected_variable}",
        )
    with col5:
        idle_high_band_range = st.number_input(
            "Idle High Band Range (+/-)",
            min_value=1,
            value=5,
            key=f"idle_high_{selected_variable}",
        )
        run_high_band_range = st.number_input(
            "Run High Band Range (+/-)",
            min_value=1,
            value=7,
            key=f"run_high_{selected_variable}",
        )

    save_threshold_values(
            selected_variable,
            #
            idle_threshold,
            run_threshold,
            #
            idle_low_band_range,
            run_low_band_range,
            #
            idle_medium_band_range,
            run_medium_band_range,
            #
            idle_high_band_range,
            run_high_band_range
    )

machine_state = "idle"  # Or get it from your DataFrame if it changes

folders = ["HTOL-09", "HTOL-10", "HTOL-11", "HTOL-12", "HTOL-13", "HTOL-14", "HTOL-15"]
paths = [os.path.join(data_directory, folder_path) for folder_path in folders]

severities = ['low', 'medium', 'high', "3-sigma"]

severities_color_map = {
    "low": "yellow",
    "medium": "orange",
    "high": "red",
    "3-sigma": "purple"
}
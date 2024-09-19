import os

data_directory = "../"

idle_threshold = 32
run_threshold = 32
machine_state = "idle"  # Or get it from your DataFrame if it changes

chiller_pressure_title = "ChlPrs"
folders = ["HTOL-09", "HTOL-10", "HTOL-11", "HTOL-12", "HTOL-13", "HTOL-14", "HTOL-15"]
paths = [os.path.join(data_directory, folder_path) for folder_path in folders]

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

severities = ['low', 'medium', 'high', "3-sigma"]

severities_color_map = {
    "low": "yellow",
    "medium": "orange",
    "high": "red",
    "3-sigma": "purple"
}
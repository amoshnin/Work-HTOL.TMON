import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from constants import folders, severities_color_map
import uuid
from visualization import visualise_time_series
from streamlit_plotly_events import plotly_events  # Import the component

def timeline(total_data, severity, selected_variable, bands):
    all_alerts = []
    alert_point_data = {}  # Initialize dictionary to store alert point datas

    for machine, data in total_data.items():
        alert_data = data['alert_data']

        for file_name, alert_info in alert_data.items():
            df = alert_info['df']
            grouped_alerts_indices = alert_info['grouped_alerts_indices']
            # Filter alerts based on severity
            filtered_alerts = grouped_alerts_indices[grouped_alerts_indices['severity'] == severity]

            # Merge to get the timestamps and sensor values
            merged_df = pd.merge(filtered_alerts, df, left_on='alert_index', right_index=True)

            # Add machine name and file name
            merged_df['machine'] = machine
            merged_df['file_name'] = file_name

            all_alerts.append(merged_df)

            for _, row in merged_df.iterrows():
                unique_key = str(row['Time']) + "_" + row['machine']  # Include file_name in the key
                alert_point_data[unique_key] = {  # Store the data associated with the alert point
                    'df': df,
                    'grouped_alerts_indices': grouped_alerts_indices,
                    'file_name': row['machine'] + "_" + file_name,  # Include machine name in the key
                }

    # Combine all alerts
    combined_df = pd.concat(all_alerts)

    # Sort by time
    combined_df = combined_df.sort_values('Time')

    alert_counts = combined_df.groupby('machine').size().to_dict()

    # Create Plotly figure
    fig = go.Figure()

    # Add trace for each alert
    for _, row in combined_df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['Time']],
            y=[f"{row['machine']} "], # (Count: {alert_counts[row['machine']]})
            text=[row[selected_variable]],  # Assuming 'ChlPrs' is the sensor value column
            mode='markers',
            marker=dict(
                color=severities_color_map.get(row['severity']),
                size=20
            ),
            hovertemplate='<b>Machine:</b> %{y}<br><b>Time:</b> %{x}<br><b>Sensor Value:</b> %{text}<br><b>Severity:</b> %{marker.color}<extra></extra>',
        ))

    # Update layout
    fig.update_layout(
        title='Timeline of Alerts',
        xaxis_title='Time',
        yaxis_title='Machine',
        showlegend=False,
        xaxis=dict(
            tickfont=dict(size=18, color='black')  # Update x-axis tick label font
        ),
        yaxis=dict(
            tickfont=dict(size=18, color='black')  # Update y-axis tick label font
        ),
        hovermode='closest',
        width=1700
    )

    # Use the plotly_events component
    selected_points = plotly_events(fig, key=f"timeline_{severity}")

    print(selected_points)

    if selected_points:
        for point in selected_points:
            key = point["x"] + "_" + point["y"].split(" ")[0]
            selected_point_data = alert_point_data[key]
            visualise_time_series(selected_point_data['df'], selected_variable, bands["idle_bands"], bands["run_bands"], selected_point_data['grouped_alerts_indices'], selected_point_data['file_name'])

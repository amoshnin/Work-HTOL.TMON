import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from constants import folders, severities_color_map, chiller_pressure_title


def timeline(total_data, severity):
    all_alerts = []

    for machine, data in total_data.items():
        alert_data = data['alert_data']

        for _, alert_info in alert_data.items():
            df = alert_info['df']
            grouped_alerts_indices = alert_info['grouped_alerts_indices']
            # Filter alerts based on severity
            filtered_alerts = grouped_alerts_indices[grouped_alerts_indices['severity'] == severity]

            # Merge to get the timestamps and sensor values
            merged_df = pd.merge(filtered_alerts, df, left_on='alert_index', right_index=True)

            # Add machine name
            merged_df['machine'] = machine

            all_alerts.append(merged_df)

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
            y=[f"{row['machine']} (Count: {alert_counts[row['machine']]})"],
            text=[row[chiller_pressure_title]],  # Assuming 'ChlPrs' is the sensor value column
            mode='markers',
            marker=dict(
                color=severities_color_map.get(row['severity']),
                size=20
            ),
             hovertemplate='<b>Machine:</b> %{y}<br><b>Time:</b> %{x}<br><b>Sensor Value:</b> %{text}<br><b>Severity:</b> %{marker.color}<extra></extra>'
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
        )
    )

    # Display the figure
    st.plotly_chart(fig)
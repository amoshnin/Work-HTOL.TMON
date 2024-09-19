import streamlit as st
from constants import severities, severities_color_map
import plotly.express as px

def list_machines_visible_title(severity):
    return f"list_machines_visible_{severity}"

def visualise_machine_stats(total_data, severity, sort_by):
    # Extract machine-wise alert counts specifically for the selected severity
    machine_alert_counts = {
        machine_id: data['alert_counts'][severity] for machine_id, data in total_data.items()
    }

    # Sort machines based on alert count for the chosen severity
    if sort_by == "No sorting":
        sorted_machines = machine_alert_counts.items()
    else:
        sorted_machines = sorted(
        machine_alert_counts.items(), key=lambda item: item[1], reverse=(sort_by == "Alert Count (Highest First)")
        )

    # Prepare data for plotting
    machines, counts = zip(*sorted_machines)

    # Create the bar plot using Plotly
    fig = px.bar(
        x=machines,
        y=counts,
        title=f"Machines Sorted by {severity.capitalize()} Severity",
        labels={"x": "Machine", "y": "Alert Count"},
        text=counts,
        color_discrete_sequence=[severities_color_map[severity]]  # Use color associated with the severity
    )

    fig.update_traces(textposition='outside')

    # Display the plot in Streamlit
    st.plotly_chart(fig)

def display_machines_stats(total_data):
    st.header(f"Machines Alerts Counts")
    for machine_id, data in total_data.items():
        alert_counts = data['alert_counts']

        st.subheader(machine_id)
        cols = st.columns(len(severities_color_map))

        for i, (sev, color) in enumerate(severities_color_map.items()):
                    cols[i].markdown(
                        f'<span style="color:{color}; font-size:30px; font-weight:semibold; padding-right: 5px">&#9679;</span> <span style="font-size:30px; font-weight:semibold;">{sev.capitalize()}</span>',
                        unsafe_allow_html=True
                    )

        # Display the corresponding counts for each severity with increased font size and semiboldness
        cols = st.columns(len(severities_color_map))
        for i, (sev, _) in enumerate(severities_color_map.items()):
            cols[i].markdown(
                f'<div style="font-size: 25px; font-weight:semibold;">{alert_counts.get(sev, 0)}</div>',
                unsafe_allow_html=True
            )

def summary_tab(total_data, alert_counts):
    col1, col2, col3, col4 = st.columns(4)

    for severity in severities:
        button_title = list_machines_visible_title(severity)
        if button_title not in st.session_state:
            st.session_state[button_title] = False

    with col1:
        severity = "low"
        sort_by_low = st.selectbox(f"Sort {severity} Severity by:", ["No sorting", "Alert Count (Highest First)", "Alert Count (Lowest First)"], key=f"sort_by_machine{severity}")
        st.metric(f"{severity} Severity Alerts", alert_counts[severity], delta_color="off", label_visibility="collapsed")
        st.markdown(f'<span style="color:yellow; font-size:20px">{"⬤"}</span>', unsafe_allow_html=True)
        button_key = list_machines_visible_title(severity)

        if st.button("Toggle Visibility of Machines Breakdown", key=f"button_{button_key}"):
            st.session_state[button_key] = not st.session_state[button_key]

        if st.session_state[button_key]:
            visualise_machine_stats(total_data, severity, sort_by_low)

    with col2:
        severity = "medium"
        sort_by_medium = st.selectbox(f"Sort {severity} Severity by:", ["No sorting", "Alert Count (Highest First)", "Alert Count (Lowest First)"], key=f"sort_by_machine{severity}")
        st.metric(f"{severity} Severity Alerts", alert_counts[severity], delta_color="off", label_visibility="collapsed")
        st.markdown(f'<span style="color:orange; font-size:20px">{"⬤"}</span>', unsafe_allow_html=True)
        button_key = list_machines_visible_title(severity)

        if st.button("Toggle Visibility of Machines Breakdown", key=f"button_{button_key}"):
            st.session_state[button_key] = not st.session_state[button_key]

        if st.session_state[button_key]:
            visualise_machine_stats(total_data, severity, sort_by_medium)

    with col3:
        severity = "high"
        sort_by_high = st.selectbox(f"Sort {severity} Severity by:", ["No sorting", "Alert Count (Highest First)", "Alert Count (Lowest First)"], key=f"sort_by_machine{severity}")
        st.metric(f"{severity} Severity Alerts", alert_counts[severity], delta_color="off", label_visibility="collapsed")
        st.markdown(f'<span style="color:red; font-size:20px">{"⬤"}</span>', unsafe_allow_html=True)
        button_key = list_machines_visible_title(severity)

        if st.button("Toggle Visibility of Machines Breakdown", key=f"button_{button_key}"):
            st.session_state[button_key] = not st.session_state[button_key]

        if st.session_state[button_key]:
            visualise_machine_stats(total_data, severity, sort_by_high)

    with col4:
        severity = "3-sigma"
        sort_by_sigma = st.selectbox(f"Sort {severity} Severity by:", ["No sorting", "Alert Count (Highest First)", "Alert Count (Lowest First)"], key=f"sort_by_machine{severity}")
        st.metric(f"{severity} Severity Alerts", alert_counts[severity], delta_color="off", label_visibility="collapsed")
        st.markdown(f'<span style="color:purple; font-size:20px">{"⬤"}</span>', unsafe_allow_html=True)
        button_key = list_machines_visible_title(severity)

        if st.button("Toggle Visibility of Machines Breakdown", key=f"button_{button_key}"):
            st.session_state[button_key] = not st.session_state[button_key]

        if st.session_state[button_key]:
            visualise_machine_stats(total_data, severity, sort_by_sigma)

    st.markdown("---")

    display_machines_stats(total_data)
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
import os
import altair as alt

from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats

import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from typing import Dict, List, Union, Tuple

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statistics
from sklearn.metrics import classification_report, precision_recall_fscore_support

main_data_path = "../../../outlier_tolerance=5_grouping_time_window=200_anomaly_threshold=6_start_date=2022-01-01_end_date=2026-01-01"
main_model_path = "production_models_solo"

class AlertPredictor:
    def __init__(self, model_type='xgboost'):
        """
        Initializes the AlertPredictor with the specified model type ('xgboost' or 'randomforest').
        """
        self.model_type = model_type
        self.models = {}
        self.scalers = {}
        self.alert_types = ['LOW', 'MEDIUM', 'HIGH', ] # 'SIGMA']
        self.features = ['ChlPrs',
                        #  'hour',
                        #  'day_of_week',
                        #  'month',
                        #  'is_weekend',
                         'rolling_mean', 'rolling_std'] + [f'time_since_{at}' for at in self.alert_types]

    def load_and_preprocess_data(self, folder):
        """
        Loads and preprocesses data from CSV files in the specified folder.
        """
        dfs = []
        for i in range(9, 16):
            file_name = f"HTOL-{i:02d}_alerts.csv"
            df = pd.read_csv(os.path.join(folder, file_name))
            df['machine_id'] = f'HTOL-{i:02d}'
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df['Time'] = pd.to_datetime(combined_df['Time'])
        combined_df = combined_df.sort_values(['machine_id', 'Time'])

        return combined_df

    def engineer_features(self, df):
        """
        Engineers features from the preprocessed data.
        """
        df['hour'] = df['Time'].dt.hour
        df['day_of_week'] = df['Time'].dt.dayofweek
        df['month'] = df['Time'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # Calculate rolling statistics
        df['rolling_mean'] = df.groupby('machine_id')['ChlPrs'].rolling(window=24, min_periods=1).mean().reset_index(0, drop=True)
        df['rolling_std'] = df.groupby('machine_id')['ChlPrs'].rolling(window=24, min_periods=1).std().reset_index(0, drop=True)

        # Calculate time since last alert for each type
        for alert_type in self.alert_types:
            df[f'time_since_{alert_type}'] = df.groupby('machine_id').apply(
                lambda x: x['Time'] - x[x['ALERT'] == alert_type]['Time'].shift(1)).reset_index(level=0, drop=True)
            df[f'time_since_{alert_type}'] = df[f'time_since_{alert_type}'].dt.total_seconds() / 3600  # Convert to hours

        return df

    def prepare_data_for_classification(self, df, target_alert_type, prediction_window):
        """
        Prepares the data for training the classification model.
        """
        df['target'] = df.groupby('machine_id').apply(
            lambda x: (x['ALERT'] == target_alert_type).rolling(window=prediction_window).max().shift(-prediction_window + 1)).reset_index(level=0,
                                                                                                                                           drop=True)

        X = df[self.features]
        y = df['target'].fillna(0)  # Fill NaN with 0 (no alert)

        return X, y

    def train_and_evaluate_classifier(self, X, y, test_size=0.2):
        """
        Trains and evaluates the classification model.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if self.model_type == 'xgboost':
            # XGBoost configuration for imbalanced classification
            model = XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_child_weight=1,
                gamma=0,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),  # Handle class imbalance
                random_state=42,
                eval_metric='logloss',
                early_stopping_rounds=10,
            )
            model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=0)
        elif self.model_type == 'randomforest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
        else:
            raise ValueError("Invalid model_type. Choose 'xgboost' or 'randomforest'.")

        y_pred = model.predict(X_test_scaled)
        print(classification_report(y_test, y_pred))

        return model, scaler

    def train(self, folder, prediction_window=7):
        """
        Trains the models for each alert type.
        """
        df = self.load_and_preprocess_data(folder)
        df = self.engineer_features(df)

        for alert_type in self.alert_types:
            print(f"\nTraining model for {alert_type} alerts:")
            X, y = self.prepare_data_for_classification(df, alert_type, prediction_window)
            model, scaler = self.train_and_evaluate_classifier(X, y)
            self.models[alert_type] = model
            self.scalers[alert_type] = scaler

    def predict(self, new_data):
        """
        Makes predictions on new data.
        """
        predictions = {}
        for alert_type in self.alert_types:
            X_new = new_data[self.features]
            X_new_scaled = self.scalers[alert_type].transform(X_new)
            alert_probability = self.models[alert_type].predict_proba(X_new_scaled)[0, 1]
            predictions[alert_type] = alert_probability
        return predictions

    def visualize_alerts(self, df, target_alert_type, prediction_window, probability_threshold=0.7):
        """
        Visualizes actual alerts and high-risk periods.
        """
        X = df[self.features]
        X_scaled = self.scalers[target_alert_type].transform(X)

        df['alert_probability'] = self.models[target_alert_type].predict_proba(X_scaled)[:, 1]
        df['high_risk'] = df['alert_probability'] > probability_threshold

        plt.figure(figsize=(20, 15))
        machines = df['machine_id'].unique()
        n_machines = len(machines)

        for i, machine_id in enumerate(machines):
            machine_df = df[df['machine_id'] == machine_id]

            # Plot actual alerts
            alerts = machine_df[machine_df['ALERT'] == target_alert_type]
            plt.scatter(alerts['Time'], [i - 0.2] * len(alerts), marker='o', s=100,
                        label=f'Actual {target_alert_type} Alert' if i == 0 else "")

            # Plot high-risk periods
            high_risk_periods = machine_df[machine_df['high_risk']]
            plt.scatter(high_risk_periods['Time'], [i + 0.2] * len(high_risk_periods), marker='x', s=100,
                        label=f'High Risk Period ({target_alert_type})' if i == 0 else "")

            plt.text(df['Time'].min(), i, machine_id, va='center', ha='right', fontweight='bold')

        plt.yticks(range(n_machines), machines)
        plt.xlabel('Date')
        plt.ylabel('Machine ID')
        plt.title(f'Actual Alerts vs High Risk Periods for {target_alert_type} Alerts')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

class ProductionAlertPredictor:
    def __init__(self):
        """
        Initialize the production predictor that handles both XGBoost and Random Forest models.
        """
        self.models = {}
        self.scalers = {}
        self.alert_types = ['LOW', 'MEDIUM', 'HIGH']
        self.features = [
            'ChlPrs',
            # 'hour',
            # 'day_of_week',
            # 'month',
            # 'is_weekend',
            'rolling_mean',
            'rolling_std'
        ] + [f'time_since_{at}' for at in self.alert_types]

    def save_models(self, xgb_predictor: AlertPredictor, rf_predictor: AlertPredictor,
                   save_dir: str) -> None:
        """
        Save trained models and scalers to disk.

        Args:
            xgb_predictor: Trained XGBoost AlertPredictor instance
            rf_predictor: Trained Random Forest AlertPredictor instance
            save_dir: Directory to save the models
        """
        os.makedirs(save_dir, exist_ok=True)

        # Save metadata
        metadata = {
            'features': self.features,
            'alert_types': self.alert_types,
            'timestamp': datetime.now().isoformat(),
        }
        joblib.dump(metadata, os.path.join(save_dir, 'metadata.joblib'))

        # Save models and scalers
        for alert_type in self.alert_types:
            # Save XGBoost models and scalers
            joblib.dump(
                xgb_predictor.models[alert_type],
                os.path.join(save_dir, f'xgboost_{alert_type.lower()}_model.joblib')
            )
            joblib.dump(
                xgb_predictor.scalers[alert_type],
                os.path.join(save_dir, f'xgboost_{alert_type.lower()}_scaler.joblib')
            )

            # Save Random Forest models and scalers
            joblib.dump(
                rf_predictor.models[alert_type],
                os.path.join(save_dir, f'randomforest_{alert_type.lower()}_model.joblib')
            )
            joblib.dump(
                rf_predictor.scalers[alert_type],
                os.path.join(save_dir, f'randomforest_{alert_type.lower()}_scaler.joblib')
            )

    def load_models(self, load_dir: str) -> None:
        """
        Load saved models and scalers from disk.

        Args:
            load_dir: Directory containing the saved models
        """
        # Load metadata
        metadata = joblib.load(os.path.join(load_dir, 'metadata.joblib'))
        self.features = metadata['features']
        self.alert_types = metadata['alert_types']

        # Initialize nested dictionaries for models and scalers
        self.models = {'xgboost': {}, 'randomforest': {}}
        self.scalers = {'xgboost': {}, 'randomforest': {}}

        # Load models and scalers
        for alert_type in self.alert_types:
            # Load XGBoost
            self.models['xgboost'][alert_type] = joblib.load(
                os.path.join(load_dir, f'xgboost_{alert_type.lower()}_model.joblib')
            )
            self.scalers['xgboost'][alert_type] = joblib.load(
                os.path.join(load_dir, f'xgboost_{alert_type.lower()}_scaler.joblib')
            )

            # Load Random Forest
            self.models['randomforest'][alert_type] = joblib.load(
                os.path.join(load_dir, f'randomforest_{alert_type.lower()}_model.joblib')
            )
            self.scalers['randomforest'][alert_type] = joblib.load(
                os.path.join(load_dir, f'randomforest_{alert_type.lower()}_scaler.joblib')
            )

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for prediction.

        Args:
            data: DataFrame containing at minimum 'Time', 'ChlPrs', and 'machine_id' columns

        Returns:
            DataFrame with engineered features
        """
        df = data.copy()

        # Time-based features
        df['Time'] = pd.to_datetime(df['Time'])
        df['hour'] = df['Time'].dt.hour
        df['day_of_week'] = df['Time'].dt.dayofweek
        df['month'] = df['Time'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # Rolling statistics
        df['rolling_mean'] = df.groupby('machine_id')['ChlPrs'].rolling(
            window=24, min_periods=1).mean().reset_index(0, drop=True)
        df['rolling_std'] = df.groupby('machine_id')['ChlPrs'].rolling(
            window=24, min_periods=1).std().reset_index(0, drop=True)

        # Calculate time since last alert for each type
        for alert_type in self.alert_types:
            if 'ALERT' in df.columns:
                # Create a series of alert timestamps for the current alert type
                alert_times = df[df['ALERT'] == alert_type].copy()

                # For each machine, calculate the time difference
                time_since = pd.DataFrame()

                for machine in df['machine_id'].unique():
                    machine_data = df[df['machine_id'] == machine].copy()
                    machine_alerts = alert_times[alert_times['machine_id'] == machine]['Time']

                    if not machine_alerts.empty:
                        # Find the most recent alert before each timestamp
                        machine_data['last_alert'] = pd.NA
                        for idx, row in machine_data.iterrows():
                            prev_alerts = machine_alerts[machine_alerts < row['Time']]
                            if not prev_alerts.empty:
                                machine_data.at[idx, 'last_alert'] = prev_alerts.iloc[-1]

                        # Calculate time difference
                        machine_data[f'time_since_{alert_type}'] = (
                            machine_data['Time'] - pd.to_datetime(machine_data['last_alert'])
                        ).dt.total_seconds() / 3600  # Convert to hours

                        # Fill NA with large value (e.g., one week in hours)
                        machine_data[f'time_since_{alert_type}'].fillna(168, inplace=True)
                    else:
                        machine_data[f'time_since_{alert_type}'] = 168  # One week in hours

                    time_since = pd.concat([time_since, machine_data[[f'time_since_{alert_type}']]])

                # Assign the calculated values back to the main DataFrame
                df[f'time_since_{alert_type}'] = time_since[f'time_since_{alert_type}']
            else:
                # For new data without alert history, use a large value
                df[f'time_since_{alert_type}'] = 168  # One week in hours

        return df[self.features]

    def predict(self, data: pd.DataFrame, model_type: str = 'xgboost') -> Dict[str, Dict[str, float]]:
        """
        Make predictions using the loaded models.

        Args:
            data: DataFrame containing required features
            model_type: 'xgboost' or 'randomforest'

        Returns:
            Dictionary containing predictions for each machine and alert type
        """
        if model_type not in ['xgboost', 'randomforest']:
            raise ValueError("model_type must be 'xgboost' or 'randomforest'")

        # Prepare features
        X = self.prepare_features(data)

        # Make predictions for each machine and alert type
        predictions = {}
        for machine_id in data['machine_id'].unique():
            machine_data = X[data['machine_id'] == machine_id]
            machine_predictions = {}

            for alert_type in self.alert_types:
                # Scale the features
                X_scaled = self.scalers[model_type][alert_type].transform(machine_data)

                # Get prediction probabilities
                probs = self.models[model_type][alert_type].predict_proba(X_scaled)[:, 1]

                # Store the average probability for this alert type
                machine_predictions[alert_type] = float(probs.mean())

            predictions[machine_id] = machine_predictions

        return predictions

def save_trained_models(xgb_predictor: AlertPredictor, rf_predictor: AlertPredictor,
                       save_dir: str) -> None:
    """
    Convenience function to save trained models.
    """
    production_predictor = ProductionAlertPredictor()
    production_predictor.save_models(xgb_predictor, rf_predictor, save_dir)
    print(f"Models saved successfully to {save_dir}")

def load_production_predictor(load_dir: str) -> ProductionAlertPredictor:
    """
    Convenience function to load saved models.
    """
    production_predictor = ProductionAlertPredictor()
    production_predictor.load_models(load_dir)
    return production_predictor

def calculate_advanced_metrics(df: pd.DataFrame, predictions_df: pd.DataFrame, window_size: timedelta = timedelta(hours=24)) -> dict:
    """Calculate advanced metrics for machine performance analysis"""
    metrics = {}

    # Pressure Statistics
    metrics['pressure_stats'] = {
        'mean': df['ChlPrs'].mean(),
        'std': df['ChlPrs'].std(),
        'min': df['ChlPrs'].min(),
        'max': df['ChlPrs'].max(),
        'median': df['ChlPrs'].median(),
        'skewness': stats.skew(df['ChlPrs']),
        'kurtosis': stats.kurtosis(df['ChlPrs'])
    }

    # Alert Statistics
    if 'ALERT' in df.columns:
        metrics['alert_counts'] = df['ALERT'].value_counts().to_dict()
        metrics['alert_intervals'] = {}
        for alert_type in ['LOW', 'MEDIUM', 'HIGH']:
            alert_times = df[df['ALERT'] == alert_type]['Time']
            if len(alert_times) > 1:
                intervals = np.diff(alert_times) / np.timedelta64(1, 'h')
                metrics['alert_intervals'][alert_type] = {
                    'mean': intervals.mean(),
                    'std': intervals.std(),
                    'min': intervals.min(),
                    'max': intervals.max()
                }

    # Prediction Performance
    if not predictions_df.empty:
        metrics['prediction_stats'] = {
            'total_predictions': len(predictions_df),
            'unique_dates': predictions_df['Time'].nunique(),
            'alert_type_distribution': predictions_df['alert_type'].value_counts().to_dict()
        }

        # Calculate prediction accuracy within time windows
        true_positives = 0
        false_positives = 0

        for _, pred in predictions_df.iterrows():
            window_start = pred['Time']
            window_end = window_start + window_size
            actual_alerts = df[(df['Time'] >= window_start) &
                             (df['Time'] <= window_end) &
                             (df['ALERT'] == pred['alert_type'])]

            if len(actual_alerts) > 0:
                true_positives += 1
            else:
                false_positives += 1

        metrics['prediction_performance'] = {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'precision': true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        }

    return metrics

def generate_advanced_visualizations(df: pd.DataFrame, predictions_df: pd.DataFrame, metrics: dict) -> dict:
    """Generate additional visualizations for the dashboard"""
    figs = {}

    def update_mean_line(fig):
        """Updates the mean line after the figure is rendered."""
        max_y = fig.data[0].y.max()
        fig.data[1].y = [0, max_y]

    # 1. Pressure Distribution Plot
    pressure_hist = go.Figure()

    # Create histogram trace
    hist_trace = go.Histogram(
        x=df['ChlPrs'],
        nbinsx=50,
        name='Pressure Distribution',
        marker_color='blue',
        opacity=0.7
    )
    pressure_hist.add_trace(hist_trace)

    # Calculate the y-max value manually
    hist_data = np.histogram(df['ChlPrs'], bins=50)
    y_max = max(hist_data[0])

    # Add mean line with calculated y-max
    pressure_hist.add_trace(go.Scatter(
        x=[metrics['pressure_stats']['mean']] * 2,
        y=[0, y_max],
        mode='lines',
        name='Mean',
        line=dict(color='red', dash='dash')
    ))

    pressure_hist.update_layout(
        title='Pressure Distribution',
        xaxis_title='Pressure (ChlPrs)',
        yaxis_title='Count',
        showlegend=True
    )
    figs['pressure_distribution'] = pressure_hist

    # 2. Alert Pattern Analysis
    if 'ALERT' in df.columns:
        alert_patterns = go.Figure()
        for alert_type in ['LOW', 'MEDIUM', 'HIGH']:
            alert_times = df[df['ALERT'] == alert_type]['Time']
            alert_patterns.add_trace(go.Scatter(
                x=alert_times,
                y=[alert_type] * len(alert_times),
                mode='markers',
                name=f'{alert_type} Alerts',
                marker=dict(
                    size=10,
                    symbol='star'
                )
            ))
        alert_patterns.update_layout(
            title='Alert Pattern Timeline',
            xaxis_title='Time',
            yaxis_title='Alert Type',
            showlegend=True
        )
        figs['alert_patterns'] = alert_patterns

    # 3. Prediction Performance Over Time
    if not predictions_df.empty:
        pred_performance = go.Figure()
        for alert_type in ['LOW', 'MEDIUM', 'HIGH']:
            type_preds = predictions_df[predictions_df['alert_type'] == alert_type]
            pred_performance.add_trace(go.Scatter(
                x=type_preds['Time'],
                y=type_preds['probability'],
                mode='markers+lines',
                name=f'{alert_type} Predictions',
                marker=dict(size=8)
            ))
        pred_performance.update_layout(
            title='Prediction Probabilities Over Time',
            xaxis_title='Time',
            yaxis_title='Prediction Probability',
            showlegend=True
        )
        figs['prediction_performance'] = pred_performance

    return figs

def generate_machine_dashboard(
    machine_id: str,
    df: pd.DataFrame,
    date_range: tuple,
    alert_threshold: float,
    predictor
) -> tuple:
    """Generate comprehensive machine dashboard"""

    if machine_id not in st.session_state['processed_machines'] or cache_changed:
        with st.spinner(f"Processing data for {machine_id}..."):
                # Generate visualization and cache it
                main_fig, base_metrics, predictions_df = generate_machine_visualization(
                    machine_id,
                    data_dict[machine_id],
                    date_range,
                    alert_threshold,
                    predictor
                )

                # Store in session state
                st.session_state['processed_machines'].add(machine_id)
    else:
        # Retrieve from cache
        main_fig, base_metrics, predictions_df = generate_machine_visualization(
                machine_id,
                data_dict[machine_id],
                date_range,
                alert_threshold,
                predictor
            )

    # Calculate advanced metrics
    advanced_metrics = calculate_advanced_metrics(df, predictions_df)

    # Generate additional visualizations
    additional_figs = generate_advanced_visualizations(df, predictions_df, advanced_metrics)

    return main_fig, base_metrics, advanced_metrics, additional_figs, predictions_df

# Page config
st.set_page_config(page_title="HTOL Machine Monitor", layout="wide")

# Sidebar controls
st.sidebar.title("Dashboard Controls")

# Load data function
@st.cache_data
def load_data(data_path: str) -> Dict[str, pd.DataFrame]:
    """Load all CSV files from the specified directory"""
    progress_text = "Loading data files..."
    progress_bar = st.progress(0)
    st.info(progress_text)

    all_files = glob.glob(os.path.join(data_path, "HTOL-*_alerts.csv"))
    total_files = len(all_files)
    data_dict = {}

    for idx, file in enumerate(all_files):
        machine_id = file.split('HTOL-')[1].split('_')[0]
        df = pd.read_csv(file)
        df['Time'] = pd.to_datetime(df['Time'])
        data_dict[f"HTOL-{machine_id}"] = df

        # Update progress
        progress = (idx + 1) / total_files
        progress_bar.progress(progress)
        st.info(f"{progress_text} ({idx + 1}/{total_files} files)")

    progress_bar.empty()
    st.success("✅ Data loading complete!")
    return data_dict

# Load models
@st.cache_resource
def load_models(model_path: str):
    """Load the prediction models"""
    with st.spinner("Loading prediction models..."):
        predictor = load_production_predictor(model_path)
        st.success("✅ Models loaded successfully!")
        return predictor

def get_predictions_for_period(data: pd.DataFrame, predictor, window_size: timedelta = timedelta(hours=24), threshold: Dict[str, float] = None) -> pd.DataFrame:
    """
    Get predictions for chunks of data throughout the entire period with individual model probabilities
    """
    if threshold is None:
        threshold = {alert_type: 0.7 for alert_type in ['LOW', 'MEDIUM', 'HIGH', 'SIGMA']}

    if 'machine_id' not in data.columns:
        data['machine_id'] = data['file_name'].str.extract(r'(HTOL-\d+)')

    all_predictions = []
    start_time = data['Time'].min()
    end_time = data['Time'].max()
    current_time = start_time

    while current_time <= end_time:
        chunk_end = current_time + window_size
        mask = (data['Time'] >= current_time) & (data['Time'] < chunk_end)
        chunk_data = data[mask].copy()

        if not chunk_data.empty:
            xgb_preds = predictor.predict(chunk_data, model_type='xgboost')
            rf_preds = predictor.predict(chunk_data, model_type='randomforest')

            for machine_id in xgb_preds:
                for alert_type in xgb_preds[machine_id]:
                    xgb_prob = xgb_preds[machine_id][alert_type]
                    rf_prob = rf_preds[machine_id][alert_type]
                    avg_prob = rf_prob if alert_type == "LOW" else xgb_prob if (alert_type == "HIGH" or alert_type == "MEDIUM") else (rf_prob + xgb_prob) / 2

                    # Use the specific threshold for this alert type
                    if alert_type in threshold and avg_prob >= threshold[alert_type]:
                        all_predictions.append({
                            'Time': current_time,
                            'machine_id': machine_id,
                            'alert_type': alert_type,
                            'probability': avg_prob,
                            'xgb_probability': xgb_prob,
                            'rf_probability': rf_prob
                        })

        current_time = chunk_end

    if all_predictions:
        return pd.DataFrame(all_predictions)
    return pd.DataFrame(columns=['Time', 'machine_id', 'alert_type', 'probability', 'xgb_probability', 'rf_probability'])

@st.cache_data
def generate_machine_visualization(
    machine_id: str,
    df: pd.DataFrame,
    date_range: tuple,
    alert_threshold: float,
    _predictor
) -> tuple:
    """
    Generate enhanced visualization with clearer markers and model probabilities
    """
    mask = (df['Time'].dt.date >= date_range[0]) & (df['Time'].dt.date <= date_range[1])
    df_filtered = df[mask]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add ChlPrs line
    fig.add_trace(
        go.Scatter(
            x=df_filtered['Time'],
            y=df_filtered['ChlPrs'],
            name="ChlPrs",
            line=dict(color='blue', width=1)
        ),
        secondary_y=False
    )

    # Enhanced actual alerts with outline
    alert_colors = {
        'LOW': {'color': 'yellow', 'outline': 'black'},
        'MEDIUM': {'color': 'orange', 'outline': 'black'},
        'HIGH': {'color': 'red', 'outline': 'black'}
    }

    for alert_type, colors in alert_colors.items():
        alert_points = df_filtered[df_filtered['ALERT'] == alert_type]
        if not alert_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=alert_points['Time'],
                    y=alert_points['ChlPrs'],
                    mode='markers',
                    name=f'Actual {alert_type} Alert',
                    marker=dict(
                        color=colors['color'],
                        size=15,
                        symbol='star',
                        line=dict(
                            color=colors['outline'],
                            width=2
                        )
                    ),
                    hovertemplate=(
                        f"<b>Actual {alert_type} Alert</b><br>" +
                        "Time: %{x}<br>" +
                        "Pressure: %{y:.2f}<br>" +
                        "<extra></extra>"
                    )
                ),
                secondary_y=False
            )

    # Get predictions with model probabilities
    predictions_df = get_predictions_for_period(
        df_filtered,
        _predictor,
        window_size=timedelta(hours=24),
        threshold=alert_threshold
    )

    # Enhanced predicted alerts with outline and probability information
    if not predictions_df.empty:
        for alert_type, colors in alert_colors.items():
            alert_predictions = predictions_df[predictions_df['alert_type'] == alert_type]
            if not alert_predictions.empty:
                pred_pressure = pd.merge_asof(
                    alert_predictions,
                    df_filtered[['Time', 'ChlPrs']],
                    on='Time',
                    direction='nearest'
                )

                fig.add_trace(
                    go.Scatter(
                        x=pred_pressure['Time'],
                        y=pred_pressure['ChlPrs'],
                        mode='markers',
                        name=f'Predicted {alert_type} Alert',
                        marker=dict(
                            color=colors['color'],
                            size=18,
                            symbol='diamond',
                            opacity=0.9,
                            line=dict(
                                color=colors['outline'],
                                width=2
                            )
                        ),
                        hovertemplate=(
                            f"<b>Predicted {alert_type} Alert</b><br>" +
                            "Time: %{x}<br>" +
                            "Pressure: %{y:.2f}<br>" +
                            "XGBoost Probability: %{customdata[0]:.3f}<br>" +
                            "Random Forest Probability: %{customdata[1]:.3f}<br>" +
                            "Average Probability: %{customdata[2]:.3f}<br>" +
                            "<extra></extra>"
                        ),
                        customdata=np.column_stack((
                            pred_pressure['xgb_probability'],
                            pred_pressure['rf_probability'],
                            pred_pressure['probability']
                        ))
                    ),
                    secondary_y=False
                )

    # Update layout with enhanced legend and hover mode
    fig.update_layout(
        title=f"{machine_id} Pressure and Alerts Over Time",
        xaxis_title="Time",
        yaxis_title="Pressure (ChlPrs)",
        height=600,
        showlegend=True,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="Black",
            borderwidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # Add grid for better readability
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    # Calculate metrics
    metrics = {
        'average_pressure': df_filtered['ChlPrs'].mean(),
        'total_alerts': df_filtered['ALERT'].notna().sum(),
        'predicted_alerts': len(predictions_df[predictions_df['machine_id'] == machine_id]),
        'alert_breakdown': pd.DataFrame(),  # Initialize empty DataFrame for alert breakdowns
        'pred_breakdown': pd.DataFrame()
    }

    # Create alert breakdown DataFrames with consistent indices
    alert_types = ['LOW', 'MEDIUM', 'HIGH']

    # Actual alerts breakdown
    if df_filtered['ALERT'].notna().sum() > 0:
        actual_counts = df_filtered['ALERT'].value_counts()
        metrics['alert_breakdown'] = pd.DataFrame(
            index=alert_types,
            data={'count': [actual_counts.get(alert_type, 0) for alert_type in alert_types]}
        )

    # Predicted alerts breakdown
    if not predictions_df.empty:
        pred_counts = predictions_df[predictions_df['machine_id'] == machine_id]['alert_type'].value_counts()
        metrics['pred_breakdown'] = pd.DataFrame(
            index=alert_types,
            data={'count': [pred_counts.get(alert_type, 0) for alert_type in alert_types]}
        )

    # Find the maximum count across both actual and predicted alerts
    max_count = max(
        metrics['alert_breakdown']['count'].max() if not metrics['alert_breakdown'].empty else 0,
        metrics['pred_breakdown']['count'].max() if not metrics['pred_breakdown'].empty else 0
    )

    # Add max_count to metrics for use in plotting
    metrics['max_count'] = max_count if max_count > 0 else 1  # Use 1 as minimum to avoid empty charts

    return fig, metrics, predictions_df

# Main app
st.title("HTOL Machine Monitoring Dashboard")

# Initialize session state for loading status
if 'loading_state' not in st.session_state:
    st.session_state['loading_state'] = {
        'data_loaded': False,
        'models_loaded': False,
        'processing_complete': False
    }

# Load data
try:
    data_path = main_data_path # st.sidebar.text_input("Data Directory Path", main_data_path)
    if not st.session_state['loading_state']['data_loaded']:
        with st.spinner("📂 Loading HTOL machine data..."):
            data_dict = load_data(data_path)
            st.session_state['loading_state']['data_loaded'] = True
    else:
        data_dict = load_data(data_path)
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Load models
try:
    model_path = main_model_path # st.sidebar.text_input("Model Directory Path", main_model_path)
    if not st.session_state['loading_state']['models_loaded']:
        with st.spinner("🤖 Loading prediction models..."):
            predictor = load_models(model_path)
            st.session_state['loading_state']['models_loaded'] = True
    else:
        predictor = load_models(model_path)
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# Dashboard controls
selected_machines = st.sidebar.multiselect(
    "Select Machines",
    options=list(data_dict.keys()),
    default=sorted(list(data_dict.keys()))
)

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(
        min(df['Time'].min() for df in data_dict.values()),
        max(df['Time'].max() for df in data_dict.values())
    )
)

# Replace single alert threshold with separate threshold for each alert type
st.sidebar.subheader("Alert Prediction Thresholds")
alert_threshold = {
    'LOW': st.sidebar.slider(
        "LOW Alert Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.01
    ),
    'MEDIUM': st.sidebar.slider(
        "MEDIUM Alert Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.01
    ),
    'HIGH': st.sidebar.slider(
        "HIGH Alert Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.01
    ),
    'SIGMA': st.sidebar.slider(
        "SIGMA Alert Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.95,
        step=0.01
    )
}

if selected_machines:
    progress_text = "Generating visualizations..."
    progress_bar = st.progress(0)
    st.info(progress_text)

    # Track which machines have been processed
    if 'processed_machines' not in st.session_state:
        st.session_state['processed_machines'] = set()

    # Track visualization cache parameters
    if 'cache_params' not in st.session_state:
        st.session_state['cache_params'] = {
            'date_range': date_range,
            'alert_threshold': alert_threshold
        }

    # Check if cache parameters have changed
    cache_changed = (
        st.session_state['cache_params']['date_range'] != date_range or
        st.session_state['cache_params']['alert_threshold'] != alert_threshold
    )

    if cache_changed:
        # Clear cache if parameters changed
        st.session_state['processed_machines'].clear()
        st.session_state['cache_params'] = {
            'date_range': date_range,
            'alert_threshold': alert_threshold
        }

    for idx, machine_id in enumerate(selected_machines):
            st.header(f"{machine_id} Monitoring")

            # Generate comprehensive dashboard
            main_fig, base_metrics, advanced_metrics, additional_figs, predictions_df = generate_machine_dashboard(
                machine_id,
                data_dict[machine_id],
                date_range,
                alert_threshold,
                predictor
            )

            # Main visualization
            st.plotly_chart(main_fig, use_container_width=True)

            # Key Performance Indicators
            st.subheader("Key Performance Indicators")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Average Pressure",
                    f"{advanced_metrics['pressure_stats']['mean']:.2f}",
                    f"{advanced_metrics['pressure_stats']['std']:.2f} σ"
                )

            with col2:
                st.metric(
                    "Total Alerts",
                    base_metrics['total_alerts'],
                    f"Predicted: {base_metrics['predicted_alerts']}"
                )

            with col3:
                if 'prediction_performance' in advanced_metrics:
                    st.metric(
                        "Prediction Precision",
                        f"{advanced_metrics['prediction_performance']['precision']:.2%}"
                    )

            with col4:
                st.metric(
                    "Pressure Range",
                    f"{advanced_metrics['pressure_stats']['max'] - advanced_metrics['pressure_stats']['min']:.2f}"
                )

            # Additional Visualizations
            st.subheader("Advanced Analytics")

            # Pressure Distribution
            st.plotly_chart(additional_figs['pressure_distribution'], use_container_width=True)

            # Alert Patterns and Predictions
            col1, col2 = st.columns(2)

            with col1:
                if 'alert_patterns' in additional_figs:
                    st.plotly_chart(additional_figs['alert_patterns'])

            with col2:
                if 'prediction_performance' in additional_figs:
                    st.plotly_chart(additional_figs['prediction_performance'])

            # Detailed Statistics
            with st.expander("Detailed Statistics"):
                st.json(advanced_metrics)

            # Alert Analysis
            if base_metrics['alert_breakdown'] is not None or base_metrics['pred_breakdown'] is not None:
                st.subheader("Alert Breakdown - Actual vs Predicted")

                # Convert both breakdowns to DataFrames
                actual_df = pd.DataFrame()
                pred_df = pd.DataFrame()

                if base_metrics['alert_breakdown'] is not None:
                    actual_df = pd.DataFrame(base_metrics['alert_breakdown'], columns=['count']).reset_index()
                    actual_df.columns = ['alert_type', 'count']
                    actual_df['category'] = 'Actual'

                if base_metrics['pred_breakdown'] is not None:
                    pred_df = pd.DataFrame(base_metrics['pred_breakdown'], columns=['count']).reset_index()
                    pred_df.columns = ['alert_type', 'count']
                    pred_df['category'] = 'Predicted'

                # Combine the dataframes
                combined_df = pd.concat([actual_df, pred_df], ignore_index=True)

                if not combined_df.empty:
                    # Create the combined chart using Altair
                    chart = alt.Chart(combined_df).mark_bar().encode(
                        x=alt.X('alert_type:N', title='Alert Type'),
                        y=alt.Y('count:Q', title='Count'),
                        color=alt.Color('category:N',
                                    scale=alt.Scale(domain=['Actual', 'Predicted'],
                                                    range=['#1f77b4', '#ff7f0e']),
                                    title='Category'),
                        xOffset='category:N'  # This creates the grouped bars
                    ).properties(
                        height=400,
                        width=600
                    )

                    # Display the chart
                    st.altair_chart(chart, use_container_width=True)

            st.markdown("---")
            # Update progress
            progress = (idx + 1) / len(selected_machines)
            progress_bar.progress(progress)
            st.info(f"{progress_text} ({idx + 1}/{len(selected_machines)} machines)")

    progress_bar.empty()
    st.success("✅ Dashboard generation complete!")
    st.session_state['loading_state']['processing_complete'] = True

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### Dashboard Information
- Data is loaded from CSV files in the specified directory
- Predictions are made using ensemble of Random Forest and XGBoost models
- Actual alerts are shown as stars
- Predicted alerts are shown as diamonds
- Alert colors: Yellow (LOW), Orange (MEDIUM), Red (HIGH)
""")

# Display loading status in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Loading Status")
status_emoji = lambda x: "✅" if x else "⏳"
st.sidebar.markdown(f"Data Loading: {status_emoji(st.session_state['loading_state']['data_loaded'])}")
st.sidebar.markdown(f"Models Loading: {status_emoji(st.session_state['loading_state']['models_loaded'])}")
st.sidebar.markdown(f"Processing: {status_emoji(st.session_state['loading_state']['processing_complete'])}")
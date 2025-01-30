"""Streamlit dashboard for visualizing experiment results and feedback."""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from .experiment import Experiment
from .feedback import FeedbackManager

def load_experiment(experiment_name: str) -> Optional[Experiment]:
    """Load an experiment by name."""
    try:
        experiment = Experiment(
            name=experiment_name,
            description="",  # Will be loaded from file
            metrics=[]  # Will be loaded from file
        )
        return experiment
    except Exception as e:
        st.error(f"Failed to load experiment: {e}")
        return None

def load_feedback() -> Optional[FeedbackManager]:
    """Load feedback data."""
    try:
        return FeedbackManager()
    except Exception as e:
        st.error(f"Failed to load feedback data: {e}")
        return None

def plot_metric_over_time(df: pd.DataFrame, metric: str, variant_col: str = "variant_name"):
    """Plot metric values over time with trend lines."""
    fig = px.scatter(
        df,
        x="timestamp",
        y=metric,
        color=variant_col,
        trendline="lowess",
        title=f"{metric} Over Time by Variant"
    )
    st.plotly_chart(fig)

def plot_metric_distribution(df: pd.DataFrame, metric: str, variant_col: str = "variant_name"):
    """Plot distribution of metric values by variant."""
    fig = px.box(
        df,
        x=variant_col,
        y=metric,
        title=f"Distribution of {metric} by Variant"
    )
    st.plotly_chart(fig)

def plot_feedback_heatmap(feedback_manager: FeedbackManager):
    """Plot heatmap of feedback criteria correlations."""
    if not feedback_manager.responses:
        st.warning("No feedback data available")
        return
        
    # Create DataFrame of ratings
    data = []
    for response in feedback_manager.responses:
        row = {"article_id": response.article_id}
        row.update(response.ratings)
        data.append(row)
    
    df = pd.DataFrame(data)
    if len(df.columns) <= 1:
        st.warning("Insufficient feedback data for correlation analysis")
        return
        
    corr = df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr,
        x=corr.columns,
        y=corr.columns,
        colorscale="RdBu"
    ))
    fig.update_layout(title="Feedback Criteria Correlations")
    st.plotly_chart(fig)

def main():
    """Main dashboard application."""
    st.set_page_config(page_title="Article Generation Experiments", layout="wide")
    st.title("Article Generation Experiments")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Experiments", "Feedback Analysis", "Combined Insights"]
    )
    
    if page == "Experiments":
        st.header("Experiment Analysis")
        
        # Load experiments
        experiment_dir = os.path.join(os.getcwd(), "experiments")
        if not os.path.exists(experiment_dir):
            st.warning("No experiments directory found")
            return
            
        experiment_files = [
            f.replace(".json", "") 
            for f in os.listdir(experiment_dir) 
            if f.endswith(".json")
        ]
        
        if not experiment_files:
            st.warning("No experiments found")
            return
            
        experiment_name = st.selectbox(
            "Select Experiment",
            experiment_files
        )
        
        experiment = load_experiment(experiment_name)
        if not experiment:
            return
            
        # Display experiment info
        st.subheader("Experiment Overview")
        st.write(f"Description: {experiment.description}")
        st.write(f"Metrics: {', '.join(experiment.metrics)}")
        st.write(f"Number of variants: {len(experiment.variants)}")
        st.write(f"Number of trials: {len(experiment.trials)}")
        
        # Convert to DataFrame for analysis
        df = experiment.to_dataframe()
        
        # Time range filter
        st.subheader("Time Range")
        date_range = st.date_input(
            "Select Date Range",
            value=(
                df["timestamp"].min().date(),
                df["timestamp"].max().date()
            )
        )
        if len(date_range) == 2:
            start_date, end_date = date_range
            mask = (
                (df["timestamp"].dt.date >= start_date) &
                (df["timestamp"].dt.date <= end_date)
            )
            df = df[mask]
        
        # Metric analysis
        st.subheader("Metric Analysis")
        metric = st.selectbox("Select Metric", experiment.metrics)
        
        col1, col2 = st.columns(2)
        
        with col1:
            plot_metric_over_time(df, metric)
            
        with col2:
            plot_metric_distribution(df, metric)
        
        # Statistical analysis
        st.subheader("Statistical Analysis")
        control_variant = st.selectbox(
            "Select Control Variant",
            [v.name for v in experiment.variants.values()]
        )
        
        control_id = next(
            vid for vid, v in experiment.variants.items()
            if v.name == control_variant
        )
        
        results = experiment.analyze_results(metric, control_id)
        
        # Display results
        st.json(results)
        
    elif page == "Feedback Analysis":
        st.header("Feedback Analysis")
        
        feedback_manager = load_feedback()
        if not feedback_manager:
            return
            
        # Display feedback stats
        stats = feedback_manager.get_feedback_stats()
        if not stats:
            st.warning("No feedback data available")
            return
            
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Responses", stats["total_responses"])
            
        with col2:
            st.metric("Unique Articles", stats["unique_articles"])
            
        with col3:
            st.metric("Unique Evaluators", stats["unique_evaluators"])
        
        # Criteria analysis
        st.subheader("Criteria Analysis")
        
        # Convert criteria stats to DataFrame
        criteria_stats = pd.DataFrame.from_dict(
            stats["criteria_stats"],
            orient="index"
        )
        
        if not criteria_stats.empty:
            fig = px.bar(
                criteria_stats,
                y=criteria_stats.index,
                x="mean",
                error_x="std",
                title="Average Ratings by Criterion",
                labels={"index": "Criterion", "mean": "Average Rating"}
            )
            st.plotly_chart(fig)
            
            # Correlation heatmap
            st.subheader("Criteria Correlations")
            plot_feedback_heatmap(feedback_manager)
            
    else:  # Combined Insights
        st.header("Combined Insights")
        
        experiment = load_experiment(st.selectbox(
            "Select Experiment",
            [f.replace(".json", "") for f in os.listdir("experiments")]
        ))
        
        feedback_manager = load_feedback()
        
        if not experiment or not feedback_manager:
            return
            
        # Combine experiment and feedback data
        df_experiment = experiment.to_dataframe()
        
        feedback_scores = {}
        for trial in experiment.trials:
            scores = feedback_manager.calculate_article_score(trial.id)
            if scores:
                feedback_scores[trial.id] = scores
        
        df_feedback = pd.DataFrame.from_dict(feedback_scores, orient="index")
        
        if not df_feedback.empty:
            # Merge experiment and feedback data
            df_combined = df_experiment.merge(
                df_feedback,
                left_on="trial_id",
                right_index=True,
                how="inner"
            )
            
            # Correlation analysis
            st.subheader("Metric-Feedback Correlations")
            
            automated_metrics = experiment.metrics
            human_metrics = df_feedback.columns
            
            corr_data = []
            for auto_metric in automated_metrics:
                for human_metric in human_metrics:
                    if auto_metric in df_combined and human_metric in df_combined:
                        correlation = df_combined[auto_metric].corr(
                            df_combined[human_metric]
                        )
                        corr_data.append({
                            "Automated Metric": auto_metric,
                            "Human Metric": human_metric,
                            "Correlation": correlation
                        })
            
            if corr_data:
                df_corr = pd.DataFrame(corr_data)
                fig = px.imshow(
                    df_corr.pivot(
                        index="Automated Metric",
                        columns="Human Metric",
                        values="Correlation"
                    ),
                    title="Correlation between Automated and Human Metrics",
                    color_continuous_scale="RdBu"
                )
                st.plotly_chart(fig)
            
            # Scatter plots
            st.subheader("Metric Relationships")
            col1, col2 = st.columns(2)
            
            with col1:
                x_metric = st.selectbox(
                    "Select X-axis Metric",
                    automated_metrics,
                    key="x_metric"
                )
                
            with col2:
                y_metric = st.selectbox(
                    "Select Y-axis Metric",
                    list(human_metrics),
                    key="y_metric"
                )
            
            if x_metric in df_combined and y_metric in df_combined:
                fig = px.scatter(
                    df_combined,
                    x=x_metric,
                    y=y_metric,
                    color="variant_name",
                    trendline="ols",
                    title=f"{x_metric} vs {y_metric}"
                )
                st.plotly_chart(fig)
        else:
            st.warning("No matching feedback data found for experiment trials")

if __name__ == "__main__":
    main() 
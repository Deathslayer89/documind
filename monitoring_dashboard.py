"""
Monitoring Dashboard
Displays analytics and metrics for the RAG system with 5+ charts.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from feedback_storage import FeedbackStorage
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Configure page
st.set_page_config(
    page_title="RAG System Monitoring Dashboard",
    page_icon="📊",
    layout="wide"
)

def load_feedback_storage():
    """Load or initialize feedback storage."""
    if 'feedback_storage' not in st.session_state:
        st.session_state.feedback_storage = FeedbackStorage()
    return st.session_state.feedback_storage

def create_feedback_chart(feedback_df):
    """Chart 1: Feedback Distribution (Positive vs Negative)"""
    if feedback_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No feedback data available yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(title="Feedback Distribution", height=300)
        return fig
    
    feedback_counts = feedback_df['feedback'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=['Positive 👍', 'Negative 👎'],
        values=[
            feedback_counts.get('positive', 0),
            feedback_counts.get('negative', 0)
        ],
        marker=dict(colors=['#00cc44', '#ff4444']),
        hole=0.4
    )])
    
    fig.update_layout(
        title="User Feedback Distribution",
        height=400,
        showlegend=True
    )
    
    return fig

def create_feedback_timeline(feedback_df):
    """Chart 2: Feedback Over Time"""
    if feedback_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No feedback data available yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(title="Feedback Timeline", height=300)
        return fig
    
    feedback_df['timestamp'] = pd.to_datetime(feedback_df['timestamp'])
    feedback_df['date'] = feedback_df['timestamp'].dt.date
    
    daily_feedback = feedback_df.groupby(['date', 'feedback']).size().reset_index(name='count')
    
    fig = px.line(
        daily_feedback,
        x='date',
        y='count',
        color='feedback',
        color_discrete_map={'positive': '#00cc44', 'negative': '#ff4444'},
        title='Feedback Timeline',
        labels={'date': 'Date', 'count': 'Number of Feedback', 'feedback': 'Type'}
    )
    
    fig.update_layout(height=400)
    
    return fig

def create_response_time_chart(interactions_df):
    """Chart 3: Response Time Distribution"""
    if interactions_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No interaction data available yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(title="Response Time Distribution", height=300)
        return fig
    
    fig = go.Figure(data=[go.Histogram(
        x=interactions_df['response_time_seconds'],
        nbinsx=20,
        marker_color='#1f77b4'
    )])
    
    fig.update_layout(
        title="Response Time Distribution",
        xaxis_title="Response Time (seconds)",
        yaxis_title="Count",
        height=400
    )
    
    return fig

def create_query_volume_chart(interactions_df):
    """Chart 4: Query Volume Over Time"""
    if interactions_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No interaction data available yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(title="Query Volume Over Time", height=300)
        return fig
    
    interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'])
    interactions_df['date'] = interactions_df['timestamp'].dt.date
    
    daily_queries = interactions_df.groupby('date').size().reset_index(name='count')
    
    fig = px.bar(
        daily_queries,
        x='date',
        y='count',
        title='Daily Query Volume',
        labels={'date': 'Date', 'count': 'Number of Queries'},
        color_discrete_sequence=['#ff7f0e']
    )
    
    fig.update_layout(height=400)
    
    return fig

def create_sources_chart(interactions_df):
    """Chart 5: Sources Retrieved Distribution"""
    if interactions_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No interaction data available yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(title="Sources Retrieved Distribution", height=300)
        return fig
    
    sources_counts = interactions_df['sources_count'].value_counts().sort_index()
    
    fig = go.Figure(data=[go.Bar(
        x=sources_counts.index,
        y=sources_counts.values,
        marker_color='#2ca02c'
    )])
    
    fig.update_layout(
        title="Sources Retrieved per Query",
        xaxis_title="Number of Sources",
        yaxis_title="Count",
        height=400
    )
    
    return fig

def create_answer_length_chart(interactions_df):
    """Chart 6: Answer Length Distribution"""
    if interactions_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No interaction data available yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(title="Answer Length Distribution", height=300)
        return fig
    
    fig = go.Figure(data=[go.Box(
        y=interactions_df['answer_length'],
        marker_color='#9467bd',
        name='Answer Length (words)'
    )])
    
    fig.update_layout(
        title="Answer Length Distribution",
        yaxis_title="Words",
        height=400
    )
    
    return fig

def create_hourly_activity_chart(interactions_df):
    """Chart 7: Hourly Activity Pattern"""
    if interactions_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No interaction data available yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(title="Hourly Activity Pattern", height=300)
        return fig
    
    interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'])
    interactions_df['hour'] = interactions_df['timestamp'].dt.hour
    
    hourly_queries = interactions_df.groupby('hour').size().reset_index(name='count')
    
    fig = px.bar(
        hourly_queries,
        x='hour',
        y='count',
        title='Query Activity by Hour of Day',
        labels={'hour': 'Hour (24h format)', 'count': 'Number of Queries'},
        color='count',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(height=400)
    
    return fig

def main():
    st.title("📊 RAG System Monitoring Dashboard")
    st.markdown("Real-time analytics and performance metrics for the CS Textbooks RAG System")
    
    # Load feedback storage
    feedback_storage = load_feedback_storage()
    
    # Get data
    feedback_df = feedback_storage.get_feedback_dataframe()
    interactions_df = feedback_storage.get_interactions_dataframe()
    feedback_stats = feedback_storage.get_feedback_stats()
    interaction_stats = feedback_storage.get_interaction_stats()
    
    # Display summary metrics
    st.markdown("## 📈 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Queries",
            value=interaction_stats['total_queries'],
            delta=None
        )
    
    with col2:
        positive_rate = feedback_stats['positive_percentage']
        st.metric(
            label="Positive Feedback",
            value=f"{positive_rate:.1f}%",
            delta=f"{feedback_stats['positive_count']} / {feedback_stats['total_feedback']}"
        )
    
    with col3:
        avg_response = interaction_stats['avg_response_time']
        st.metric(
            label="Avg Response Time",
            value=f"{avg_response:.2f}s",
            delta=None
        )
    
    with col4:
        avg_sources = interaction_stats['avg_sources_count']
        st.metric(
            label="Avg Sources Retrieved",
            value=f"{avg_sources:.1f}",
            delta=None
        )
    
    st.markdown("---")
    
    # Chart 1 & 2: Feedback Analysis
    st.markdown("## 👍👎 Feedback Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_feedback_chart(feedback_df), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_feedback_timeline(feedback_df), use_container_width=True)
    
    st.markdown("---")
    
    # Chart 3 & 4: Performance Metrics
    st.markdown("## ⚡ Performance Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_response_time_chart(interactions_df), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_query_volume_chart(interactions_df), use_container_width=True)
    
    st.markdown("---")
    
    # Chart 5 & 6: Content Analysis
    st.markdown("## 📚 Content Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_sources_chart(interactions_df), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_answer_length_chart(interactions_df), use_container_width=True)
    
    st.markdown("---")
    
    # Chart 7: Activity Pattern
    st.markdown("## 🕐 Activity Patterns")
    st.plotly_chart(create_hourly_activity_chart(interactions_df), use_container_width=True)
    
    st.markdown("---")
    
    # Recent Feedback Table
    st.markdown("## 📝 Recent Feedback")
    
    if not feedback_df.empty:
        recent_feedback = feedback_df.tail(10)[['timestamp', 'question', 'feedback', 'sources_count', 'answer_length']].copy()
        recent_feedback['timestamp'] = pd.to_datetime(recent_feedback['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        recent_feedback['feedback'] = recent_feedback['feedback'].map({'positive': '👍 Positive', 'negative': '👎 Negative'})
        recent_feedback.columns = ['Time', 'Question', 'Feedback', 'Sources', 'Answer Length']
        
        st.dataframe(recent_feedback, use_container_width=True, hide_index=True)
    else:
        st.info("No feedback data available yet. Users need to rate answers in the main app.")
    
    st.markdown("---")
    
    # Export data option
    st.markdown("## 💾 Export Data")
    col1, col2 = st.columns(2)
    
    with col1:
        if not feedback_df.empty:
            csv_feedback = feedback_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Feedback Data (CSV)",
                data=csv_feedback,
                file_name=f"feedback_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No feedback data to export")
    
    with col2:
        if not interactions_df.empty:
            csv_interactions = interactions_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Interaction Data (CSV)",
                data=csv_interactions,
                file_name=f"interaction_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No interaction data to export")

if __name__ == "__main__":
    main()


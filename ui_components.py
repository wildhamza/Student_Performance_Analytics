import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.decomposition import PCA
from utils import get_grade_color_map

def display_eda(df):
    """Displays the Exploratory Data Analysis section."""
    tab1, tab2, tab3 = st.tabs([
        "Overview", "Behavioral Analysis", "Correlations"
    ])
    
    with tab1:
        st.markdown("### Dataset Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Students Tracked", len(df))
            st.metric("Average GPA", f"{df['GPA-current/previous-semester'].mean():.2f}")
        with col2:
            grade_counts = df['Grade_Band'].value_counts().sort_index()
            fig = px.pie(values=grade_counts.values, names=grade_counts.index, 
                         title='Students per Performance Level', 
                         color_discrete_sequence=px.colors.sequential.Plotly3)
            st.plotly_chart(fig, use_container_width=True)
            
        st.markdown("""
        **Data Insight:** This section shows the distribution of students across performance levels. 
        Most datasets have a 'Normal Distribution', but high 'Low' performer counts may indicate difficult coursework.
        """)

    with tab2:
        st.markdown("### How Habits Impact Performance")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(df, x='Grade_Band', y='hours-study-per-day-average', title='Study Hours by Performance')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(df, x='Grade_Band', y='hours-per-day-use-mobile phone', title='Mobile Phone Usage by Performance')
            st.plotly_chart(fig, use_container_width=True)
            
        st.markdown("""
        **How to read this:** Check the 'median' line in the box plots. 
        Higher study hours *should* correlate with the 'High' performance level, 
        while high mobile usage often correlates with 'Low' performance.
        """)

    with tab3:
        st.markdown("### Statistical Correlations")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu', title='Relationship Heatmap')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Interpretation:** Dark red boxes mean the factors increase together (Positive Correlation). 
        Blue boxes mean when one factor increases, the other decreases (Negative Correlation).
        """)

def display_model_dashboard(results, metrics_df, best_model_name):
    """Displays comparison of trained classification models."""
    st.markdown("### Predictive Performance")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        metrics_melted = metrics_df.melt(id_vars=['Model'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'], var_name='Metric', value_name='Score')
        fig = px.bar(metrics_melted, x='Model', y='Score', color='Metric', barmode='group', title='Accuracy Comparison')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.success(f"**Top Model:** {best_model_name}")
        st.markdown(f"""
        **What this means:** The {best_model_name} model was the most reliable in finding patterns 
        between study habits and final grades. 
        An accuracy of 0.60+ is generally considered 'Good' for behavioral data.
        """)

def display_cluster_analysis(df, X_scaled, n_clusters, features, algorithm='K-Means'):
    """Visualizes student segmentation results."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    cluster_viz_df = pd.DataFrame({'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1], 
                                  'Group': df['Cluster'].astype(str), 'Level': df['Grade_Band']})
    
    st.markdown(f"#### Student Groups ({algorithm})")
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.scatter(cluster_viz_df, x='PC1', y='PC2', color='Group', title='Behavioral Segments')
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.scatter(cluster_viz_df, x='PC1', y='PC2', color='Level', title='Segments by Academic Level')
        st.plotly_chart(fig2, use_container_width=True)
        
    st.markdown("""
    **Understanding Segmentation:** The left chart groups students by *habits* (study hours, sleep, etc.). 
    The right chart shows their *actual performance*. 
    If a specific group on the left matches a specific level on the right, we've found a 'Success Pattern'.
    """)

def display_insights(importance_df, profile_dict):
    """Displays feature rankings and persona descriptions."""
    st.markdown("### Actionable Success Factors")
    
    tab1, tab2 = st.tabs(["Key Success Factors", "Student Personas"])
    
    with tab1:
        st.markdown("#### What drives academic success?")
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', 
                     title='Feature Impact Ranking',
                     color='Importance', color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Note:** Features at the top are the strongest predictors. 
        If 'Social Media' is at the top, it means students' grades are highly sensitive to social media use.
        """)

    with tab2:
        st.markdown("#### Typical Student Profiles")
        cols = st.columns(len(profile_dict))
        for i, (cluster, profile) in enumerate(profile_dict.items()):
            with cols[i]:
                st.markdown(f"""
                <div style="background-color: #F0F9FF; padding: 1rem; border-radius: 8px; border-top: 4px solid #0EA5E9; height: 100%;">
                    <h4 style="color: #0369A1; margin: 0;">Group {cluster}</h4>
                    <p style="font-size: 0.9rem;">{profile}</p>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("""
        <br>
        **How to use:** Identification of these profiles helps in designing targeted support for each group of students.
        """, unsafe_allow_html=True)

def display_prediction_results(prediction, probabilities, classes):
    """Renders the final classification output and confidence scores."""
    color_map = get_grade_color_map()
    prediction_color = color_map.get(prediction, '#6B7280')
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style="background-color: {prediction_color}; color: white; padding: 2rem; border-radius: 10px; text-align: center;">
            <h3>Predicted Performance</h3>
            <h1 style="font-size: 3.5rem; margin: 0;">{prediction}</h1>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        prob_df = pd.DataFrame({'Level': classes, 'Confidence %': probabilities * 100})
        fig = px.bar(prob_df, x='Confidence %', y='Level', title='Model Confidence', 
                     orientation='h', color='Level', color_discrete_map=color_map)
        st.plotly_chart(fig, use_container_width=True)

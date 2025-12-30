import streamlit as st
import pandas as pd
import numpy as np
from data_engine import load_data, preprocess_data
from model_engine import (
    train_classification_models, custom_kmeans, get_prediction_recommendations,
    get_cluster_profiles, hierarchical_clustering
)
from ui_components import (
    display_eda, display_model_dashboard, display_cluster_analysis, 
    display_prediction_results, display_insights
)
from utils import get_grade_color_map

# Page configuration
st.set_page_config(
    page_title="Student Analytics",
    layout="wide"
)

# Application Styles
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; color: #1E3A8A; text-align: center; margin-bottom: 2rem; }
    .stAlert { margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">Student Performance Analytics</h1>', unsafe_allow_html=True)
    
    try:
        data = load_data()
        df, le_dict, categorical_cols, summary = preprocess_data(data)
        
        with st.sidebar:
            st.markdown("### Navigation")
            section = st.radio("Go to:", ["Overview", "Analysis", "Clustering", "Predictor"])
            
            st.markdown("---")
            st.markdown("**How to use:**")
            st.info("Start with **Overview** to see data stats, then use **Analysis** to train models and see insights.")
        
        if section == "Overview":
            display_eda(df)
            
        elif section == "Analysis":
            st.markdown("### Model Training & Insights")
            results, best_model_name, features, scaler, importance_df = train_classification_models(df, categorical_cols)
            
            # Simple metrics display
            metrics_df = pd.DataFrame({
                'Model': list(results.keys()),
                'Accuracy': [results[m]['accuracy'] for m in results],
                'Precision': [results[m]['precision'] for m in results],
                'Recall': [results[m]['recall'] for m in results],
                'F1-Score': [results[m]['f1_score'] for m in results],
                'Training Time (s)': [results[m]['training_time'] for m in results],
                'Inference Time (s)': [results[m]['inference_time'] for m in results]
            })
            
            display_model_dashboard(results, metrics_df, best_model_name)
            
            st.markdown("---")
            # Automatically show insights after training
            clustering_features = ['hours-study-per-day-average', 'average-class-attendance', 'hours-sleep-per-night']
            X = df[clustering_features].values
            from sklearn.preprocessing import StandardScaler
            X_scaled = StandardScaler().fit_transform(X)
            
            labels, _, _ = custom_kmeans(X_scaled, n_clusters=3)
            df['Cluster'] = labels
            profiles = get_cluster_profiles(df, clustering_features)
            
            display_insights(importance_df, profiles)

            # Store in session state for predictor
            st.session_state.update({
                'results': results, 
                'best_model': best_model_name, 
                'features': features, 
                'scaler': scaler, 
                'le_dict': le_dict
            })

        elif section == "Clustering":
            st.markdown("### Student Segmentation")
            clustering_features = ['hours-study-per-day-average', 'average-class-attendance', 'hours-sleep-per-night']
            X = df[clustering_features].values
            from sklearn.preprocessing import StandardScaler
            X_scaled = StandardScaler().fit_transform(X)
            
            col1, col2 = st.columns(2)
            with col1:
                n_clusters = st.slider("Number of Groups:", 2, 5, 3)
            with col2:
                algo = st.selectbox("Method:", ["K-Means", "Hierarchical"])
            
            labels = custom_kmeans(X_scaled, n_clusters=n_clusters)[0] if algo == "K-Means" else hierarchical_clustering(X_scaled, n_clusters=n_clusters)
            df['Cluster'] = labels
            display_cluster_analysis(df, X_scaled, n_clusters, clustering_features, algorithm=algo)
            
        elif section == "Predictor":
            st.markdown("### Performance Predictor")
            if 'results' in st.session_state:
                with st.form("pred_form"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        semester = st.slider("Semester", 1, 8, 4)
                        study = st.slider("Study Hours", 0.0, 12.0, 4.0)
                    with col2:
                        attend = st.slider("Attendance %", 0.0, 100.0, 85.0)
                        mobile = st.slider("Mobile Use", 0.0, 10.0, 3.0)
                    with col3:
                        sleep = st.slider("Sleep Hours", 4.0, 10.0, 7.0)
                        stress = st.slider("Stress (0-5)", 0, 5, 2)
                    
                    if st.form_submit_button("Predict Performance"):
                        best_model = st.session_state['results'][st.session_state['best_model']]['model']
                        features = st.session_state['features']
                        scaler = st.session_state['scaler']
                        
                        input_dict = {
                            'current-semester': semester,
                            'hours-study-per-day-average': study,
                            'hours-per-day-use-mobile phone': mobile,
                            'hours-sleep-per-night': sleep,
                            'average-class-attendance': attend,
                            'academic-stress-level': stress
                        }
                        
                        input_df = pd.DataFrame([input_dict])
                        for f in features:
                            if f not in input_df.columns: input_df[f] = 0
                        input_df = input_df[features]
                        
                        input_scaled = scaler.transform(input_df)
                        prediction = best_model.predict(input_scaled)[0]
                        probs = best_model.predict_proba(input_scaled)[0]
                        display_prediction_results(prediction, probs, best_model.classes_)
                        st.info(get_prediction_recommendations(prediction))
            else:
                st.warning("Please run 'Analysis' first to calibrate the predictor.")

    except Exception as e:
        st.error(f"Execution Error: {e}")

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, silhouette_score
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #1E40AF;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3B82F6;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .graph-explanation {
        background-color: #F0F9FF;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        font-size: 0.9rem;
        color: #1E40AF;
        border-left: 3px solid #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv('synthetic_survey_3000.csv')
    return data

@st.cache_data
def preprocess_data(data):
    df = data.copy()
    
    # Store initial statistics
    initial_count = len(df)
    
    # 1. Remove specified departments if 'department' column exists
    if 'department' in df.columns:
        departments_to_remove = ['DPT', 'BS Urdu', 'Radiology', 'BS Mathematics']
        df = df[~df['department'].isin(departments_to_remove)]
        removed_dept = initial_count - len(df)
        if removed_dept > 0:
            st.sidebar.success(f"Removed {removed_dept} students from specified departments")
    
    # 2. Remove duplicate rows
    before_duplicates = len(df)
    df = df.drop_duplicates()
    removed_duplicates = before_duplicates - len(df)
    if removed_duplicates > 0:
        st.sidebar.info(f"Removed {removed_duplicates} duplicate records")
    
    # 3. Handle missing values
    before_missing = len(df)
    df = df.dropna()
    removed_missing = before_missing - len(df)
    if removed_missing > 0:
        st.sidebar.warning(f"Removed {removed_missing} records with missing values")
    
    # 4. Remove outliers for key numerical columns using IQR method
    def remove_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    # Define columns to check for outliers
    outlier_columns = [
        'hours-study-per-day-average',
        'hours-per-day-use-mobile phone',
        'hours-sleep-per-night',
        'hours-per-day-use-social-media-apps',
        'GPA-current/previous-semester'
    ]
    
    # Remove outliers for each column
    before_outliers = len(df)
    for col in outlier_columns:
        if col in df.columns:
            df = remove_outliers_iqr(df, col)
    removed_outliers = before_outliers - len(df)
    if removed_outliers > 0:
        st.sidebar.info(f"Removed {removed_outliers} outlier records")
    
    # Show data cleaning summary
    total_removed = initial_count - len(df)
    st.sidebar.markdown(f"**Final dataset:** {len(df)} students (Removed {total_removed} records)")
    
    # Create grade bands according to proposal (A+, A, B+, B, C+, C)
    def create_grade_bands(gpa):
        if gpa >= 3.7:
            return 'A+'
        elif gpa >= 3.3:
            return 'A'
        elif gpa >= 3.0:
            return 'B+'
        elif gpa >= 2.7:
            return 'B'
        elif gpa >= 2.3:
            return 'C+'
        else:
            return 'C'
    
    df['Grade_Band'] = df['GPA-current/previous-semester'].apply(create_grade_bands)
    
    # Identify categorical columns (excluding the target)
    categorical_cols = []
    for col in df.columns:
        if col != 'Grade_Band' and col != 'GPA-current/previous-semester' and col != 'expected-GPA-this-course/semester':
            if df[col].dtype == 'object' or df[col].nunique() < 15:
                categorical_cols.append(col)
    
    # Convert categorical variables
    le_dict = {}
    for col in categorical_cols:
        if col in df.columns and col not in ['GPA-current/previous-semester', 'expected-GPA-this-course/semester']:
            try:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                le_dict[col] = le
            except Exception as e:
                # If encoding fails, drop the column
                df = df.drop(columns=[col])
                if col in categorical_cols:
                    categorical_cols.remove(col)
    
    return df, le_dict, categorical_cols

# EDA Functions with improved visualizations
def perform_eda(df):
    st.markdown('<div class="section-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    # Create tabs for different EDA sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview & Distribution", 
        "🎯 Grade Band Analysis", 
        "📱 Behavioral Patterns", 
        "📊 Correlations", 
        "📉 Statistical Insights"
    ])
    
    with tab1:
        st.markdown("### Dataset Overview & Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Key Statistics")
            st.write(f"**Total Students:** {len(df)}")
            st.write(f"**Total Features:** {len(df.columns)}")
            st.write(f"**Average GPA:** {df['GPA-current/previous-semester'].mean():.2f}")
            st.write(f"**Grade Bands:** {sorted(df['Grade_Band'].unique())}")
            
            # Show data types
            st.markdown("#### Data Types")
            dtype_counts = df.dtypes.value_counts()
            for dtype, count in dtype_counts.items():
                st.write(f"- {dtype}: {count} columns")
        
        with col2:
            st.markdown("#### GPA Distribution")
            fig = px.histogram(df, x='GPA-current/previous-semester', 
                             nbins=20, title='Distribution of GPA Scores',
                             color_discrete_sequence=['#3B82F6'])
            fig.update_layout(xaxis_title="GPA", yaxis_title="Number of Students")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('<div class="graph-explanation">This histogram shows the distribution of GPA scores. A normal distribution indicates balanced performance across students.</div>', unsafe_allow_html=True)
        
        # Grade Band Distribution
        st.markdown("### Grade Band Distribution")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            grade_counts = df['Grade_Band'].value_counts().sort_index()
            fig = px.bar(x=grade_counts.index, y=grade_counts.values,
                        title='Number of Students in Each Grade Band',
                        color=grade_counts.values,
                        color_continuous_scale='Blues',
                        labels={'x': 'Grade Band', 'y': 'Number of Students'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('<div class="graph-explanation">This bar chart shows the count of students in each grade band, helping identify common performance levels.</div>', unsafe_allow_html=True)
        
        with col2:
            fig = px.pie(values=grade_counts.values, names=grade_counts.index,
                        title='Grade Band Proportions',
                        color_discrete_sequence=px.colors.sequential.Blues_r)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('<div class="graph-explanation">This pie chart shows the percentage distribution of students across grade bands.</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### Grade Band Analysis by Different Factors")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Study Hours vs GPA by Grade Band
            fig = px.box(df, x='Grade_Band', y='hours-study-per-day-average',
                        title='Study Hours Distribution by Grade Band',
                        color='Grade_Band',
                        color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(xaxis_title="Grade Band", yaxis_title="Study Hours per Day")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('<div class="graph-explanation">Box plot showing how study hours vary across different grade bands. The boxes show quartiles, median, and outliers.</div>', unsafe_allow_html=True)
        
        with col2:
            # Attendance vs GPA by Grade Band
            fig = px.box(df, x='Grade_Band', y='average-class-attendance',
                        title='Class Attendance by Grade Band',
                        color='Grade_Band',
                        color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(xaxis_title="Grade Band", yaxis_title="Attendance Percentage")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('<div class="graph-explanation">Box plot showing class attendance patterns across grade bands, highlighting the relationship between attendance and grades.</div>', unsafe_allow_html=True)
        
        # Mobile Usage vs Social Media by Grade Band
        st.markdown("### Digital Behavior by Grade Band")
        fig = px.scatter(df, x='hours-per-day-use-mobile phone', 
                       y='hours-per-day-use-social-media-apps',
                       color='Grade_Band',
                       title='Mobile Usage vs Social Media Usage by Grade Band',
                       opacity=0.7,
                       color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(xaxis_title="Mobile Usage (hours/day)", yaxis_title="Social Media Usage (hours/day)")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="graph-explanation">Scatter plot showing the relationship between mobile usage and social media usage, colored by grade bands. Helps identify digital behavior patterns.</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### Behavioral Pattern Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sleep Patterns
            fig = px.violin(df, y='hours-sleep-per-night', x='Grade_Band',
                          title='Sleep Duration Distribution by Grade Band',
                          color='Grade_Band',
                          box=True,
                          points='all',
                          color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(xaxis_title="Grade Band", yaxis_title="Sleep Hours per Night")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('<div class="graph-explanation">Violin plot showing the distribution of sleep hours across grade bands. Combines box plot and density estimation.</div>', unsafe_allow_html=True)
        
        with col2:
            # Stress vs Motivation
            fig = px.scatter(df, x='academic-stress-level', y='motivation-level',
                           color='Grade_Band',
                           title='Stress Level vs Motivation Level by Grade Band',
                           color_discrete_sequence=px.colors.qualitative.Vivid)
            fig.update_layout(xaxis_title="Academic Stress Level (0-5)", yaxis_title="Motivation Level (1-5)")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('<div class="graph-explanation">Scatter plot showing the relationship between stress and motivation levels, colored by grade bands.</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown("### Feature Correlation Analysis")
        
        # Select only numeric columns for correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_df = df[numeric_cols]
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        fig = px.imshow(corr_matrix,
                       text_auto=True,
                       aspect="auto",
                       color_continuous_scale='RdBu',
                       title='Correlation Matrix of Numerical Features')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="graph-explanation">Heatmap showing correlations between numerical features. Red indicates positive correlation, blue indicates negative correlation.</div>', unsafe_allow_html=True)
        
        # Top correlations with GPA
        st.markdown("### Top Correlations with GPA")
        gpa_correlations = corr_matrix['GPA-current/previous-semester'].sort_values(ascending=False)
        
        fig = px.bar(x=gpa_correlations.index[1:11], y=gpa_correlations.values[1:11],
                    title='Top 10 Features Correlated with GPA',
                    color=gpa_correlations.values[1:11],
                    color_continuous_scale='Blues')
        fig.update_layout(xaxis_title="Feature", yaxis_title="Correlation Coefficient")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="graph-explanation">Bar chart showing features most strongly correlated with GPA. Helps identify key predictors.</div>', unsafe_allow_html=True)
    
    with tab5:
        st.markdown("### Statistical Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Summary statistics
            st.markdown("#### Summary Statistics")
            numeric_summary = df.select_dtypes(include=[np.number]).describe()
            st.dataframe(numeric_summary.style.background_gradient(cmap='Blues'))
        
        with col2:
            # Distribution of key variables
            st.markdown("#### Key Variable Distributions")
            
            selected_var = st.selectbox(
                "Select variable to view distribution:",
                ['hours-study-per-day-average', 'hours-per-day-use-mobile phone',
                 'hours-sleep-per-night', 'average-class-attendance',
                 'hours-per-day-use-social-media-apps']
            )
            
            fig = px.histogram(df, x=selected_var, nbins=20,
                             title=f'Distribution of {selected_var}',
                             color_discrete_sequence=['#10B981'])
            st.plotly_chart(fig, use_container_width=True)

# Classification Models with Algorithm Explanation
def train_classification_models(df, categorical_cols):
    st.markdown('<div class="section-header">🎯 Classification Models (Grade Band Prediction)</div>', unsafe_allow_html=True)
    
    # Algorithm Explanation
    with st.expander("ℹ️ **Simple Guide to Our Algorithms**"):
        st.markdown("""
        ### Why We Chose These 4 Algorithms:
        
        **1. Logistic Regression** 📈
           - **What it does**: Simple "yes/no" type predictions
           - **Why we use it**: Easy to understand, gives us a starting point
           - **Like asking**: "Does this student's study pattern match A+ students?"
        
        **2. Decision Tree** 🌳
           - **What it does**: Makes decisions like a flowchart
           - **Why we use it**: Shows clear rules (e.g., "if study hours > 4 AND attendance > 80%, then likely A")
           - **Like asking**: "What study habits lead to which grades?"
        
        **3. Random Forest** 🌲🌲🌲
           - **What it does**: Uses many decision trees together
           - **Why we use it**: Most accurate, doesn't make silly mistakes
           - **Like asking**: "What do most decision trees predict for this student?"
        
        **4. Support Vector Machine (SVM)** 🔗
           - **What it does**: Finds clear dividing lines between groups
           - **Why we use it**: Works well when groups are clearly different
           - **Like asking**: "Can we draw clear lines between A, B, and C students?"
        
        ### Why We Didn't Use Other Algorithms:
        - **Neural Networks** 🧠: Too complex for our small dataset (like using a rocket to go to the store)
        - **K-Nearest Neighbors** 👥: Slower with many students, needs more computing power
        - **Naive Bayes** 📊: Makes assumptions that don't fit our data well
        
        """)
    
    # Select features
    numeric_features = ['current-semester', 'hours-study-per-day-average', 
                       'hours-per-day-use-mobile phone', 'hours-sleep-per-night',
                       'average-class-attendance', 'hours-per-day-use-social-media-apps',
                       'academic-stress-level', 'motivation-level']
    
    # Get available features
    features = [f for f in numeric_features if f in df.columns]
    
    # Add categorical features if they exist and are encoded
    for cat_col in categorical_cols:
        if cat_col in df.columns:
            features.append(cat_col)
    
    X = df[features]
    y = df['Grade_Band']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize classification models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Support Vector Machine': SVC(kernel='rbf', random_state=42, probability=True)
    }
    
    results = {}
    
    # Train and evaluate models
    progress_bar = st.progress(0)
    
    for i, (name, model) in enumerate(models.items()):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
        
        # Calculate multiple metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        # Get precision, recall, f1 for each class
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1 = report['weighted avg']['f1-score']
        
        # Calculate training time (simple measure)
        import time
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        
        # Calculate inference time
        start_time = time.time()
        model.predict(X_test_scaled[:10])  # Small sample
        inference_time = (time.time() - start_time) / 10
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'training_time': training_time,
            'inference_time': inference_time,
            'predictions': y_pred,
            'predictions_proba': y_pred_proba,
            'confusion_matrix': cm,
            'classification_report': report,
            'y_test': y_test
        }
        progress_bar.progress((i + 1) / len(models))
    
    # Create comprehensive comparison
    st.markdown("### 📊 Model Performance Dashboard")
    
    # Create metrics comparison
    metrics_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[m]['accuracy'] for m in results],
        'Precision': [results[m]['precision'] for m in results],
        'Recall': [results[m]['recall'] for m in results],
        'F1-Score': [results[m]['f1_score'] for m in results],
        'Training Time (s)': [results[m]['training_time'] for m in results],
        'Inference Time (s)': [results[m]['inference_time'] for m in results]
    })
    
    # Display metrics comparison
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("#### 📈 Performance Metrics Comparison")
        
        # Melt dataframe for easier plotting
        metrics_melted = metrics_df.melt(id_vars=['Model'], 
                                       value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                                       var_name='Metric', value_name='Score')
        
        fig = px.bar(metrics_melted, x='Model', y='Score', color='Metric',
                    barmode='group', title='Model Performance Across Metrics',
                    color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="graph-explanation">Comparing multiple metrics helps identify which model performs best overall, not just in accuracy.</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### ⏱️ Speed Comparison")
        
        # Speed comparison
        fig_speed = go.Figure(data=[
            go.Bar(name='Training Time', x=metrics_df['Model'], y=metrics_df['Training Time (s)']),
            go.Bar(name='Inference Time', x=metrics_df['Model'], y=metrics_df['Inference Time (s)'])
        ])
        
        fig_speed.update_layout(
            title='Model Speed Comparison',
            barmode='group',
            height=400,
            yaxis_title='Time (seconds)'
        )
        
        st.plotly_chart(fig_speed, use_container_width=True)
        
        st.markdown('<div class="graph-explanation">Training time and inference speed matter for practical applications.</div>', unsafe_allow_html=True)
    
    # Model Selection Criteria Analysis
    st.markdown("### 🎯 Model Selection Matrix")
    
    # Create radar chart for model comparison
    metrics_for_radar = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Speed']
    
    fig_radar = go.Figure()
    
    for model_name in results.keys():
        # Normalize values for radar chart (0 to 1)
        accuracy_norm = results[model_name]['accuracy']
        precision_norm = results[model_name]['precision']
        recall_norm = results[model_name]['recall']
        f1_norm = results[model_name]['f1_score']
        
        # Speed score (faster is better, so invert)
        speed_norm = 1 - min(results[model_name]['inference_time'] / metrics_df['Inference Time (s)'].max(), 1)
        
        fig_radar.add_trace(go.Scatterpolar(
            r=[accuracy_norm, precision_norm, recall_norm, f1_norm, speed_norm],
            theta=metrics_for_radar,
            fill='toself',
            name=model_name
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        height=500,
        title='Model Performance Radar Chart'
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    st.markdown('<div class="graph-explanation">Radar chart shows each model\'s strengths and weaknesses across different metrics.</div>', unsafe_allow_html=True)
    
    # Model Selection Guide
    st.markdown("### 🏆 Final Model Selection")
    
    # Calculate overall scores (weighted average)
    weights = {
        'accuracy': 0.3,
        'precision': 0.2,
        'recall': 0.2,
        'f1_score': 0.2,
        'speed': 0.1  # Speed weight
    }
    
    overall_scores = {}
    for model_name in results.keys():
        score = (results[model_name]['accuracy'] * weights['accuracy'] +
                results[model_name]['precision'] * weights['precision'] +
                results[model_name]['recall'] * weights['recall'] +
                results[model_name]['f1_score'] * weights['f1_score'] +
                (1 - results[model_name]['inference_time'] / metrics_df['Inference Time (s)'].max()) * weights['speed'])
        overall_scores[model_name] = score
    
    # Find best model based on overall score
    best_model_name = max(overall_scores, key=overall_scores.get)
    best_model = results[best_model_name]
    
    # Display selection results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Selected Model", best_model_name)
    
    with col2:
        st.metric("Overall Score", f"{overall_scores[best_model_name]:.3f}")
    
    with col3:
        st.metric("Primary Strength", 
                 "Accuracy" if best_model['accuracy'] == max([results[m]['accuracy'] for m in results]) else
                 "F1-Score" if best_model['f1_score'] == max([results[m]['f1_score'] for m in results]) else
                 "Speed" if best_model['inference_time'] == min([results[m]['inference_time'] for m in results]) else
                 "Balanced")
    
    # Detailed comparison table
    with st.expander("📋 Detailed Model Comparison Table"):
        # Format metrics for display
        display_df = metrics_df.copy()
        for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")
        for col in ['Training Time (s)', 'Inference Time (s)']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
        
        # Highlight best in each category
        def highlight_max(s):
            is_max = s == s.max()
            return ['background-color: #90EE90' if v else '' for v in is_max]
        
        def highlight_min(s):
            is_min = s == s.min()
            return ['background-color: #90EE90' if v else '' for v in is_min]
        
        # Convert back to numeric for highlighting
        numeric_df = metrics_df.copy()
        styled_df = numeric_df.style\
            .apply(highlight_max, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'])\
            .apply(highlight_min, subset=['Training Time (s)', 'Inference Time (s)'])\
            .format({
                'Accuracy': '{:.3f}',
                'Precision': '{:.3f}',
                'Recall': '{:.3f}',
                'F1-Score': '{:.3f}',
                'Training Time (s)': '{:.4f}',
                'Inference Time (s)': '{:.4f}'
            })
        
        st.dataframe(styled_df)
    
    # Model Selection Explanation
    st.markdown("### 📝 Selection Criteria Explained")
    
    selection_criteria = """
    #### Why We Look Beyond Accuracy:
    
    **1. Precision** - When we predict A grade, how often are we right?
       - *Important for*: Giving reliable grade predictions
    
    **2. Recall** - Can we find all the A students?
       - *Important for*: Identifying at-risk students
    
    **3. F1-Score** - Balance between precision and recall
       - *Important for*: Overall model quality
    
    **4. Training Time** - How long to train the model
       - *Important for*: Research and development
    
    **5. Inference Time** - How fast to make predictions
       - *Important for*: Real-time applications
    
    #### Our Weighting Strategy:
    - **Accuracy (30%)**: Primary goal is correct predictions
    - **Precision (20%)**: Avoid false high-grade predictions
    - **Recall (20%)**: Don't miss at-risk students
    - **F1-Score (20%)**: Overall balanced performance
    - **Speed (10%)**: Practical usability
    
    #### Final Recommendation:
    Based on **{best_model_name}**'s balanced performance across all metrics.
    """.format(best_model_name=best_model_name)
    
    st.info(selection_criteria)
    
    # Show confusion matrix for selected model
    st.markdown(f"### 🎯 Confusion Matrix - {best_model_name}")
    
    cm = best_model['confusion_matrix']
    labels = sorted(y.unique())
    
    fig_cm = ff.create_annotated_heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        showscale=True,
        annotation_text=cm.astype(str)
    )
    
    fig_cm.update_layout(
        title=f'Confusion Matrix - {best_model_name}',
        height=500,
        xaxis_title="Predicted Label",
        yaxis_title="True Label"
    )
    
    st.plotly_chart(fig_cm, use_container_width=True)
    
    # Model-specific insights
    st.markdown(f"### 💡 Insights for {best_model_name}")
    
    if best_model_name == 'Logistic Regression':
        st.markdown("""
        #### Why Logistic Regression Might Be Best:
        - **Simple linear relationships**: Student habits follow predictable patterns
        - **No complex interactions**: Each factor works independently
        - **Well-separated classes**: Grade bands are clearly different
        - **Interpretability**: Easy to understand coefficients
        
        #### Limitations to Consider:
        - May not capture complex interactions between habits
        - Assumes linear relationships
        - Sensitive to outliers
        """)
    elif best_model_name == 'Decision Tree':
        st.markdown("""
        #### Why Decision Tree Might Be Best:
        - **Clear decision rules**: Creates intuitive "if-then" rules that are easy to understand
        - **Handles non-linear relationships**: Study habits may have complex thresholds
        - **No feature scaling needed**: Works well with different measurement scales
        - **Feature importance**: Shows which factors split students most effectively
        
        #### Limitations to Consider:
        - Can overfit to training data (prone to memorizing)
        - Small changes in data can create very different trees
        - May create overly complex trees if not pruned
        - Less accurate than ensemble methods
        """)
    elif best_model_name == 'Random Forest':
        st.markdown("""
        #### Why Random Forest Might Be Best:
        - **Captures complex patterns**: Study habits interact in complex ways
        - **Robust to outliers**: One unusual student doesn't ruin predictions
        - **Feature importance**: Shows which habits matter most
        - **Good generalization**: Works well on new students
        
        #### Limitations to Consider:
        - Less interpretable than simpler models
        - Can be slower to train
        - Requires more computational resources
        """)
    elif best_model_name == 'Support Vector Machine':
        st.markdown("""
        #### Why Support Vector Machine Might Be Best:
        - **Effective with clear boundaries**: Grade bands are well-separated in feature space
        - **Works well in high dimensions**: Handles multiple student features effectively
        - **Robust to overfitting**: Especially with appropriate kernel selection
        - **Memory efficient**: Uses only support vectors for predictions
        
        #### Limitations to Consider:
        - Less interpretable than tree-based models
        - Sensitive to parameter tuning (C, gamma)
        - Can be slow with large datasets
        - Not probabilistic by default (uses Platt scaling)
        """)
        st.markdown("""
        #### Why Random Forest Might Be Best:
        - **Captures complex patterns**: Study habits interact in complex ways
        - **Robust to outliers**: One unusual student doesn't ruin predictions
        - **Feature importance**: Shows which habits matter most
        - **Good generalization**: Works well on new students
        
        #### Limitations to Consider:
        - Less interpretable than simpler models
        - Can be slower to train
        """)
        
        return results, best_model_name, features, scaler

def custom_kmeans(X, n_clusters=3, max_iter=100):
    """Custom K-Means implementation without scikit-learn dependency"""
    np.random.seed(42)
    
    # Ensure X is a numpy array and handle NaN values
    X = np.array(X, dtype=np.float64)
    
    # Check for NaN or infinite values
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        # Replace NaN with column mean
        col_mean = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_mean, inds[1])
    
    n_samples, n_features = X.shape
    
    # Initialize centroids randomly from existing points
    if n_samples > 0:
        centroids = X[np.random.choice(n_samples, min(n_clusters, n_samples), replace=False)]
        
        # If we need more centroids than samples, replicate some
        if n_clusters > n_samples:
            additional = n_clusters - n_samples
            extra_centroids = X[np.random.choice(n_samples, additional, replace=True)]
            centroids = np.vstack([centroids, extra_centroids])
    else:
        # Fallback: random initialization
        centroids = np.random.randn(n_clusters, n_features)
    
    for iteration in range(max_iter):
        # Calculate distances between each point and each centroid
        # Using efficient numpy broadcasting
        distances = np.sqrt(((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
        
        # Assign each point to the nearest centroid
        labels = np.argmin(distances, axis=1)
        
        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for i in range(n_clusters):
            # Get points assigned to cluster i
            cluster_points = X[labels == i]
            
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)
            else:
                # If cluster is empty, reinitialize with a random point
                new_centroids[i] = X[np.random.choice(n_samples, 1)[0]]
        
        # Check convergence (if centroids don't change much)
        if np.allclose(centroids, new_centroids, rtol=1e-4):
            break
            
        centroids = new_centroids
    
    # Calculate inertia (WCSS)
    distances_final = np.sqrt(((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
    inertia = np.sum(np.min(distances_final, axis=1) ** 2)
    
    return labels, centroids, inertia

def visualize_clusters(df, X_scaled, cluster_column, features, n_clusters):
    """Helper function to visualize clusters"""
    
    st.markdown(f"### Cluster Analysis ({n_clusters} clusters)")
    
    # Reduce dimensions for visualization
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    cluster_viz_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Cluster': df[cluster_column],
        'GPA': df['GPA-current/previous-semester'],
        'Grade_Band': df['Grade_Band']
    })
    
    # Create visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.scatter(cluster_viz_df, x='PC1', y='PC2',
                        color='Cluster',
                        title=f'Cluster Visualization (PCA)',
                        color_discrete_sequence=px.colors.qualitative.Set1,
                        hover_data=['GPA', 'Grade_Band'])
        st.plotly_chart(fig1, use_container_width=True)
        
        st.markdown('<div class="graph-explanation">2D visualization of clusters using PCA. Each point represents a student, colored by cluster assignment.</div>', unsafe_allow_html=True)
    
    with col2:
        fig2 = px.scatter(cluster_viz_df, x='PC1', y='PC2',
                        color='Grade_Band',
                        title='Clusters Colored by Grade Band',
                        color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown('<div class="graph-explanation">Same visualization colored by actual grade bands. Shows if clusters correspond to academic performance.</div>', unsafe_allow_html=True)
    
    # Analyze clusters
    st.markdown("### Cluster Profiles")
    
    for cluster_id in sorted(df[cluster_column].unique()):
        cluster_data = df[df[cluster_column] == cluster_id]
        
        if cluster_id == -1:  # Noise points in DBSCAN
            cluster_name = "Noise/Outliers"
        else:
            cluster_name = f"Cluster {cluster_id}"
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            percentage = len(cluster_data) / len(df) * 100
            st.markdown(f"""
            <div style="background-color: #F0F9FF; 
                        color: #1E3A8A;  
                        padding: 1rem; 
                        border-radius: 10px; 
                        margin-bottom: 1rem; 
                        border-left: 4px solid {px.colors.qualitative.Set1[cluster_id % len(px.colors.qualitative.Set1) if cluster_id != -1 else 0]}">
                <h4 style="color: #1E40AF;">📊 {cluster_name}</h4>  
                <p><strong>Number of Students:</strong> {len(cluster_data)} ({percentage:.1f}%)</p>
                <p><strong>Average GPA:</strong> {cluster_data['GPA-current/previous-semester'].mean():.2f}</p>
                <p><strong>Most Common Grade Band:</strong> {cluster_data['Grade_Band'].mode().iloc[0] if not cluster_data['Grade_Band'].mode().empty else 'N/A'}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Mini metrics for key features
            if cluster_id != -1:
                metrics = ['Study Hours', 'Attendance', 'Mobile Use']
                values = [
                    cluster_data['hours-study-per-day-average'].mean(),
                    cluster_data['average-class-attendance'].mean(),
                    cluster_data['hours-per-day-use-mobile phone'].mean()
                ]
                
                fig = go.Figure(data=[
                    go.Bar(x=metrics, y=values, 
                          marker_color=px.colors.qualitative.Set1[cluster_id % len(px.colors.qualitative.Set1)])
                ])
                
                fig.update_layout(
                    height=200,
                    title=f'Key Metrics - {cluster_name}',
                    showlegend=False,
                    yaxis_title="Average Value"
                )
                
                st.plotly_chart(fig, use_container_width=True, use_container_height=True)
    
    # Compare clusters
    st.markdown("### Cluster Comparison")
    
    # Calculate average values for each cluster
    cluster_stats = df.groupby(cluster_column)[features + ['GPA-current/previous-semester']].mean()
    
    # Create comparison heatmap
    fig = px.imshow(cluster_stats.T,
                   text_auto='.2f',
                   aspect="auto",
                   color_continuous_scale='RdBu',
                   title='Average Feature Values by Cluster')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="graph-explanation">Heatmap comparing average feature values across clusters. Helps identify defining characteristics of each cluster.</div>', unsafe_allow_html=True)

# Fixed Clustering Analysis Function
def perform_clustering_analysis(df):
    st.markdown('<div class="section-header">🔍 Clustering Analysis (Study Patterns)</div>', unsafe_allow_html=True)
    
    with st.expander("ℹ️ **About Clustering**"):
        st.markdown("""
        ### Why Clustering?
        
        **Purpose**: Group students with similar study habits and behaviors
        
        **Benefits**:
        - Identifies natural student groupings
        - Reveals hidden patterns in study behaviors
        - Helps create targeted intervention strategies
        - Useful for personalized academic advising
        
        **What we're clustering**: Study habits, digital behavior, attendance patterns
        """)
    
    # Select features for clustering
    clustering_features = ['hours-study-per-day-average', 
                          'hours-per-day-use-mobile phone',
                          'hours-sleep-per-night',
                          'average-class-attendance',
                          'hours-per-day-use-social-media-apps',
                          'academic-stress-level',
                          'motivation-level']
    
    # Get available features
    features = [f for f in clustering_features if f in df.columns]
    
    X = df[features].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Elbow method using custom K-Means
    st.markdown("### Determine Optimal Number of Clusters")
    
    wcss = []
    
    # Dynamic K range based on the elbow plot
    k_range = range(1, 8)  # Keep original range for elbow plot
    
    for k in k_range:
        try:
            labels, centroids, inertia = custom_kmeans(X_scaled, n_clusters=k)
            wcss.append(inertia)
        except Exception as e:
            st.warning(f"Error with k={k}: {str(e)}")
            wcss.append(0)
    
    # Plot elbow curve
    fig = px.line(x=list(k_range), y=wcss, 
                 title='Elbow Method for Optimal K',
                 labels={'x': 'Number of Clusters (K)', 'y': 'Within-Cluster Sum of Squares (WCSS)'},
                 markers=True)
    
    # Find elbow point (where slope changes significantly)
    max_clusters_for_slider = 6  # Default minimum
    
    if len(wcss) > 2:
        slopes = np.diff(wcss)
        elbow_point = np.argmax(slopes) + 1 if len(slopes) > 0 else 2
        
        # Dynamic max clusters: elbow_point + 2, but max 8
        max_clusters_for_slider = min(elbow_point + 2, 8)
        # Ensure at least 6 for backward compatibility
        max_clusters_for_slider = max(max_clusters_for_slider, 6)
        
        fig.add_vline(x=elbow_point+1, line_dash="dash", line_color="red", 
                     annotation_text=f"Suggested: K={elbow_point+1}")
    
    fig.update_traces(line=dict(color='#3B82F6', width=3))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="graph-explanation">Elbow plot helps determine optimal number of clusters. Look for the "elbow" point where adding more clusters doesn\'t significantly reduce WCSS.</div>', unsafe_allow_html=True)
    
    # Let user select number of clusters with dynamic max
    n_clusters = st.slider("Select number of clusters:", 2, max_clusters_for_slider, 3)
    
    # Perform custom K-Means clustering
    labels, centroids, inertia = custom_kmeans(X_scaled, n_clusters=n_clusters)
    df['Cluster'] = labels
    
    # Visualize clusters
    visualize_clusters(df, X_scaled, 'Cluster', features, n_clusters)
    
    return df, None, scaler, features
# Grade Predictor Function
def create_grade_predictor(results, best_model_name, features, scaler, le_dict):
    st.markdown('<div class="section-header">🔮 Grade Band Predictor</div>', unsafe_allow_html=True)
    
    best_model = results[best_model_name]['model']
    
    with st.expander("ℹ️ **How predictions work**"):
        st.markdown("""
        ### Prediction Process:
        
        1. **Input Collection**: User provides study habits and behavioral data
        2. **Preprocessing**: Data is scaled using the same method as training
        3. **Prediction**: Model predicts grade band (A+ to C)
        4. **Probability**: Shows confidence for each possible grade band
        5. **Recommendations**: Provides personalized suggestions based on prediction
        
        **Note**: Model accuracy depends on training data quality
        """)
    
    # Create input form
    with st.form("prediction_form"):
        st.markdown("### Enter Student Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Academic Information")
            current_semester = st.slider("Current Semester", 1, 8, 4)
            study_hours = st.slider("Study Hours per Day", 0.5, 12.0, 3.0, 0.5)
            attendance = st.slider("Class Attendance %", 50.0, 100.0, 85.0, 5.0)
        
        with col2:
            st.markdown("#### Daily Habits")
            mobile_hours = st.slider("Mobile Usage (hours/day)", 0.0, 10.0, 2.0, 0.5)
            sleep_hours = st.slider("Sleep Hours per Night", 4.0, 12.0, 7.0, 0.5)
            social_media = st.slider("Social Media (hours/day)", 0.0, 8.0, 2.0, 0.5)
        
        with col3:
            st.markdown("#### Psychological Factors")
            stress_level = st.slider("Academic Stress (0-5)", 0, 5, 2, 1,
                                   help="0 = No stress, 5 = Very high stress")
            motivation = st.slider("Motivation Level (1-5)", 1, 5, 3, 1,
                                 help="1 = Low motivation, 5 = Very motivated")
            
            # Add categorical inputs if available
            if 'use-mobile-during-study-sessions' in le_dict:
                mobile_options = list(le_dict['use-mobile-during-study-sessions'].classes_)
                use_mobile = st.selectbox("Use Mobile During Study", mobile_options)
        
        submitted = st.form_submit_button("🎯 Predict Grade Band")
    
    if submitted:
        # Prepare input data
        input_data = {}
        
        # Add numeric features
        numeric_features = {
            'current-semester': current_semester,
            'hours-study-per-day-average': study_hours,
            'hours-per-day-use-mobile phone': mobile_hours,
            'hours-sleep-per-night': sleep_hours,
            'average-class-attendance': attendance,
            'hours-per-day-use-social-media-apps': social_media,
            'academic-stress-level': stress_level,
            'motivation-level': motivation
        }
        
        for feature, value in numeric_features.items():
            if feature in features:
                input_data[feature] = value
        
        # Add categorical features
        if 'use-mobile-during-study-sessions' in le_dict and 'use-mobile-during-study-sessions' in features:
            input_data['use-mobile-during-study-sessions'] = le_dict['use-mobile-during-study-sessions'].transform([use_mobile])[0]
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Ensure all features are present
        for feature in features:
            if feature not in input_df.columns:
                input_df[feature] = 0  # Default value for missing features
        
        # Reorder columns to match training
        input_df = input_df[features]
        
        # Scale the input
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = best_model.predict(input_scaled)[0]
        probabilities = best_model.predict_proba(input_scaled)[0]
        
        # Display results
        st.markdown("### Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Display prediction with color coding
            color_map = {
                'A+': '#10B981',  # Green
                'A': '#3B82F6',   # Blue
                'B+': '#8B5CF6',  # Purple
                'B': '#F59E0B',   # Orange
                'C+': '#EF4444',  # Red
                'C': '#DC2626'    # Dark Red
            }
            
            prediction_color = color_map.get(prediction, '#6B7280')
            
            st.markdown(f"""
            <div style="background-color: {prediction_color}; 
                        color: white; padding: 2rem; border-radius: 10px; text-align: center; margin-bottom: 1rem;">
                <h2 style="margin: 0;">Predicted Grade Band</h2>
                <h1 style="font-size: 3rem; margin: 1rem 0;">{prediction}</h1>
                <p style="margin: 0; opacity: 0.9;">Based on provided study habits and behaviors</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show prediction confidence
            prediction_prob = probabilities[list(best_model.classes_).index(prediction)]
            st.metric("Prediction Confidence", f"{prediction_prob:.1%}")
        
        with col2:
            # Display probabilities as horizontal bar chart
            prob_df = pd.DataFrame({
                'Grade Band': best_model.classes_,
                'Probability': probabilities
            }).sort_values('Probability', ascending=False)
            
            fig = px.bar(prob_df, x='Probability', y='Grade Band',
                        title='Prediction Probabilities',
                        orientation='h',
                        color='Grade Band',
                        color_discrete_map=color_map)
            
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                xaxis_tickformat='.0%',
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Show recommendations
        st.markdown("### 📋 Recommendations")
        
        recommendations = {
            'A+': """
            **🎉 Excellent Performance!**
            - Maintain your current study routine
            - Consider becoming a peer tutor
            - Explore advanced topics in your field
            """,
            'A': """
            **👍 Very Good Performance!**
            - Continue with your effective habits
            - Focus on maintaining consistency
            - Set incremental improvement goals
            """,
            'B+': """
            **📚 Good Performance with Room for Improvement**
            - Increase study hours by 30 minutes daily
            - Reduce mobile usage during study sessions
            - Improve time management skills
            """,
            'B': """
            **📖 Average Performance - Consider Changes**
            - Aim for 3-4 hours of focused study daily
            - Attend all classes regularly
            - Seek help from professors during office hours
            """,
            'C+': """
            **⚠️ Below Average - Action Needed**
            - Increase study hours to 4-5 hours daily
            - Limit mobile usage to 1-2 hours/day
            - Attend academic skills workshops
            - Consider study group participation
            """,
            'C': """
            **🚨 Needs Significant Improvement - Urgent Action**
            - Seek academic counseling immediately
            - Develop structured study schedule
            - Reduce digital distractions significantly
            - Attend all classes without exception
            - Consider peer mentoring program
            """
        }
        
        st.info(recommendations.get(prediction, "No specific recommendations available."))
        
        # Risk assessment
        if prediction in ['C+', 'C']:
            st.warning("""
            ⚠️ **At-Risk Student Detected!**
            
            **Recommended Interventions:**
            1. Schedule meeting with academic advisor
            2. Enroll in study skills workshop
            3. Implement digital detox plan
            4. Regular progress monitoring
            """)

# Student Personas Function
def student_personas_analysis(df):
    st.markdown('<div class="section-header">👥 Student Persona Analysis</div>', unsafe_allow_html=True)
    
    # Create personas based on clustering if available
    if 'Cluster' in df.columns:
        clusters_exist = True
    else:
        clusters_exist = False
        # Create temporary clusters based on GPA for demonstration
        from sklearn.cluster import KMeans
        X = df[['hours-study-per-day-average', 'average-class-attendance', 'hours-per-day-use-mobile phone']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(X_scaled)
        clusters_exist = True
    
    if clusters_exist:
        # Define persona descriptions based on cluster characteristics
        personas = []
        
        for cluster_id in sorted(df['Cluster'].unique()):
            cluster_data = df[df['Cluster'] == cluster_id]
            avg_gpa = cluster_data['GPA-current/previous-semester'].mean()
            avg_study = cluster_data['hours-study-per-day-average'].mean()
            avg_mobile = cluster_data['hours-per-day-use-mobile phone'].mean()
            avg_attendance = cluster_data['average-class-attendance'].mean()
            
            # Determine persona type
            if avg_gpa >= 3.5 and avg_study >= 3 and avg_attendance >= 85:
                persona_type = "High Achiever"
                description = "Consistent, dedicated students with excellent performance"
            elif avg_gpa >= 3.0 and avg_mobile <= 3:
                persona_type = "Balanced Performer"
                description = "Students with good balance of study and leisure"
            elif avg_gpa < 2.5 or avg_attendance < 70:
                persona_type = "Struggling Student"
                description = "Students facing academic challenges needing support"
            elif avg_mobile > 4:
                persona_type = "Digitally Distracted"
                description = "Students whose academic performance is affected by high digital usage"
            else:
                persona_type = "Average Student"
                description = "Students with typical academic performance"
            
            personas.append({
                'id': cluster_id,
                'type': persona_type,
                'description': description,
                'data': cluster_data
            })
        
        # Display personas
        for persona in personas:
            cluster_data = persona['data']
            
            st.markdown(f"### 👤 {persona['type']} (Cluster {persona['id']})")
            st.markdown(f"*{persona['description']}*")
            
            # Key metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Students", f"{len(cluster_data)}", 
                         f"{len(cluster_data)/len(df)*100:.1f}%")
            
            with col2:
                st.metric("Avg GPA", f"{cluster_data['GPA-current/previous-semester'].mean():.2f}")
            
            with col3:
                st.metric("Avg Study Hours", f"{cluster_data['hours-study-per-day-average'].mean():.1f}")
            
            with col4:
                st.metric("Avg Attendance", f"{cluster_data['average-class-attendance'].mean():.1f}%")
            
            # Create radar chart for key characteristics
            metrics = ['Study Hours', 'Attendance', 'Sleep', 'Stress', 'Motivation']
            
            # Normalize values
            study_norm = cluster_data['hours-study-per-day-average'].mean() / df['hours-study-per-day-average'].max()
            attendance_norm = cluster_data['average-class-attendance'].mean() / 100
            sleep_norm = cluster_data['hours-sleep-per-night'].mean() / df['hours-sleep-per-night'].max()
            stress_norm = 1 - (cluster_data['academic-stress-level'].mean() / df['academic-stress-level'].max())
            motivation_norm = cluster_data['motivation-level'].mean() / df['motivation-level'].max()
            
            fig = go.Figure(data=go.Scatterpolar(
                r=[study_norm, attendance_norm, sleep_norm, stress_norm, motivation_norm],
                theta=metrics,
                fill='toself',
                name=persona['type']
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                height=300,
                title=f'Behavioral Profile: {persona["type"]}'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Grade distribution for this persona
            grade_dist = cluster_data['Grade_Band'].value_counts().sort_index()
            fig2 = px.bar(x=grade_dist.index, y=grade_dist.values,
                         title=f'Grade Distribution: {persona["type"]}',
                         labels={'x': 'Grade Band', 'y': 'Number of Students'},
                         color=grade_dist.values,
                         color_continuous_scale='Blues')
            
            st.plotly_chart(fig2, use_container_width=True)
            
            st.markdown("---")

# Main Application
def main():
    # Header
    st.markdown('<h1 class="main-header">📚 Student Performance Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("*Data-Driven Insights for Academic Success*")
    
    # Load data
    try:
        data = load_data()
        df, le_dict, categorical_cols = preprocess_data(data)
        
        # Create sidebar
        with st.sidebar:
            st.image("https://cdn-icons-png.flaticon.com/512/2232/2232688.png", width=100)
            st.markdown("## 📍 Navigation")
            
            section = st.radio(
                "Select Analysis Section:",
                ["📊 EDA & Analysis", "🎯 Classification Models", "🔍 Clustering Analysis", 
                 "🔮 Grade Predictor", "👥 Student Personas"]
            )
            
            st.markdown("---")
            st.markdown("### 📋 Dataset Info")
            st.write(f"**Total Students:** {len(df)}")
            st.write(f"**Avg GPA:** {df['GPA-current/previous-semester'].mean():.2f}")
            st.write(f"**Features:** {len(df.columns)}")
            
            st.markdown("---")
            st.markdown("### ⚡ Quick Stats")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Avg Study", f"{df['hours-study-per-day-average'].mean():.1f}h")
                st.metric("Avg Sleep", f"{df['hours-sleep-per-night'].mean():.1f}h")
            with col2:
                st.metric("Avg Mobile", f"{df['hours-per-day-use-mobile phone'].mean():.1f}h")
                st.metric("Avg Attend", f"{df['average-class-attendance'].mean():.0f}%")
        
        # Display selected section
        if section == "📊 EDA & Analysis":
            perform_eda(df)
            
        elif section == "🎯 Classification Models":
            results, best_model_name, features, scaler = train_classification_models(df, categorical_cols)
            # Store in session state for use in predictor
            st.session_state['results'] = results
            st.session_state['best_model_name'] = best_model_name
            st.session_state['features'] = features
            st.session_state['scaler'] = scaler
            st.session_state['le_dict'] = le_dict
            
        elif section == "🔍 Clustering Analysis":
            df_with_clusters, cluster_model, cluster_scaler, cluster_features = perform_clustering_analysis(df)
            # Store in session state
            if df_with_clusters is not None:
                st.session_state['df_with_clusters'] = df_with_clusters
                st.session_state['cluster_model'] = cluster_model
            
        elif section == "🔮 Grade Predictor":
            if 'results' in st.session_state:
                create_grade_predictor(
                    st.session_state['results'],
                    st.session_state['best_model_name'],
                    st.session_state['features'],
                    st.session_state['scaler'],
                    st.session_state['le_dict']
                )
            else:
                st.warning("⚠️ Please train classification models first!")
                if st.button("Go to Classification Models"):
                    st.rerun()
                    
        elif section == "👥 Student Personas":
            if 'df_with_clusters' in st.session_state:
                student_personas_analysis(st.session_state['df_with_clusters'])
            else:
                student_personas_analysis(df)
                
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure 'synthetic_survey_3000.csv' is in the same directory.")

if __name__ == "__main__":
    main()
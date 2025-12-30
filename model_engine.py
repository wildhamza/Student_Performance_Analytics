import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from utils import NUMERIC_FEATURES

def train_classification_models(df, categorical_cols):
    """
    Trains multiple classification models and evaluates performance.
    Returns results dictionary, best model name, feature list, and scaler.
    """
    features = [f for f in NUMERIC_FEATURES if f in df.columns]
    for cat_col in categorical_cols:
        if cat_col in df.columns:
            features.append(cat_col)
    
    X = df[features]
    y = df['Grade_Band']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Support Vector Machine': SVC(kernel='rbf', random_state=42, probability=True)
    }
    
    results = {}
    
    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        start_time = time.time()
        model.predict(X_test_scaled[:10])
        inference_time = (time.time() - start_time) / 10
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score'],
            'training_time': training_time,
            'inference_time': inference_time,
            'predictions': y_pred,
            'predictions_proba': y_pred_proba,
            'confusion_matrix': cm,
            'classification_report': report,
            'y_test': y_test
        }
    
    weights = {'accuracy': 0.3, 'precision': 0.2, 'recall': 0.2, 'f1_score': 0.2, 'speed': 0.1}
    max_inference_time = max([r['inference_time'] for r in results.values()])
    
    overall_scores = {}
    for name, res in results.items():
        speed_score = 1 - (res['inference_time'] / max_inference_time) if max_inference_time > 0 else 1
        score = (res['accuracy'] * weights['accuracy'] +
                 res['precision'] * weights['precision'] +
                 res['recall'] * weights['recall'] +
                 res['f1_score'] * weights['f1_score'] +
                 speed_score * weights['speed'])
        overall_scores[name] = score
    
    best_model_name = max(overall_scores, key=overall_scores.get)
    
    rf_model = results['Random Forest']['model']
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    return results, best_model_name, features, scaler, importance_df

def custom_kmeans(X, n_clusters=3, max_iter=100):
    """Custom implementation of the K-Means clustering algorithm."""
    np.random.seed(42)
    X = np.array(X, dtype=np.float64)
    
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        col_mean = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_mean, inds[1])
    
    n_samples, n_features = X.shape
    if n_samples == 0:
        return np.array([]), np.zeros((n_clusters, n_features)), 0
        
    centroids = X[np.random.choice(n_samples, min(n_clusters, n_samples), replace=False)]
    if n_clusters > n_samples:
        additional = n_clusters - n_samples
        extra_centroids = X[np.random.choice(n_samples, additional, replace=True)]
        centroids = np.vstack([centroids, extra_centroids])
    
    for _ in range(max_iter):
        distances = np.sqrt(((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=1)
        
        new_centroids = np.zeros_like(centroids)
        for i in range(n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)
            else:
                new_centroids[i] = X[np.random.choice(n_samples, 1)[0]]
        
        if np.allclose(centroids, new_centroids, rtol=1e-4):
            break
        centroids = new_centroids
    
    distances_final = np.sqrt(((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
    inertia = np.sum(np.min(distances_final, axis=1) ** 2)
    
    return labels, centroids, inertia

def hierarchical_clustering(X, n_clusters=3):
    """Performs Agglomerative Hierarchical Clustering."""
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X)
    return labels

def get_cluster_profiles(df, features):
    """Generates descriptive profiles for identified student clusters."""
    profiles = {}
    for cluster in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster]
        means = cluster_data[features].mean()
        
        desc = []
        if means.get('hours-study-per-day-average', 0) > df['hours-study-per-day-average'].mean():
            desc.append("High Study Effort")
        else:
            desc.append("Low Study Effort")
        
        if means.get('hours-sleep-per-night', 0) > df['hours-sleep-per-night'].mean():
            desc.append("Healthy Sleep Patterns")
        
        if means.get('academic-stress-level', 0) > df['academic-stress-level'].mean():
            desc.append("High Academic Stress")
            
        profiles[cluster] = " - ".join(desc)
    return profiles

def get_prediction_recommendations(prediction):
    """Provides academic recommendations based on predicted performance."""
    recommendations = {
        'High': "**Strong Performance**\n- Keep up the good work and maintain your routine.\n- Consider peer mentoring to solidify your knowledge.\n- Explore advanced electives or certifications.",
        'Medium': "**Good Performance**\n- You are on the right track but there is room for growth.\n- Try to identify the 'distraction factors' at the top of your importance chart.\n- Focused study sessions of 2-3 hours can boost results.",
        'Low': "**Action Recommended**\n- Your habits suggest a higher risk of academic struggle.\n- Consult with an academic advisor to plan your semester.\n- Prioritize sleep and reduce digital distractions significantly."
    }
    return recommendations.get(prediction, "No specific recommendations available.")

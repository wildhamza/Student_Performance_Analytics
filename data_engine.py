import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from utils import create_grade_bands, DEPARTMENTS_TO_REMOVE, OUTLIER_COLUMNS

@st.cache_data
def load_data(file_path='synthetic_survey_3000.csv'):
    """
    Load student performance data from CSV.
    """
    data = pd.read_csv(file_path)
    return data

@st.cache_data
def preprocess_data(data):
    """
    Perform data cleansing, outlier removal, and categorical encoding.
    Returns processed dataframe, encoder dictionary, feature columns, and summary metrics.
    """
    df = data.copy()
    initial_count = len(df)
    summary = {}
    
    # Filtering by department
    if 'department' in df.columns:
        df = df[~df['department'].isin(DEPARTMENTS_TO_REMOVE)]
        summary['removed_dept'] = initial_count - len(df)
    
    # Duplicate removal
    before_duplicates = len(df)
    df = df.drop_duplicates()
    summary['removed_duplicates'] = before_duplicates - len(df)
    
    # Missing value handling
    before_missing = len(df)
    df = df.dropna()
    summary['removed_missing'] = before_missing - len(df)
    
    # Outlier removal using IQR method
    def remove_outliers_iqr(df_in, column):
        Q1 = df_in[column].quantile(0.25)
        Q3 = df_in[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df_in[(df_in[column] >= lower_bound) & (df_in[column] <= upper_bound)]
    
    before_outliers = len(df)
    for col in OUTLIER_COLUMNS:
        if col in df.columns:
            df = remove_outliers_iqr(df, col)
    summary['removed_outliers'] = before_outliers - len(df)
    
    summary['final_count'] = len(df)
    summary['total_removed'] = initial_count - len(df)
    
    # Generate target variable
    df['Grade_Band'] = df['GPA-current/previous-semester'].apply(create_grade_bands)
    
    # Feature selection for categorical encoding
    categorical_cols = []
    for col in df.columns:
        if col not in ['Grade_Band', 'GPA-current/previous-semester', 'expected-GPA-this-course/semester']:
            if df[col].dtype == 'object' or df[col].nunique() < 15:
                categorical_cols.append(col)
    
    # Label Encoding for categorical features
    le_dict = {}
    cols_to_drop = []
    for col in categorical_cols:
        if col in df.columns:
            try:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                le_dict[col] = le
            except Exception:
                cols_to_drop.append(col)
    
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        categorical_cols = [c for c in categorical_cols if c not in cols_to_drop]
    
    return df, le_dict, categorical_cols, summary

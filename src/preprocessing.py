import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def validate_schema(df):
    required_cols = {'user_id', 'listening_time', 'episode_length_pref', 
                     'genre', 'skip_rate', 'completion_rate', 'frequency'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

def preprocess_features(df):
    num_cols = ['listening_time', 'skip_rate', 'completion_rate']
    cat_cols = ['episode_length_pref', 'genre', 'frequency']
    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ])
    
    X_processed = preprocessor.fit_transform(df)
    return X_processed
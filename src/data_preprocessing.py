import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Load dataset and ensure numeric data."""
    df = pd.read_csv(filepath)
    
    # Convert all columns to numeric, coercing errors to NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    
    return df

def handle_missing_values(df):
    """Handle missing values by filling with median or zero if entire column is NaN."""
    for col in df.columns:
        if df[col].isna().all():  # If the entire column is NaN, replace with 0
            df[col] = 0
        else:
            df[col].fillna(df[col].median(), inplace=True)  # Fill missing values with median
    return df

def normalize_features(df):
    """Normalize spectral reflectance features."""
    if df.shape[0] == 0:
        raise ValueError("No data available after preprocessing. Check missing values and anomalies handling.")
    
    scaler = StandardScaler()
    df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])  # Normalize all but the target
    return df, scaler

def detect_anomalies(df):
    """Detect anomalies using the interquartile range (IQR) method, but keep at least 1 row."""
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    mask = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
    df_filtered = df[~mask.any(axis=1)]
    
    if df_filtered.shape[0] == 0:
        print("Warning: All rows were removed as anomalies. Keeping original data.")
        return df  # Keep original data if everything is removed
    
    return df_filtered

def preprocess_data(filepath):
    """Complete preprocessing pipeline."""
    df = load_data(filepath)
    df = handle_missing_values(df)  # Ensure no NaN values exist before training
    df = detect_anomalies(df)
    df, scaler = normalize_features(df)
    
    # Final check for NaN values before saving
    if df.isna().sum().sum() > 0:
        print("Warning: Processed dataset still contains NaN values. Replacing with zero.")
        df = df.fillna(0)  # Final safeguard to remove NaN values
    
    return df, scaler

if __name__ == "__main__":
    input_file = "/Users/apple/DON_Concentration/data/MLE-Assignment.csv"
    processed_df, scaler = preprocess_data(input_file)
    processed_df.to_csv("/Users/apple/DON_Concentration/data/processed_data.csv", index=False)
    print("Preprocessing complete. Processed data saved.")
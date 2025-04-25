import pandas as pd
import numpy as np

def clean_data(df):
    """Clean and preprocess the healthcare dataset"""
    # Convert column names to lowercase and strip spaces
    df.columns = df.columns.str.lower().str.strip()
    
    # Clean age and convert to numeric
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    
    # Handle missing values
    df = df.dropna()
    
    return df

def calculate_egfr(age, gender, creatinine):
    """Calculate eGFR using CKD-EPI equation"""
    k = 0.7 if gender == 'female' else 0.9
    a = -0.241 if gender == 'female' else -0.302
    
    egfr = 142 * min(creatinine/k, 1)**a * max(creatinine/k, 1)**-1.200 * 0.9938**age
    if gender == 'female':
        egfr *= 1.012
    
    return egfr

def classify_ckd(egfr):
    """Classify CKD stages based on eGFR value"""
    if egfr >= 90:
        return 'Stage 1'
    elif 60 <= egfr < 90:
        return 'Stage 2'
    elif 30 <= egfr < 60:
        return 'Stage 3'
    elif 15 <= egfr < 30:
        return 'Stage 4'
    else:
        return 'Stage 5'
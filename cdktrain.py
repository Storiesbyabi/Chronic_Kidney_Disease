import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score

def train_model():
    # Load dataset
    file_path = "./download.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError("Dataset file not found!")
        
    df = pd.read_csv(file_path)
    print("\n✅ Dataset loaded successfully!")
    
    # Clean and prepare data
    df.columns = ['id', 'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
                  'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
                  'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'classification']
    
    # Handle missing values
    df = df.replace('?', np.nan)
    df = df.replace('\t?', np.nan)
    
    # Convert numeric columns
    numeric_columns = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 
                      'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Select features for model
    selected_features = [
        'age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 
        'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
        'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
    ]
    
    # Prepare features and target
    X = df[selected_features]
    y = df['classification']
    
    # Handle categorical variables
    categorical_features = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    label_encoders = {}
    feature_classes = {}
    
    for col in categorical_features:
        # Get unique values and store them
        unique_values = X[col].dropna().unique()
        feature_classes[col] = list(unique_values)
        
        # Encode the feature
        label_encoders[col] = LabelEncoder()
        X[col] = label_encoders[col].fit_transform(X[col].fillna('missing'))
    
    # Fill missing numeric values with median
    for col in numeric_columns:
        X[col] = X[col].fillna(X[col].median())
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale numeric features
    scaler = StandardScaler()
    X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
    X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])
    
    # Train model
    model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model and preprocessors
    joblib.dump(model, "ckd_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(label_encoders, "label_encoders.pkl")
    joblib.dump(feature_classes, "feature_classes.pkl")
    joblib.dump(selected_features, "selected_features.pkl")
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

if __name__ == "__main__":
    accuracy = train_model()
    print(f"\n🎯 Model Accuracy: {accuracy:.2f}")
    print("\n💾 Model and preprocessors saved successfully!")
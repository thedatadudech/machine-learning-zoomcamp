import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import bentoml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    """Load the Heart Disease dataset."""
    try:
        # Using a more reliable source for the heart disease dataset
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                       'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        df = pd.read_csv(url, names=column_names, na_values='?')
        logging.info(f"Successfully loaded data with shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def preprocess_data(df):
    """Clean and preprocess the data."""
    # Handle missing values if any
    df = df.dropna()
    logging.info(f"Shape after dropping NA: {df.shape}")
    
    X = df.drop('target', axis=1)
    y = (df['target'] > 0).astype(int)  # Convert to binary classification
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, X.columns.tolist()

def train_model():
    """Train and save the heart disease prediction model."""
    logging.info("Starting model training process...")
    
    try:
        # Load and preprocess data
        df = load_data()
        X_scaled, y, scaler, feature_names = preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        logging.info(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        logging.info(f"Train accuracy: {train_score:.3f}")
        logging.info(f"Test accuracy: {test_score:.3f}")
        
        # Save model with BentoML
        model_service = bentoml.sklearn.save_model(
            "heart_disease_classifier",
            model,
            custom_objects={
                "scaler": scaler,
                "feature_names": feature_names
            },
            signatures={
                "predict": {"batchable": True},
                "predict_proba": {"batchable": True}
            }
        )
        logging.info(f"Model saved successfully with tag: {model_service.tag}")
        return model_service.tag
        
    except Exception as e:
        logging.error(f"Error in training process: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        tag = train_model()
        logging.info(f"Training completed successfully. Model tag: {tag}")
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

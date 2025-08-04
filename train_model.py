# train_model.py - MBTI Personality Predictor Training Script

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(file_path):
    """Load and preprocess the MBTI dataset"""
    print("Loading dataset...")
    df = pd.read_csv(file_path, encoding='Windows-1252')
    
    # Display basic info
    print(f"Dataset shape: {df.shape}")
    print(f"Personality types distribution:")
    print(df['Personality'].value_counts().sort_index())
    
    # Separate features and target
    X = df.drop(['Response Id', 'Personality'], axis=1)
    y = df['Personality']
    
    print(f"Features shape: {X.shape}")
    print(f"Number of unique personality types: {y.nunique()}")
    
    return X, y, df

def train_model(X, y):
    """Train the Random Forest model"""
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, X_test, y_test, y_pred

def analyze_feature_importance(model, feature_names):
    """Analyze and plot feature importance"""
    print("\nAnalyzing feature importance...")
    
    # Get feature importance
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("Top 15 most important features:")
    print(feature_importance_df.head(15))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance_df.head(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Most Important Features for MBTI Prediction')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return feature_importance_df

def plot_confusion_matrix(y_test, y_pred):
    """Plot confusion matrix"""
    plt.figure(figsize=(14, 12))
    cm = confusion_matrix(y_test, y_pred)
    
    # Get unique labels in sorted order
    labels = sorted(y_test.unique())
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix - MBTI Personality Prediction')
    plt.xlabel('Predicted Personality Type')
    plt.ylabel('Actual Personality Type')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_model_and_features(model, feature_names):
    """Save the trained model and feature names"""
    print("\nSaving model...")
    
    # Save the model
    joblib.dump(model, 'mbti_model.pkl', compress=3)
    
    # Save feature names for later use
    with open('feature_names.txt', 'w', encoding='utf-8') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")
    
    print("Model saved as 'mbti_model.pkl'")
    print("Feature names saved as 'feature_names.txt'")

def main():
    """Main training pipeline"""
    # Load and preprocess data
    X, y, df = load_and_preprocess_data('16P.csv')
    
    # Train model
    model, X_test, y_test, y_pred = train_model(X, y)
    
    # Analyze feature importance
    feature_importance_df = analyze_feature_importance(model, X.columns)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)
    
    # Save model and features
    save_model_and_features(model, X.columns)
    
    print("\nTraining completed successfully!")
    print("Files created:")
    print("- mbti_model.pkl (trained model)")
    print("- feature_names.txt (feature names)")
    print("- feature_importance.png (feature importance plot)")
    print("- confusion_matrix.png (confusion matrix plot)")

if __name__ == "__main__":
    main()
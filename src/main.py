"""
Telco Customer Churn Prediction - Main Pipeline
Author: Obasi-Uzoma Blessing
Date: January 2026
Description: End-to-end machine learning pipeline for predicting customer churn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set(style="whitegrid")
plt.style.use('seaborn-v0_8-darkgrid')

# ============================================================================
# 1. DATA LOADING & PREPROCESSING
# ============================================================================

def load_and_clean_data(filepath):
    """
    Load raw data and perform initial cleaning
    
    Parameters:
    -----------
    filepath : str
        Path to the raw CSV file
    
    Returns:
    --------
    pd.DataFrame : Cleaned dataframe
    """
    print("="*70)
    print("STEP 1: Loading and Cleaning Data")
    print("="*70)
    
    # Load data
    data = pd.read_csv(r'c:\Users\obasi\Downloads\SP 500 Stock Prices 2014-2017.csv\WA_Fn-UseC_-Telco-Customer-Churn 2.csv', encoding= 'latin1')
    print(f"Loaded {len(data)} records from {r'c:\Users\obasi\Downloads\SP 500 Stock Prices 2014-2017.csv\WA_Fn-UseC_-Telco-Customer-Churn 2.csv'}")
    
    # Handle TotalCharges: Convert to numeric, fill missing with 0
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce').fillna(0)
    print(f"Cleaned TotalCharges column (filled {data['TotalCharges'].isna().sum()} missing values)")
    
    # Encode Target Variable
    data['Churn_Numeric'] = data['Churn'].map({'Yes': 1, 'No': 0})
    print(f"Encoded target variable: {data['Churn_Numeric'].value_counts().to_dict()}")
    
    # Save cleaned data
    data.to_csv('data/processed/cleaned_churn_data.csv', index=False)
    print("Saved cleaned data to 'data/processed/cleaned_churn_data.csv'")
    
    return data

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================

def engineer_features(data):
    """
    Create new features to improve model performance
    
    Parameters:
    -----------
    data : pd.DataFrame
        Cleaned dataframe
    
    Returns:
    --------
    pd.DataFrame : Dataframe with engineered features
    """
    print("\n" + "="*70)
    print("STEP 2: Feature Engineering")
    print("="*70)
    
    # A. Total Services (Engagement Depth)
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['TotalServices'] = (data[services] == 'Yes').sum(axis=1)
    print("Created 'TotalServices' feature (customer engagement metric)")
    
    # B. Average Cost Per Service (Value Perception)
    data['AvgCostPerService'] = data['MonthlyCharges'] / (data['TotalServices'] + 1)
    print("Created 'AvgCostPerService' feature (value perception)")
    
    # C. Tenure Grouping (Lifecycle Stage)
    def tenure_group(t):
        if t <= 12: return '0-1 Year'
        elif t <= 24: return '1-2 Years'
        elif t <= 48: return '2-4 Years'
        else: return '4+ Years'
    
    data['TenureGroup'] = data['tenure'].apply(tenure_group)
    print("Created 'TenureGroup' feature (lifecycle segmentation)")
    
    # D. High Risk Flag: Month-to-Month + Fiber Optic
    data['Risk_FiberM2M'] = ((data['InternetService'] == 'Fiber optic') & 
                           (data['Contract'] == 'Month-to-month')).astype(int)
    print("Created 'Risk_FiberM2M' feature (high-risk segment identifier)")
    
    # E. Payment Method: Automatic vs Manual
    data['IsAutomaticPayment'] = data['PaymentMethod'].str.contains('automatic').astype(int)
    print("Created 'IsAutomaticPayment' feature (payment convenience)")
    
    # Save feature-engineered data
    data.to_csv('data/processed/feature_engineered_churn_data.csv', index=False)
    print("Saved feature-engineered data to 'data/processed/feature_engineered_churn_data.csv'")
    
    return data

# ============================================================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================================================

def perform_eda(data):
    """
    Generate exploratory data analysis visualizations
    
    Parameters:
    -----------
    data : pd.DataFrame
        Feature-engineered dataframe
    """
    print("\n" + "="*70)
    print("STEP 3: Exploratory Data Analysis")
    print("="*70)
    
    # Plot 1: Churn Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Churn', data=data, palette='viridis')
    plt.title('Distribution of Customer Churn', fontsize=14, fontweight='bold')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('visualizations/eda/churn_distribution.png', dpi=300)
    plt.close()
    print("Generated: churn_distribution.png")
    
    # Plot 2: Tenure Density by Churn
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=data, x='tenure', hue='Churn', fill=True, palette='crest', alpha=0.6)
    plt.title('Customer Tenure Distribution by Churn Status', fontsize=14, fontweight='bold')
    plt.xlabel('Tenure (months)')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig('visualizations/eda/tenure_density.png', dpi=300)
    plt.close()
    print("✓ Generated: tenure_density.png")
    
    # Plot 3: Churn by Contract Type
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Contract', hue='Churn', data=data, palette='Set2')
    plt.title('Churn Rates by Contract Type', fontsize=14, fontweight='bold')
    plt.ylabel('Count')
    plt.legend(title='Churn', loc='upper right')
    plt.tight_layout()
    plt.savefig('visualizations/eda/contract_type_churn.png', dpi=300)
    plt.close()
    print("Generated: contract_type_churn.png")
    
    # Plot 4: Service Density Impact
    plt.figure(figsize=(8, 5))
    sns.barplot(x='TotalServices', y='Churn_Numeric', data=data, palette='magma', ci=None)
    plt.title('Churn Probability by Number of Services', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Services')
    plt.ylabel('Churn Probability')
    plt.tight_layout()
    plt.savefig('visualizations/eda/service_density_impact.png', dpi=300)
    plt.close()
    print("Generated: service_density_impact.png")
    
    # Plot 5: Correlation Heatmap
    plt.figure(figsize=(12, 10))
    corr_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'TotalServices', 
                 'AvgCostPerService', 'Risk_FiberM2M', 'IsAutomaticPayment', 'Churn_Numeric']
    corr_matrix = data[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                vmin=-1, vmax=1, center=0, square=True, linewidths=1)
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualizations/eda/correlation_heatmap.png', dpi=300)
    plt.close()
    print("Generated: correlation_heatmap.png")

# ============================================================================
# 4. MODEL TRAINING
# ============================================================================

def train_models(data):
    """
    Train and evaluate multiple classification models
    
    Parameters:
    -----------
    data : pd.DataFrame
        Feature-engineered dataframe
    
    Returns:
    --------
    tuple : (models_dict, X_test, y_test, predictions_dict)
    """
    print("\n" + "="*70)
    print("STEP 4: Model Training")
    print("="*70)
    
    # Prepare features and target
    X_raw = data.drop(columns=['customerID', 'Churn', 'Churn_Numeric', 'TenureGroup'])
    y = data['Churn_Numeric']
    
    # One-Hot Encoding
    X_encoded = pd.get_dummies(X_raw, drop_first=True)
    print(f"Features prepared: {X_encoded.shape[1]} features after encoding")
    
    # Train-Test Split (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data split: {len(X_train)} training, {len(X_test)} testing samples")
    
    # Scaling (for Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Features scaled using StandardScaler")
    
    # Model A: Logistic Regression
    print("\n→ Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
    print("Logistic Regression trained")
    
    # Model B: Random Forest
    print("→ Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=None,  # Let trees grow naturally
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    print("Random Forest trained")
    
    # Save models
    joblib.dump(lr, 'models/logistic_regression_model.pkl')
    joblib.dump(rf, 'models/random_forest_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("\nModels saved to 'models/' directory")
    
    models = {
        'Logistic Regression': lr,
        'Random Forest': rf
    }
    
    predictions = {
        'lr': {'pred': y_pred_lr, 'prob': y_prob_lr},
        'rf': {'pred': y_pred_rf, 'prob': y_prob_rf}
    }
    
    return models, X_test, y_test, predictions, X_encoded.columns

# ============================================================================
# 5. MODEL EVALUATION
# ============================================================================

def evaluate_models(y_test, predictions, feature_names, models):
    """
    Evaluate and visualize model performance
    
    Parameters:
    -----------
    y_test : pd.Series
        True labels
    predictions : dict
        Dictionary containing predictions from both models
    feature_names : list
        List of feature names
    models : dict
        Dictionary of trained models
    """
    print("\n" + "="*70)
    print("STEP 5: Model Evaluation")
    print("="*70)
    
    # Print Classification Reports
    print("\n" + "-"*70)
    print("LOGISTIC REGRESSION - Classification Report")
    print("-"*70)
    print(classification_report(y_test, predictions['lr']['pred'], 
                                target_names=['No Churn', 'Churn']))
    
    print("\n" + "-"*70)
    print("RANDOM FOREST - Classification Report")
    print("-"*70)
    print(classification_report(y_test, predictions['rf']['pred'], 
                                target_names=['No Churn', 'Churn']))
    
    # Confusion Matrix (Logistic Regression)
    cm = confusion_matrix(y_test, predictions['lr']['pred'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix - Logistic Regression', fontsize=14, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('visualizations/evaluation/confusion_matrix.png', dpi=300)
    plt.close()
    print("\nGenerated: confusion_matrix.png")
    
    # Feature Importance (Random Forest)
    feat_imp = pd.Series(models['Random Forest'].feature_importances_, 
                        index=feature_names).sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    feat_imp.plot(kind='barh', color='#4c72b0')
    plt.title('Top 10 Feature Importances - Random Forest', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('visualizations/evaluation/feature_importance.png', dpi=300)
    plt.close()
    print("Generated: feature_importance.png")
    
    # ROC Curve Comparison
    fpr_lr, tpr_lr, _ = roc_curve(y_test, predictions['lr']['prob'])
    fpr_rf, tpr_rf, _ = roc_curve(y_test, predictions['rf']['prob'])
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {roc_auc_score(y_test, predictions['lr']['prob']):.3f})", 
             linewidth=2)
    plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_auc_score(y_test, predictions['rf']['prob']):.3f})", 
             linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/evaluation/roc_curve_comparison.png', dpi=300)
    plt.close()
    print("Generated: roc_curve_comparison.png")

# ============================================================================
# 6. EXPORT RESULTS
# ============================================================================

def export_results(data, X_test, y_test, predictions):
    """
    Export final predictions for further analysis
    
    Parameters:
    -----------
    data : pd.DataFrame
        Original dataframe
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        True labels
    predictions : dict
        Model predictions
    """
    print("\n" + "="*70)
    print("STEP 6: Exporting Results")
    print("="*70)
    
    results_df = data.loc[X_test.index].copy()
    results_df['Actual_Churn'] = y_test
    results_df['Pred_Prob_LR'] = predictions['lr']['prob']
    results_df['Pred_Label_LR'] = predictions['lr']['pred']
    results_df['Pred_Prob_RF'] = predictions['rf']['prob']
    results_df['Pred_Label_RF'] = predictions['rf']['pred']
    
    results_df.to_csv('data/results/final_churn_prediction_analysis.csv', index=False)
    print("✓ Predictions exported to 'data/results/final_churn_prediction_analysis.csv'")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Execute the complete churn prediction pipeline
    """
    print("\n")
    print("="*70)
    print(" "*15 + "TELCO CUSTOMER CHURN PREDICTION")
    print(" "*20 + "Machine Learning Pipeline")
    print("="*70)
    
    # Execute pipeline
    data = load_and_clean_data('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    data = engineer_features(data)
    perform_eda(data)
    models, X_test, y_test, predictions, feature_names = train_models(data)
    evaluate_models(y_test, predictions, feature_names, models)
    export_results(df, X_test, y_test, predictions)
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated files:")
    print("  • Cleaned data: data/processed/cleaned_churn_data.csv")
    print("  • Engineered features: data/processed/feature_engineered_churn_data.csv")
    print("  • Visualizations: visualizations/eda/ and visualizations/evaluation/")
    print("  • Trained models: models/")
    print("  • Final predictions: data/results/final_churn_prediction_analysis.csv")
    print("\n")

if __name__ == "__main__":
    main()

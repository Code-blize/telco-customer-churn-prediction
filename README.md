# Telco Customer Churn Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning project to predict customer churn in the telecommunications industry using Logistic Regression and Random Forest classifiers.

##  Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Key Features](#key-features)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

##  Overview

This project provides an end-to-end churn analysis for a telecommunications company, focusing on identifying why customers leave and building a predictive model to flag at-risk individuals.
Customer churn is a critical metric for subscription-based businesses. This project analyzes telecom customer data to:
- Identify key factors contributing to customer churn
- Build predictive models to forecast churn probability
- Provide actionable insights for customer retention strategies


**Problem Framing**

- **What is churn?** Churn occurs when a customer cancels their subscription or stops using the service. In this context, it is represented by the Churn column (Yes/No).

- **Why measure it?** It is significantly more expensive to acquire a new customer than to retain an existing one. High churn rates directly impact revenue stability and indicate potential issues with service quality, pricing, or customer satisfaction.


**Key Highlights:**
-  **Accuracy**: 80%+ on test set
-  **Models**: Logistic Regression & Random Forest
-  **Features**: 5 engineered features for enhanced prediction
-  **ROC-AUC**: 0.84+ for best model

##  Dataset

**Source**: https://drive.google.com/file/d/1763OlxZ9Fun9-x3GYi6BUu_7ot9AfEkJ/view?usp=sharing

**Description**: 
- **Rows**: 7,043 customers
- **Features**: 21 (demographic, account, and service information)
- **Target**: Churn (Yes/No)
- **Class Distribution**: ~26% churn rate

**Key Variables**:
- Customer demographics (gender, age, dependents)
- Service subscriptions (internet, phone, streaming)
- Account information (tenure, contract type, payment method)
- Billing (monthly charges, total charges)

##  Project Structure

```
telco-churn-prediction/
│
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   ├── processed/
│   │   ├── cleaned_churn_data.csv
│   │   └── feature_engineered_churn_data.csv
│   └── results/
│       └── final_churn_prediction_analysis.csv
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── evaluation.py
│
├── models/
│   ├── logistic_regression_model.pkl
│   └── random_forest_model.pkl
│
├── visualizations/
│   ├── eda/
│   │   ├── churn_distribution.png
│   │   ├── tenure_density.png
│   │   ├── contract_type_churn.png
│   │   ├── service_density_impact.png
│   │   └── correlation_heatmap.png
│   └── evaluation/
│       ├── confusion_matrix.png
│       ├── feature_importance.png
│       └── roc_curve_comparison.png
│
├── reports/
│   └── project_report.pdf
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

##  Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/Code-blize/telco-churn-prediction.git
cd telco-churn-prediction
```

2. **Create a virtual environment** (recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Quick Start

Run the complete pipeline:
```bash
python src/main.py
```

### Step-by-Step Execution

1. **Data Preprocessing**
```bash
python src/data_preprocessing.py
```

2. **Feature Engineering**
```bash
python src/feature_engineering.py
```

3. **Model Training**
```bash
python src/model_training.py
```

4. **Evaluation**
```bash
python src/evaluation.py
```

### Using Jupyter Notebooks

Explore the analysis interactively:
```bash
jupyter notebook notebooks/
```

##  Methodology

### 1. Data Preprocessing
The dataset contains 7,043 customers and 21 features. Key cleaning steps performed:
- TotalCharges Fix: Converted TotalCharges from a string to a numeric float. Identified 11 missing values (representing customers with 0 tenure) and filled them with 0.
- Data Types: Validated that tenure and MonthlyCharges are numeric, while service details are categorical.
- Target Validation: The target variable Churn was found to be imbalanced, with ~26.5% of customers having churned.

### 2. Feature Engineering

Created 5 new features to enhance predictive power:

| Feature | Description | Rationale |
|---------|-------------|-----------|
| **TotalServices** | Count of subscribed services | Measures customer engagement depth |
| **AvgCostPerService** | Monthly charges / (services + 1) | Indicates value perception |
| **TenureGroup** | Categorized customer lifecycle stage | Captures non-linear tenure effects |
| **Risk_FiberM2M** | Fiber optic + Month-to-month contract | Identifies high-risk combination |
| **IsAutomaticPayment** | Binary flag for automatic payment | Payment convenience indicator |

### 3. Exploratory Data Analysis

Key insights discovered:
- Tenure & Churn: Most churn occurs within the first 12 months. Long-term customers (tenure > 60 months) are highly loyal.
- Contract Types: Customers on Month-to-month contracts are significantly more likely to churn compared to those on One-year or Two-year contracts.
- Service Impact: Customers with Fiber optic internet service show higher churn rates than those with DSL, suggesting a potential gap in pricing or service reliability.
- Payment Method: Electronic Check users have the highest churn rate among all payment methods.

### 4. Modeling Approach

**Data Split**: 80% training, 20% testing (stratified)

**Models Trained**:
1. **Logistic Regression** (Baseline)
   - Interpretable linear model
   - Standardized features using StandardScaler
   - Provides probability estimates

2. **Random Forest** (Ensemble)
   - Handles non-linear relationships
   - 100 trees, max depth = 8
   - Robust to outliers

**Evaluation Metrics**:
- Confusion Matrix
- Precision, Recall, F1-Score
- ROC-AUC Score
- Feature Importance (for RF)

##  Results

### Model Performance

| Metric              | Logistic Regression | Random Forest |
|---------------------|---------------------|---------------|
| Accuracy            | 81%                 | 78%           |
| ROC-AUC             | 0.84                | 0.82          |
| Recall (Churn)      | 56%                 | 50%           |
| Precision (Churn)   | 66%                 | 61%           |

**Model Choice:** Logistic Regression performed slightly better in terms of both ROC-AUC and Recall. In churn prediction, Recall is critical because we want to capture as many potential "churners" as possible, even at the cost of some false alarms.
Logistic Regression achieved superior performance across all evaluation metrics, including accuracy (81%), ROC-AUC (0.84), and churn recall (56%). This indicates better overall discrimination and a stronger ability to identify customers at risk of churn compared to Random Forest.

### Key Findings

**Top 5 Churn Predictors** (from Random Forest):
1. Tenure (months with company)
2. Monthly Charges
3. Total Charges
4. Contract Type (Month-to-month)
5. Internet Service Type (Fiber optic)

**Business Recommendations**:
Based on the analysis, the company should implement the following strategies:
- Incentivize Long-term Contracts: Offer discounts or loyalty points for customers switching from month-to-month to 1-year plans. Month-to-month is the single biggest risk factor.
- Fiber Optic Audit: Investigate why Fiber Optic customers are churning more than DSL users despite the higher speed. Is the price point too high, or is the service unstable?
- Target Early-Stage Customers: Implement a "First 6 Months" engagement program. High-touch customer service during the first few months can reduce early tenure churn.
- Payment Method Migration: Encourage customers using "Electronic Check" to move to "Credit Card (automatic)" or "Bank Transfer" by offering a small one-time credit, as automatic payments correlate with higher retention.

### Visualizations

See the `visualizations/` folder for:
- Distribution plots
- Correlation heatmaps
- ROC curves
- Feature importance charts

##  Key Features

-  **Comprehensive EDA**: 5 detailed visualizations
-  **Feature Engineering**: 5 domain-driven features
-  **Multiple Models**: Logistic Regression & Random Forest
-  **Proper Validation**: Stratified train-test split, no data leakage
-  **Interpretability**: Feature importance and model coefficients
-  **Production Ready**: Saved models for deployment
-  **Reproducible**: Fixed random seeds, clear documentation

##  Future Improvements

- [ ] Implement cross-validation for more robust evaluation
- [ ] Address class imbalance using SMOTE or class weights
- [ ] Hyperparameter tuning using GridSearchCV
- [ ] Try advanced models (XGBoost, LightGBM)
- [ ] Create interactive dashboard (Streamlit/Dash)
- [ ] Deploy model as REST API (Flask/FastAPI)
- [ ] Add customer lifetime value (CLV) analysis
- [ ] Implement A/B testing framework for interventions

##  Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Obasi-Uzoma Blessing**
- GitHub: [@Code-blize](https://github.com/Code-blize)
- LinkedIn: [Obasi-Uzoma Blessing](https://linkedin.com/in/blessingobasiuzoma)
- Email: blessingobasiuzoma@gmail.com

##  Acknowledgments

- Dr. Okunola Orogon, PhD for providing the dataset
- Scikit-learn community for excellent ML tools
- All contributors who helped improve this project

---

⭐ If you found this project helpful, please consider giving it a star!

**Last Updated**: January 2026

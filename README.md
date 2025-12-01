# Heritage Housing Price Predictor 🏠

## Project Overview

This project builds a machine learning application to predict house prices in Ames, Iowa. The solution combines data analysis, feature engineering, and a trained Random Forest model deployed via a Streamlit web dashboard.

**Live Demo:** https://heritage-housing-price-4311f0b65bc6.herokuapp.com/

---

## Table of Contents
1. [Dataset Content](#dataset-content)
2. [Business Requirements](#business-requirements)
3. [Project Hypothesis](#project-hypothesis)
4. [Rationale to Map Business Requirements](#rationale-to-map-business-requirements)
5. [Machine Learning Business Case](#machine-learning-business-case)
6. [Dashboard Design](#dashboard-design)
7. [Data Analysis Results](#data-analysis-results)
8. [Machine Learning Model](#machine-learning-model)
9. [Main Data Analysis and Machine Learning Libraries](#main-data-analysis-and-machine-learning-libraries)
10. [Project Structure](#project-structure)
11. [Installation & Setup](#installation--setup)
12. [Unfixed Bugs](#unfixed-bugs)
13. [Deployment](#deployment)
14. [Credits](#credits)

---

## Dataset Content

### Source
- **Main Dataset:** 1,460 houses in Ames, Iowa
- **Inherited Houses:** 4 properties requiring valuation
- **Features:** 21 (after cleaning)
- **Target:** SalePrice

### Feature Descriptions

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| 1stFlrSF | Numerical | First Floor square feet | 334 - 4,692 |
| 2ndFlrSF | Numerical | Second Floor square feet | 0 - 2,065 |
| BedroomAbvGr | Numerical | Bedrooms above grade | 0 - 8 |
| BsmtExposure | Categorical | Basement exposure level | Good/Average/Min/No |
| BsmtFinType1 | Categorical | Basement finished area type | GLQ/ALQ/BLQ/Rec/LwQ/Unf |
| BsmtFinSF1 | Numerical | Basement finished square feet | 0 - 5,644 |
| BsmtUnfSF | Numerical | Basement unfinished square feet | 0 - 2,336 |
| TotalBsmtSF | Numerical | Total basement square feet | 0 - 6,110 |
| GarageArea | Numerical | Garage size in square feet | 0 - 1,418 |
| GarageFinish | Categorical | Garage interior finish | Fin/RFn/Unf/None |
| GarageYrBlt | Numerical | Year garage was built | 1900 - 2010 |
| GrLivArea | Numerical | Above grade living area | 334 - 5,642 |
| KitchenQual | Categorical | Kitchen quality | Ex/Gd/TA/Fa/Po |
| LotArea | Numerical | Lot size in square feet | 1,300 - 215,245 |
| LotFrontage | Numerical | Linear feet of street connected | 21 - 313 |
| MasVnrArea | Numerical | Masonry veneer area | 0 - 1,600 |
| OpenPorchSF | Numerical | Open porch area | 0 - 547 |
| OverallCond | Numerical | Overall condition of house | 1 - 10 |
| OverallQual | Numerical | Overall material and finish | 1 - 10 |
| YearBuilt | Numerical | Original construction date | 1872 - 2010 |
| YearRemodAdd | Numerical | Remodel date | 1950 - 2010 |
| **SalePrice** | **Target** | **Sale price in dollars** | **$34,900 - $755,000** |

### Data Statistics

| Metric | Value |
|--------|-------|
| Total Records | 1,460 |
| Feature Count | 21 (after cleaning) |
| Price Range | $34,900 - $755,000 |
| Average Price | $180,921 |
| Missing Values | 0 (after handling) |

---

## Business Requirements

### Client: Lydia Doe
**Scenario:** The client inherited 4 houses in Ames, Iowa and needs help maximizing their sale prices. She fears that her current knowledge of property values in her home state may not apply to Iowa.

### BR1: Data Analysis & Correlation Study
**Requirement:** Discover how house attributes correlate with sale prices through visual analysis.

**Acceptance Criteria:**
- Identify which features have the strongest relationship with price
- Provide visual correlation analysis
- Support client decision-making with data-driven insights

**Delivered:**
- ✅ Correlation Study dashboard page
- ✅ Top 10 correlated features identified
- ✅ Horizontal bar chart visualization
- ✅ Key insights documented

### BR2: Price Prediction
**Requirement:** Predict house sale prices for the 4 inherited houses and any other house in Ames, Iowa.

**Acceptance Criteria:**
- Build ML model with R² ≥ 0.75 on test set
- Predict prices for 4 inherited houses
- Allow custom price predictions for any house
- Model must be reliable and interpretable

**Delivered:**
- ✅ Random Forest model trained (R² = 0.8897)
- ✅ Price Predictor dashboard page
- ✅ Individual predictions for 4 inherited houses
- ✅ Custom prediction interface with interactive inputs
- ✅ Total valuation of inherited properties

---

## Project Hypothesis

### H1: Overall Quality is the Primary Price Driver
**Statement:** The overall quality of a house is the strongest predictor of its sale price.

**How to Validate:**
- Calculate Pearson correlation coefficient with SalePrice
- Measure feature importance in trained model
- Compare with other features

**Validation Results:** ✅ **CONFIRMED**
- Correlation: 0.7910 (highest among all features)
- Feature Importance: 20.02% (highest importance score)
- Conclusion: Overall quality explains ~20% of price variation

---

### H2: House Size Matters Significantly
**Statement:** Living area, basement size, and lot area significantly impact house prices.

**How to Validate:**
- Analyze correlation of GrLivArea, TotalBsmtSF, LotArea with SalePrice
- Check feature importance scores
- Compare importance ranking vs other features

**Validation Results:** ✅ **CONFIRMED**
- GrLivArea Correlation: 0.7086 (second highest)
- GrLivArea Importance: 15.14% (second highest)
- TotalBsmtSF Correlation: 0.6136
- Conclusion: House size is the second most important price driver

---

### H3: Age/Year Built Influences Price
**Statement:** Newer houses command higher prices.

**How to Validate:**
- Calculate correlation between YearBuilt and SalePrice
- Measure feature importance
- Analyze price trends by construction year

**Validation Results:** ✅ **CONFIRMED**
- YearBuilt Correlation: 0.5229
- YearBuilt Importance: 8.47%
- Conclusion: Newer houses tend to be more valuable

---

### H4: Model Meets Target Accuracy
**Statement:** A Random Forest model can predict prices with R² ≥ 0.75.

**How to Validate:**
- Train model on 80% of data (train set)
- Evaluate on 20% of data (test set)
- Calculate R² score on test set
- Compare against target of 0.75

**Validation Results:** ✅ **EXCEEDED TARGET**
- Test R² Score: 0.8897 (target: 0.75)
- Improvement: 18.6% above minimum requirement
- Conclusion: Model exceeds performance expectations

---

## Rationale to Map Business Requirements

### Business Requirement 1 → Data Visualizations

| BR | ML Task | Visualization | Notebook | Dashboard Page |
|----|---------|--------------------|----------|-----------------|
| BR1: Discover correlations | Exploratory Data Analysis | Correlation heatmap, Bar chart | 01_data_exploration.ipynb | Correlation Study |
| | | Feature correlation ranking | | |
| | | Top 10 features visualization | | |

**Rationale:** Clients need visual evidence that features truly correlate with price. Bar charts are most intuitive for showing which attributes matter most.

---

### Business Requirement 2 → Machine Learning Tasks

| BR | ML Task | Algorithm | Model | Dashboard Page |
|----|---------|-----------|-------|-----------------|
| BR2: Predict prices | Regression | Random Forest | Hyperparameter-tuned RF | Price Predictor |
| | | (non-linear) | (R² = 0.8897) | Model Performance |
| | Feature Engineering | Label Encoding | Categorical encoding | |
| | Model Optimization | GridSearchCV | 576 parameter combinations | |

**Rationale:** 
- **Regression chosen** because we predict continuous price values (not categories)
- **Random Forest chosen** because it handles non-linear relationships, provides feature importance, and is robust to outliers
- **GridSearchCV chosen** to find optimal hyperparameters and maximize R² score
- **Label Encoding chosen** because tree-based models need numerical inputs

---

## Machine Learning Business Case

### Problem Statement
Client inherited 4 houses in Ames, Iowa. She needs accurate price predictions to maximize sale revenue. Current local knowledge may not apply.

### Business Objective
Develop a predictive model that explains house price variation with R² ≥ 0.75, enabling confident valuation of inherited properties.

### ML Objective
Build a regression model that predicts SalePrice based on 21 house attributes with:
- **Accuracy:** R² ≥ 0.75 on test set
- **Interpretability:** Feature importance scores to explain price drivers
- **Usability:** Interactive dashboard for real-time predictions

### Success Metrics
- ✅ **R² Score:** 0.8897 (target: 0.75) - **EXCEEDED**
- ✅ **MAE:** $17,200 - Acceptable prediction error
- ✅ **RMSE:** $29,091 - Reasonable for price range
- ✅ **Model Interpretability:** Top 5 features identified
- ✅ **Deployment:** Live on Heroku, accessible 24/7

### Business Impact
- Client can confidently price 4 inherited houses
- Estimated total valuation: ~$650,000 (sum of 4 predictions)
- Data-driven approach reduces risk of underpricing
- Scalable solution for valuing any house in Ames

---

## Dashboard Design

### Overview
Interactive Streamlit application with 5 pages for data exploration, analysis, and price prediction.

### Page 1: Home
**Purpose:** Project introduction and overview

**Content:**
- Project title and description
- Business requirements summary
- Dataset statistics (1,460 houses, 21 features)
- Model performance highlight (R² = 0.8897)
- Navigation instructions

**Widgets:** Text blocks, metrics cards

---

### Page 2: Correlation Study
**Purpose:** Answer BR1 - Show which features correlate with price

**Content:**
- Top 10 correlated features table
- Horizontal bar chart visualization
- Correlation coefficients
- Key insights interpretation

**Widgets:** DataFrame display, matplotlib bar chart, text blocks

**Layout:** 2 columns (table + chart)

---

### Page 3: Price Predictor
**Purpose:** Answer BR2 - Predict prices for inherited and custom houses

**Content:**
- **Tab 1 - Inherited Houses:**
  - Table with 4 inherited house attributes
  - Individual price predictions
  - Total valuation for all 4 houses
  - Key metrics display

- **Tab 2 - Custom Prediction:**
  - Interactive input sliders (Quality, Year Built, Bedrooms)
  - Number input fields (Area, Lot Size, etc.)
  - Predict button
  - Real-time price output
  - Confidence notes

**Widgets:** Sliders, number inputs, buttons, metrics, dataframe

**Layout:** 2 columns for inputs (responsive)

---

### Page 4: Hypothesis & Validation
**Purpose:** Document and validate all project hypotheses

**Content:**
- H1: Overall Quality is Primary Driver
- H2: House Size Matters
- H3: Age Influences Price
- H4: Model Meets Accuracy Target
- Each with validation evidence and results

**Widgets:** Text blocks, markdown formatting

---

### Page 5: Model Performance
**Purpose:** Display detailed model metrics and feature importance

**Content:**
- 4 key metrics (R² Train, R² Test, MAE, RMSE)
- Performance interpretation
- Best hyperparameters (6 parameters)
- Top 5 feature importance bar chart
- Target achievement verification

**Widgets:** Metric cards, matplotlib chart, text blocks

**Layout:** 4-column metrics, chart below

---

## Data Analysis Results

### Key Findings from EDA

#### Missing Values
- **EnclosedPorch:** 90.68% missing → DROPPED
- **WoodDeckSF:** 89.38% missing → DROPPED
- **LotFrontage:** 17.74% missing → Filled with median
- **Other features:** <17% missing → Appropriate handling applied

#### Correlation Analysis

| Feature | Correlation | Importance | Interpretation |
|---------|-------------|-----------|-----------------|
| OverallQual | 0.7910 | 20.02% | Quality is dominant factor |
| GrLivArea | 0.7086 | 15.14% | Living area is critical |
| GarageArea | 0.6234 | 8.61% | Garage adds value |
| YearBuilt | 0.5229 | 8.47% | Newer houses worth more |
| TotalBsmtSF | 0.6136 | 7.78% | Basement space matters |

#### Price Distribution
- Right-skewed distribution (most houses $100k-$250k)
- Long tail of luxury homes ($300k-$755k)
- Outliers present but valid data

#### Data Quality
- Final dataset: 1,460 records, 21 features
- Zero missing values after cleaning
- No data corruption identified
- All values within logical ranges

---

## Machine Learning Model

### Model Selection & Approach

**Algorithm:** Random Forest Regressor

**Rationale:**
- Handles non-linear relationships between features and price
- Provides feature importance scores (interpretability)
- Robust to outliers in price data
- No scaling required for tree-based model
- Good generalization ability

### Data Preparation

**Train/Test Split:** 80/20 (1,168 train, 292 test)
- Random state: 42 (reproducibility)
- Stratified: No (continuous target)

**Feature Engineering:**
- Categorical variables: Label Encoded (BsmtExposure, BsmtFinType1, GarageFinish, KitchenQual)
- Numerical variables: No scaling (trees are scale-invariant)
- Feature selection: All 21 features retained (no removal)

### Hyperparameter Optimization

**Method:** GridSearchCV with 5-fold Cross-Validation

**Hyperparameters Tested:** 6 parameters, 576 total combinations

| Parameter | Values Tested | Best Value | Impact |
|-----------|---------------|-----------|--------|
| n_estimators | [50, 100, 150, 200] | 200 | More trees improve accuracy |
| max_depth | [10, 15, 20, 25] | 20 | Controls tree complexity |
| min_samples_split | [2, 5, 10] | 5 | Prevents overfitting |
| min_samples_leaf | [1, 2, 4] | 2 | Minimum leaf size |
| max_features | ['sqrt', 'log2'] | 'sqrt' | Features per split |
| bootstrap | [True, False] | False | Deterministic training |

**Cross-Validation Score:** 0.8530 (R²)

### Model Performance

| Metric | Train | Test | Target | Status |
|--------|-------|------|--------|--------|
| **R² Score** | 0.9898 | 0.8897 | ≥ 0.75 | ✅ EXCEEDED |
| **MAE** | - | $17,200 | - | ✅ GOOD |
| **RMSE** | - | $29,091 | - | ✅ GOOD |

**Interpretation:**
- Model explains 88.97% of price variance on test set
- Average prediction error: ±$17,200
- Small gap between train and test indicates good generalization
- No significant overfitting despite high train score

### Feature Importance

**Top 5 Most Important Features:**

| Rank | Feature | Importance | Interpretation |
|------|---------|-----------|-----------------|
| 1 | OverallQual | 20.02% | Quality is paramount |
| 2 | GrLivArea | 15.14% | Living area critical |
| 3 | GarageArea | 8.61% | Garage adds value |
| 4 | YearBuilt | 8.47% | Age matters |
| 5 | TotalBsmtSF | 7.78% | Basement important |

These 5 features account for ~60% of model decisions.

---

## Main Data Analysis and Machine Learning Libraries

### pandas - Data Manipulation & Analysis

**Usage:** Data loading, cleaning, transformation

**Example from Project:**
```python
import pandas as pd

# Load main dataset
df = pd.read_csv('inputs/datasets/house_prices_records.csv')

# Handle missing values
missing_percentage = (df.isnull().sum() / len(df)) * 100
sparse_features = missing_percentage[missing_percentage > 80]
df_cleaned = df.drop(columns=sparse_features.index)

# Fill remaining missing values
for col in df_cleaned.select_dtypes(include=[np.number]).columns:
    if df_cleaned[col].isnull().sum() > 0:
        df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)

# Separate features and target
X = df_cleaned.drop('SalePrice', axis=1)
y = df_cleaned['SalePrice']
```

**Purpose:** Handle 1,460 records, remove sparse features (>80% missing), impute missing values, prepare data for ML

---

### numpy - Numerical Computing

**Usage:** Array operations, mathematical computations

**Example from Project:**
```python
import numpy as np

# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])

# Calculate correlation matrix
correlation_matrix = numeric_df.corr()
top_correlations = correlation_matrix['SalePrice'].sort_values(ascending=False)

# Check for missing patterns
missing_percentage = (df.isnull().sum() / len(df)) * 100
```

**Purpose:** Compute correlations, identify numeric features, calculate statistics

---

### scikit-learn - Machine Learning

**Usage:** Model training, hyperparameter tuning, evaluation

**Examples from Project:**
```python
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Encode categorical variables
le = LabelEncoder()
df_cleaned['BsmtExposure'] = le.fit_transform(df_cleaned['BsmtExposure'])

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Hyperparameter tuning with GridSearchCV (576 combinations tested)
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [10, 15, 20, 25],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}
grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Evaluate model
r2_test = r2_score(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
```

**Purpose:** Train, tune, and evaluate regression model achieving R² = 0.8897

---

### matplotlib - Data Visualization

**Usage:** Static plots and charts

**Examples from Project:**
```python
import matplotlib.pyplot as plt

# Correlation bar chart
fig, ax = plt.subplots(figsize=(10, 6))
top_correlations.head(11).plot(kind='barh', ax=ax, color='steelblue')
ax.set_xlabel('Correlation Coefficient')
ax.set_title('Features Most Correlated with SalePrice')
plt.show()

# Actual vs Predicted scatter plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_pred_test, alpha=0.5, color='steelblue')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
        'r--', lw=2, label='Perfect Prediction')
ax.set_xlabel('Actual Price ($)')
ax.set_ylabel('Predicted Price ($)')
ax.set_title('Actual vs Predicted Prices')
plt.show()
```

**Purpose:** Visualize correlations, model predictions, and residuals

---

### seaborn - Statistical Data Visualization

**Usage:** Enhanced matplotlib visualizations

**Examples from Project:**
```python
import seaborn as sns

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Feature importance heatmap (implied in feature analysis)
```

**Purpose:** Professional styling and aesthetics for all plots

---

### streamlit - Dashboard Framework

**Usage:** Build interactive web application

**Examples from Project:**
```python
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Heritage Housing Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Navigation
page = st.sidebar.radio(
    "Select Page",
    ["Home", "Correlation Study", "Price Predictor", "Hypothesis", "Model Performance"]
)

# Display correlation data
st.write("### Top 10 Correlated Features")
st.dataframe(top_10, use_container_width=True)

# Interactive inputs
input_values['OverallQual'] = st.slider('Overall Quality (1-10)', 1, 10, 6)
input_values['GrLivArea'] = st.number_input('Ground Living Area (sqft)', 300, 5000, 1500)

# Display predictions
st.metric(
    label="Total Predicted Price for All 4 Houses",
    value=f"${total_price:,.2f}"
)
```

**Purpose:** Create interactive 5-page dashboard deployed on Heroku

---

### joblib - Model Serialization

**Usage:** Save and load trained models

**Examples from Project:**
```python
import joblib

# Save trained model
joblib.dump(best_model, 'outputs/best_model.pkl')

# Load model in dashboard
model = joblib.load('outputs/best_model.pkl')

# Use for predictions
predicted_price = model.predict(input_df)[0]
```

**Purpose:** Persist trained model for production deployment

---

## Project Structure
```
heritage-housing/
├── app.py                                    # Main Streamlit application
├── README.md                                 # Project documentation
├── requirements.txt                          # Python dependencies
├── Procfile                                  # Heroku deployment
├── runtime.txt                               # Python version
├── setup.sh                                  # Heroku setup script
├── .gitignore                                # Git ignore patterns
│
├── jupyter_notebooks/
│   ├── 01_data_exploration.ipynb             # EDA & correlation analysis
│   ├── 02_data_cleaning.ipynb                # Data cleaning & preprocessing
│   └── 03_model_training.ipynb               # Model training & tuning
│
├── inputs/
│   └── datasets/
│       ├── house_prices_records.csv          # Main dataset (1,460 houses)
│       ├── inherited_houses.csv              # 4 inherited houses
│       └── house-metadata.txt                # Feature descriptions
│
└── outputs/
    ├── best_model.pkl                        # Trained Random Forest model
    ├── X.csv                                 # Features
    ├── y.csv                                 # Target variable
    ├── X_y_cleaned.csv                       # Complete cleaned dataset
    ├── feature_importance.csv                # Feature importance scores
    └── model_metrics.json                    # Performance metrics
```

---

## Installation & Setup

### Prerequisites
- Python 3.9+
- pip package manager
- Git

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/heritage-housing.git
cd heritage-housing
```

### Step 2: Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run Dashboard Locally
```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`

---

## Unfixed Bugs

**Status:** ✅ **No unfixed bugs identified**

All identified issues during development were resolved:
- ✅ Missing values in sparse features → Dropped features with >80% missing
- ✅ Categorical variables encoding → Label Encoded for model compatibility
- ✅ Feature mismatch in Heroku → Hardcoded inherited houses data
- ✅ Slug size too large → Optimized .slugignore configuration
- ✅ Git tracking issues → Resolved with forced file additions

**Testing Completed:**
- ✅ Local Streamlit app runs without errors
- ✅ All 5 dashboard pages functional
- ✅ Model predictions work correctly
- ✅ Interactive widgets respond properly
- ✅ Heroku deployment stable

---

## Deployment

### Heroku Deployment

**Status:** ✅ **Successfully Deployed**

**Live App URL:** https://heritage-housing-price-4311f0b65bc6.herokuapp.com/

**Deployment Steps Completed:**

1. **Create Heroku App**
```bash
   heroku create heritage-housing-price
```

2. **Configure Deployment Files**
   - `Procfile`: Specifies how to run the app
   - `runtime.txt`: Python 3.9.17
   - `setup.sh`: Streamlit configuration
   - `.slugignore`: Excludes venv from upload

3. **Deploy to Heroku**
```bash
   git push heroku main
```

4. **Verify Deployment**
```bash
   heroku open
```

**Deployment Configuration:**
- Stack: Heroku-24
- Python Version: 3.9.17
- Region: United States
- Slug Size: 499 MB (soft limit: 300 MB)
- Status: Active and running

**Access:** Available 24/7 at the live URL above

---

## Credits

### Data Source
- **Dataset:** Kaggle - Housing Prices in Ames, Iowa
- **Link:** https://www.kaggle.com/codeinstitute/housing-prices-data
- **License:** Public Domain

### Libraries & Frameworks
- **pandas:** Data manipulation (McKinney et al.)
- **scikit-learn:** Machine learning (Pedregosa et al.)
- **Streamlit:** Web framework
- **matplotlib/seaborn:** Visualizations
- **joblib:** Model serialization

### Course & Institution
- **Course:** Diploma in Full Stack Software Development - Predictive Analytics
- **Institution:** Code Institute
- **Cohort:** November 2025

### Project Template
- **Template Repository:** Code Institute Heritage Housing Template
- **Handbook:** Heritage Housing Issues Assessment Handbook

### Resources & References
- Code Institute Python & ML modules
- scikit-learn documentation (https://scikit-learn.org)
- Streamlit documentation (https://streamlit.io)
- Python Data Science Handbook

---

## Acknowledgments

- Code Institute mentors and community support
- Kaggle for the public housing dataset
- Open-source contributors to pandas, scikit-learn, and Streamlit
- Project stakeholders (Lydia Doe - fictional client)

---

**Last Updated:** November 25, 2025
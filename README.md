# Heritage Housing Price Predictor 🏠

## Project Overview

This project builds a machine learning application to predict house prices in Ames, Iowa. 
The solution combines data analysis, feature engineering, and a trained Random Forest model deployed via a Streamlit web dashboard.

**Live Demo:** [[Link will be added after Heroku deployment](https://heritage-housing-price-4311f0b65bc6.herokuapp.com/)]

---

## Table of Contents
1. [Business Case Assessment](#business-case-assessment)
2. [Dataset Content](#dataset-content)
3. [Project Structure](#project-structure)
4. [Installation & Setup](#installation--setup)
5. [Data Analysis Results](#data-analysis-results)
6. [Machine Learning Model](#machine-learning-model)
7. [Project Hypothesis](#project-hypothesis)
8. [Dashboard Pages](#dashboard-pages)
9. [Technologies Used](#technologies-used)
10. [Author & Contact](#author--contact)

---

## Business Case Assessment

### Business Requirements

**BR1 - Data Analysis:**
The client is interested in discovering how house attributes correlate with sale prices. This requires visual analysis and statistical correlation studies.

**BR2 - ML Prediction:**
The client needs to predict house prices for 4 inherited houses and any other house in Ames, Iowa with high accuracy.

### Success Criteria

- **Target R² Score:** ≥ 0.75 on test set ✅ **ACHIEVED: 0.8897**
- **Data Quality:** Handle missing values and encode categorical variables
- **User Experience:** Intuitive dashboard with clear predictions and visualizations

### Project Stakeholders

- **Lydia Doe** (Property Owner): Needs accurate price predictions for 4 inherited houses
- **Business Impact:** Maximize sale prices and inform property valuation decisions

---

## Dataset Content

### Source
- **Main Dataset:** 1,460 houses in Ames, Iowa
- **Inherited Houses:** 4 properties requiring valuation
- **Features:** 21 (after cleaning)
- **Target:** SalePrice

### Data Statistics

| Metric | Value |
|--------|-------|
| Total Records | 1,460 |
| Feature Count | 21 (after cleaning) |
| Price Range | $34,900 - $755,000 |
| Average Price | $180,921 |
| Missing Values | Handled (0 remaining) |

### Feature Categories

**Numerical Features (13):**
1stFlrSF, 2ndFlrSF, BsmtFinSF1, BsmtUnfSF, GarageArea, GrLivArea, LotArea, LotFrontage, MasVnrArea, OpenPorchSF, TotalBsmtSF, YearBuilt, YearRemodAdd

**Categorical Features (4):**
BsmtExposure, BsmtFinType1, GarageFinish, KitchenQual

**Target Variable:**
SalePrice

---

## Project Structure
```
heritage-housing/
├── app.py                          # Main Streamlit application
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── Procfile                        # Heroku deployment
├── runtime.txt                     # Python version
├── setup.sh                        # Heroku setup script
│
├── jupyter_notebooks/
│   ├── 01_data_exploration.ipynb          # Data exploration & correlation
│   ├── 02_data_cleaning.ipynb             # Data cleaning & preprocessing
│   └── 03_model_training.ipynb            # Model training & tuning
│
├── inputs/
│   └── datasets/
│       ├── house_prices_records.csv       # Main dataset (1,460 houses)
│       ├── inherited_houses.csv           # 4 inherited houses
│       └── house-metadata.txt             # Feature descriptions
│
└── outputs/
    ├── best_model.pkl                     # Trained Random Forest model
    ├── X.csv                              # Features (X)
    ├── y.csv                              # Target (y)
    ├── X_y_cleaned.csv                    # Cleaned dataset
    ├── feature_importance.csv             # Feature importance scores
    └── model_metrics.json                 # Performance metrics
```

---

## Installation & Setup

### Prerequisites
- Python 3.9+
- pip package manager
- Git

### Step 1: Clone Repository
```bash
git clone <repository-url>
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

The dashboard will open at `http://localhost:8501`

---

## Data Analysis Results

### Key Findings from EDA

#### Missing Values
- **EnclosedPorch:** 90.68% missing → DROPPED
- **WoodDeckSF:** 89.38% missing → DROPPED
- **LotFrontage:** 17.74% missing → Filled with median
- **Other features:** <17% missing → Handled appropriately

#### Correlation with SalePrice

| Feature | Correlation | Importance |
|---------|-------------|-----------|
| OverallQual | 0.7910 | 20.02% |
| GrLivArea | 0.7086 | 15.14% |
| GarageArea | 0.6234 | 8.61% |
| YearBuilt | 0.5229 | 8.47% |
| TotalBsmtSF | 0.6136 | 7.78% |

### Insights

1. **Quality is Paramount:** OverallQual is the strongest predictor (0.79 correlation, 20% importance)
2. **Size Matters:** Living area (GrLivArea) is the second most important factor
3. **Garage & Basement:** These features significantly influence price
4. **Age Consideration:** Newer houses tend to command higher prices
5. **Price Distribution:** Right-skewed distribution with outliers at high end

---

## Machine Learning Model

### Model Selection & Approach

**Algorithm:** Random Forest Regressor
- **Reason:** Handles non-linear relationships, robust to outliers, provides feature importance
- **Training Data:** 1,168 houses (80%)
- **Test Data:** 292 houses (20%)

### Hyperparameter Optimization

Performed extensive GridSearchCV with 576 parameter combinations:

| Hyperparameter | Best Value | Range Tested |
|----------------|-----------|--------------|
| n_estimators | 200 | [50, 100, 150, 200] |
| max_depth | 20 | [10, 15, 20, 25] |
| min_samples_split | 5 | [2, 5, 10] |
| min_samples_leaf | 2 | [1, 2, 4] |
| max_features | sqrt | ['sqrt', 'log2'] |
| bootstrap | False | [True, False] |

### Model Performance

| Metric | Train | Test | Target |
|--------|-------|------|--------|
| **R² Score** | 0.9898 | 0.8897 | ≥ 0.75 |
| **MAE** | - | $17,200 | - |
| **RMSE** | - | $29,091 | - |

**Status:** ✅ **TARGET EXCEEDED** (0.8897 > 0.75)

### Performance Interpretation

- **R² = 0.8897:** Model explains 88.97% of price variance
- **Mean Absolute Error:** Predictions off by ~$17,200 on average
- **Generalization:** Small gap between train (0.9898) and test (0.8897) indicates good generalization
- **Reliability:** Model is ready for production predictions

---

## Project Hypothesis

### Hypothesis 1: Quality is the Primary Price Driver
**Statement:** Overall quality has the strongest relationship with house price.

**Validation:**
- Correlation coefficient: 0.7910 ✅
- Feature importance: 20.02% (highest) ✅
- **Result:** ✅ **CONFIRMED**

### Hypothesis 2: House Size Matters
**Statement:** Living area and basement size significantly impact price.

**Validation:**
- GrLivArea correlation: 0.7086 ✅
- GrLivArea importance: 15.14% (2nd highest) ✅
- TotalBsmtSF correlation: 0.6136 ✅
- **Result:** ✅ **CONFIRMED**

### Hypothesis 3: Age Influences Price
**Statement:** Newer houses command higher prices.

**Validation:**
- YearBuilt correlation: 0.5229 ✅
- YearBuilt importance: 8.47% ✅
- **Result:** ✅ **CONFIRMED**

### Hypothesis 4: Model Meets Target Accuracy
**Statement:** Random Forest can predict prices with R² ≥ 0.75.

**Validation:**
- Test R² Score: 0.8897 ✅
- Target: 0.75 ✅
- **Result:** ✅ **EXCEEDED TARGET** (18.6% above minimum)

---

## Dashboard Pages

### Page 1: Home
- Project overview and objectives
- Dataset summary (1,460 houses, 21 features)
- Business requirements and success criteria
- Model performance highlight (R² = 0.8897)

### Page 2: Correlation Study
- **Purpose:** Answer BR1 - Show feature correlations with price
- **Content:**
  - Table of top 10 correlated features
  - Horizontal bar chart visualization
  - Key insights about price drivers

### Page 3: Price Predictor
- **Purpose:** Answer BR2 - Predict prices for inherited and custom houses
- **Tab 1 - Inherited Houses:**
  - Table showing 4 inherited houses
  - Individual price predictions
  - Total predicted price for all 4 houses
- **Tab 2 - Custom Prediction:**
  - Interactive sliders and input fields
  - Real-time price prediction
  - Results with confidence information

### Page 4: Hypothesis & Validation
- Lists all 4 project hypotheses
- Shows validation evidence for each
- Displays correlation and importance metrics
- Confirms all hypotheses were validated

### Page 5: Model Performance
- Key metrics in 4-column layout (R², MAE, RMSE)
- Performance interpretation
- Best hyperparameters used
- Top 5 feature importance bar chart

---

## Technologies Used

### Python Libraries
- **Data Processing:** pandas, numpy
- **Machine Learning:** scikit-learn (RandomForestRegressor, GridSearchCV)
- **Visualization:** matplotlib, seaborn, plotly
- **Web Framework:** streamlit
- **Model Serialization:** joblib

### Development Tools
- **IDE:** Visual Studio Code
- **Version Control:** Git & GitHub
- **Deployment:** Heroku
- **Environment:** Python 3.9

### Key Packages
```
pandas==1.3.5
numpy==1.21.6
scikit-learn==1.0.2
streamlit==1.24.0
matplotlib==3.5.1
seaborn==0.11.2
joblib==1.1.1
```

---

## Model Deployment

### Local Testing
```bash
streamlit run app.py
```

### Heroku Deployment (Coming Soon)
```bash
heroku login
heroku create heritage-housing-predictor
git push heroku main
```

**Deployed App:** [Link will be updated after deployment]

---

## Key Learnings & Conclusions

1. **Data Cleaning is Critical:** Removing sparse features (>80% missing) significantly improved model robustness
2. **Feature Engineering Matters:** Proper encoding of categorical variables was essential
3. **Hyperparameter Tuning Pays Off:** Extensive GridSearchCV (576 combinations) led to optimal model performance
4. **Business Alignment:** Model directly addresses client needs with actionable insights
5. **Dashboard Value:** Interactive visualization makes predictions accessible to non-technical users

---

## Future Enhancements

- Add confidence intervals for predictions
- Implement SHAP values for model explainability
- Expand to predict prices for multiple listings at once
- Add historical price trend analysis
- Integrate real estate market data APIs
- Create ensemble models combining multiple algorithms

---

## Author & Contact

**Developer:** Sara Rosati  
**Course:** Diploma in Full Stack Software Development (Predictive Analytics)  
**Institution:** Code Institute  
**Project Date:** November 2025  

**GitHub Repository:** [Link to repository]  
**Live Dashboard:** [Link to Heroku deployment - coming soon]

---

## License

This project is part of Code Institute's curriculum and is for educational purposes.

---

## Acknowledgments

- Code Institute for the project structure and guidelines
- Kaggle for the housing dataset
- scikit-learn and Streamlit documentation
- The mentors and community support

---

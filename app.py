import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Set page configuration
st.set_page_config(
    page_title="Heritage Housing Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load data
@st.cache_resource
def load_model():
    return joblib.load('outputs/best_model.pkl')

@st.cache_data
def load_data():
    return pd.read_csv('outputs/X_y_cleaned.csv')

model = load_model()
data = load_data()

# Navigation sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Home", "Correlation Study", "Price Predictor", "Hypothesis", "Model Performance"]
)

# ========== PAGE 1: HOME ==========
if page == "Home":
    st.title("🏠 Heritage Housing Price Predictor")
    
    st.write("""
    ## Project Overview
    
    This dashboard helps predict house prices in Ames, Iowa based on property attributes.
    
    ### Business Requirements
    - **BR1:** Analyze how house attributes correlate with sale prices
    - **BR2:** Predict sale prices for 4 inherited houses and any other house in Ames, Iowa
    
    ### Dataset Information
    - **Total Houses:** 1,460
    - **Features:** 21 (after cleaning)
    - **Target Variable:** SalePrice
    - **Price Range:** $34,900 - $755,000
    - **Average Price:** $180,921
    
    ### Model Performance
    - **Algorithm:** Random Forest Regressor
    - **R² Score (Test):** 0.8897 ✅
    - **MAE:** $17,200
    - **RMSE:** $29,091
    
    ---
    
    **Use the navigation menu on the left to explore the analysis and make predictions!**
    """)

# ========== PLACEHOLDER PAGES ==========
elif page == "Correlation Study":
    st.title("📊 Correlation Analysis")
    
    st.write("""
    ## Feature Correlation with Sale Price
    
    This analysis shows which house attributes have the strongest relationship with sale price.
    """)
    
    # Load correlation data
    numeric_data = data.select_dtypes(include=[np.number])
    correlation = numeric_data.corr()['SalePrice'].sort_values(ascending=False)
    
    # Display top correlations
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Top 10 Correlated Features")
        top_10 = correlation.head(11)
        st.dataframe(top_10, use_container_width=True)
    
    with col2:
        st.write("### Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        correlation.head(11).plot(kind='barh', ax=ax, color='steelblue')
        ax.set_xlabel('Correlation Coefficient')
        ax.set_title('Features Most Correlated with SalePrice')
        st.pyplot(fig)
    
    st.write("""
    ### Key Insights
    - **OverallQual** (0.79): Overall quality is the strongest price driver
    - **GrLivArea** (0.71): Ground living area is very important
    - **GarageArea** (0.62): Garage size influences price
    - **YearBuilt** (0.52): Newer houses tend to be more expensive
    - **TotalBsmtSF** (0.61): Basement area matters
    """)

elif page == "Price Predictor":
    st.title("🔮 House Price Predictor")
    
    st.write("""
    ## Make Price Predictions
    
    Enter house attributes below to predict the sale price.
    """)
    
    # Load inherited houses (hardcoded data if file not found)
    try:
        inherited_df = pd.read_csv('inputs/datasets/inherited_houses.csv')
    except FileNotFoundError:
        # Hardcoded inherited houses data for Heroku
    inherited_df = pd.DataFrame({
        'OverallQual': [5, 6, 5, 6],
        'GrLivArea': [896, 1329, 1629, 1604],
        'YearBuilt': [1961, 1958, 1997, 1998],
        'GarageArea': [0, 460, 576, 576],
        'TotalBsmtSF': [0, 1262, 1629, 1604],
        '1stFlrSF': [896, 1262, 928, 926],
        '2ndFlrSF': [0, 0, 701, 678],
        'BedroomAbvGr': [3, 3, 3, 3],
        'LotArea': [9600, 9600, 11250, 11250],
        'LotFrontage': [80, 80, 68, 60],
        'BsmtExposure': [1, 1, 2, 3],
        'BsmtFinType1': [0, 0, 2, 2],
        'BsmtUnfSF': [0, 284, 434, 540],
        'GarageFinish': [2, 1, 1, 2],
        'GarageYrBlt': [1980, 1976, 2001, 1998],
        'KitchenQual': [2, 3, 2, 2],
        'OpenPorchSF': [0, 0, 42, 35],
        'OverallCond': [5, 8, 5, 5],
        'YearRemodAdd': [1961, 1976, 2002, 1970],
        'MasVnrArea': [0, 0, 162, 0]
    })
    
    # Tabs for different sections
    tab1, tab2 = st.tabs(["Inherited Houses", "Custom Prediction"])
    
    # ========== TAB 1: INHERITED HOUSES ==========
    with tab1:
        st.write("### The 4 Inherited Houses")
        
        # Prepare inherited houses for prediction
        from sklearn.preprocessing import LabelEncoder

        X_inherited = inherited_df.copy()

        # Remove the same sparse features that were removed during training
        features_to_drop = ['EnclosedPorch', 'WoodDeckSF']
        X_inherited = X_inherited.drop(columns=features_to_drop)
        
        # Drop SalePrice if it exists
        if 'SalePrice' in X_inherited.columns:
            X_inherited = X_inherited.drop('SalePrice', axis=1)

        # Encode categorical variables (same as training data)
        categorical_cols = X_inherited.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X_inherited[col] = le.fit_transform(X_inherited[col])

        # Reorder columns to match training data
        X_inherited = X_inherited[data.drop('SalePrice', axis=1).columns]
        
        predictions_inherited = model.predict(X_inherited)
        
        # Display inherited houses with predictions
        inherited_results = pd.DataFrame({
            'House': ['House 1', 'House 2', 'House 3', 'House 4'],
            'Year Built': inherited_df['YearBuilt'].values,
            'Overall Quality': inherited_df['OverallQual'].values,
            'Living Area (sqft)': inherited_df['GrLivArea'].values,
            'Predicted Price': predictions_inherited
        })
        
        st.dataframe(inherited_results, use_container_width=True)
        
        # Total predicted price
        total_price = predictions_inherited.sum()
        
        st.write("---")
        st.metric(
            label="Total Predicted Price for All 4 Houses",
            value=f"${total_price:,.2f}",
            delta=None
        )
        
        st.write(f"""
        ### Summary
        - **House 1:** ${predictions_inherited[0]:,.2f}
        - **House 2:** ${predictions_inherited[1]:,.2f}
        - **House 3:** ${predictions_inherited[2]:,.2f}
        - **House 4:** ${predictions_inherited[3]:,.2f}
        - **TOTAL:** ${total_price:,.2f}
        """)
    
    # ========== TAB 2: CUSTOM PREDICTION ==========
    with tab2:
        st.write("### Predict Price for Any House")
        
        # Get feature names
        feature_names = data.drop('SalePrice', axis=1).columns.tolist()
        
        # Create input widgets
        st.write("#### Enter House Attributes")
        
        col1, col2 = st.columns(2)
        
        input_values = {}
        
        with col1:
            input_values['OverallQual'] = st.slider('Overall Quality (1-10)', 1, 10, 6)
            input_values['GrLivArea'] = st.number_input('Ground Living Area (sqft)', 300, 5000, 1500)
            input_values['GarageArea'] = st.number_input('Garage Area (sqft)', 0, 2000, 500)
            input_values['TotalBsmtSF'] = st.number_input('Total Basement Area (sqft)', 0, 5000, 1000)
            input_values['YearBuilt'] = st.slider('Year Built', 1870, 2020, 2000)
        
        with col2:
            input_values['1stFlrSF'] = st.number_input('1st Floor Area (sqft)', 300, 5000, 1200)
            input_values['2ndFlrSF'] = st.number_input('2nd Floor Area (sqft)', 0, 3000, 0)
            input_values['BedroomAbvGr'] = st.slider('Bedrooms', 0, 8, 3)
            input_values['LotArea'] = st.number_input('Lot Area (sqft)', 1000, 200000, 10000)
            input_values['LotFrontage'] = st.number_input('Lot Frontage (ft)', 0, 300, 70)
        
        # Add remaining features with default values
        for feature in feature_names:
            if feature not in input_values:
                input_values[feature] = data[feature].median()
        
        # Predict button
        if st.button("Predict Price", key="predict_button"):
            # Prepare input for model
            input_df = pd.DataFrame([input_values])
            
            # Reorder columns to match training data
            input_df = input_df[feature_names]
            
            # Make prediction
            predicted_price = model.predict(input_df)[0]
            
            # Display result
            st.success(f"### Predicted Price: ${predicted_price:,.2f}")
            
            st.write(f"""
            Based on the entered attributes, this house would sell for approximately **${predicted_price:,.2f}**.
            
            **Note:** This prediction is based on the Random Forest model trained on 1,460 houses in Ames, Iowa.
            Actual prices may vary based on market conditions and other factors.
            """)

elif page == "Hypothesis":
    st.title("🔬 Project Hypothesis & Validation")
    
    st.write("""
    ## Project Hypotheses
    
    ### Hypothesis 1: Quality is the Primary Price Driver
    **Statement:** Overall quality has the strongest correlation with sale price.
    
    **Validation:**
    - Correlation coefficient: 0.79 ✅
    - Feature importance: 20.02% (highest) ✅
    - **Result:** CONFIRMED
    
    ### Hypothesis 2: House Size Matters
    **Statement:** Living area and lot size significantly impact price.
    
    **Validation:**
    - GrLivArea correlation: 0.71 ✅
    - GrLivArea importance: 15.14% (second highest) ✅
    - TotalBsmtSF correlation: 0.61 ✅
    - **Result:** CONFIRMED
    
    ### Hypothesis 3: Age of House is Important
    **Statement:** Year built influences price.
    
    **Validation:**
    - YearBuilt correlation: 0.52 ✅
    - YearBuilt importance: 8.47% ✅
    - **Result:** CONFIRMED
    
    ### Hypothesis 4: Model Accuracy
    **Statement:** We can predict house prices with R² ≥ 0.75.
    
    **Validation:**
    - Test R² Score: 0.8897 ✅
    - Target: 0.75 ✅
    - **Result:** EXCEEDED TARGET
    
    ---
    
    ## Conclusion
    All hypotheses were validated. The machine learning model successfully predicts house prices
    with high accuracy, with overall quality being the most important factor.
    """)

elif page == "Model Performance":
    st.title("📈 Model Performance Metrics")
    
    st.write("""
    ## Model Overview
    **Algorithm:** Random Forest Regressor
    **Training Data:** 1,168 houses (80%)
    **Test Data:** 292 houses (20%)
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="R² Score (Train)", value="0.9898")
    
    with col2:
        st.metric(label="R² Score (Test)", value="0.8897")
    
    with col3:
        st.metric(label="MAE", value="$17,200")
    
    with col4:
        st.metric(label="RMSE", value="$29,091")
    
    st.write("---")
    
    st.write("""
    ## Performance Interpretation
    
    - **R² Score:** 0.8897 means the model explains 88.97% of price variance ✅
    - **MAE:** On average, predictions are off by $17,200
    - **RMSE:** Root mean squared error of $29,091
    - **Target Achievement:** R² ≥ 0.75 requirement **EXCEEDED** ✅
    
    ## Best Model Hyperparameters
    """)
    
    params = {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'bootstrap': False
    }
    
    for param, value in params.items():
        st.write(f"- **{param}:** {value}")
    
    st.write("""
    ## Top 5 Most Important Features
    """)
    
    feature_importance_data = {
        'Feature': ['OverallQual', 'GrLivArea', 'GarageArea', 'YearBuilt', 'TotalBsmtSF'],
        'Importance (%)': [20.02, 15.14, 8.61, 8.47, 7.78]
    }
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(feature_importance_data['Feature'], feature_importance_data['Importance (%)'], color='steelblue')
    ax.set_xlabel('Importance (%)')
    ax.set_title('Top 5 Most Important Features')
    st.pyplot(fig)
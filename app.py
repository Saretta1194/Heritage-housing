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
    st.title("🔮 Price Predictor")
    st.write("Page in development...")

elif page == "Hypothesis":
    st.title("🔬 Project Hypothesis")
    st.write("Page in development...")

elif page == "Model Performance":
    st.title("📈 Model Performance")
    st.write("Page in development...")
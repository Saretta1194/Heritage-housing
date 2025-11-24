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
    st.write("Page in development...")

elif page == "Price Predictor":
    st.title("🔮 Price Predictor")
    st.write("Page in development...")

elif page == "Hypothesis":
    st.title("🔬 Project Hypothesis")
    st.write("Page in development...")

elif page == "Model Performance":
    st.title("📈 Model Performance")
    st.write("Page in development...")
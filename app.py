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
    layout="wide",
)

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Load data


@st.cache_resource
def load_model():
    return joblib.load("outputs/best_model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("outputs/X_y_cleaned.csv")


model = load_model()
data = load_data()

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Correlation Study",
        "Price Predictor",
        "Hypothesis",
        "Model Performance",
    ],
)
if page == "Home":
    st.markdown(
        """
    <style>
    body { background-color: #f5f5f5; }

    .header-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 50px 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 40px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }

    .header-box h1 {
        font-size: 2.8em;
        margin: 0;
        font-weight: bold;
    }

    .header-box p {
        font-size: 1.2em;
        margin: 15px 0 0 0;
        opacity: 0.95;
    }

    .stats-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 20px;
        margin: 40px 0;
    }

    .stat-card {
        background: white;
        padding: 30px 20px;
        border-radius: 12px;
        text-align: center;
        border-top: 5px solid #667eea;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        min-height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .stat-icon {
        font-size: 3em;
        margin-bottom: 10px;
    }

    .stat-number {
        font-size: 2.2em;
        font-weight: bold;
        color: #667eea;
        margin: 10px 0;
    }

    .stat-label {
        font-size: 0.95em;
        color: #555;
        font-weight: 500;
    }

    .requirement-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 25px;
        margin: 30px 0;
    }

    .requirement-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 35px;
        border-radius: 12px;
        color: white;
        box-shadow: 0 8px 20px rgba(245, 87, 108, 0.3);
        min-height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .requirement-card h3 {
        margin-top: 0;
        font-size: 1.4em;
        margin-bottom: 15px;
    }

    .requirement-card p {
        margin: 8px 0;
        line-height: 1.6;
        font-size: 1em;
    }

    .performance-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 25px;
        margin: 40px 0;
    }

    .performance-card {
        border-radius: 12px;
        padding: 35px;
        text-align: center;
        color: white;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    }

    .performance-card.success {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }

    .performance-card.info {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }

    .performance-card p {
        font-size: 0.95em;
        margin: 5px 0;
    }

    .performance-number {
        font-size: 3em;
        font-weight: bold;
        margin: 15px 0;
    }

    .insight-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 35px;
        border-radius: 12px;
        color: white;
        margin: 30px 0;
        box-shadow: 0 8px 20px rgba(79, 172, 254, 0.3);
    }

    .insight-box h4 {
        margin-top: 0;
        font-size: 1.4em;
        margin-bottom: 20px;
    }

    .insight-box p {
        margin: 10px 0;
        font-size: 1.05em;
        line-height: 1.7;
    }

    .instruction-box {
        background: white;
        padding: 35px;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 30px 0;
    }

    .instruction-box h4 {
        margin-top: 0;
        color: #667eea;
        font-size: 1.3em;
    }

    .instruction-box ul {
        margin: 20px 0;
        padding-left: 25px;
    }

    .instruction-box li {
        margin: 12px 0;
        font-size: 1.05em;
        color: #333;
        line-height: 1.6;
    }

    .footer-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 35px;
        border-radius: 12px;
        text-align: center;
        margin-top: 50px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }

    .footer-box p {
        margin: 10px 0;
        font-size: 1em;
        line-height: 1.8;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # HEADER
    st.markdown(
        """
    <div class="header-box">
        <h1>🏠 Heritage Housing Price Predictor</h1>
        <p>🤖 AI-Powered Real Estate Valuation System</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # STATS
    st.markdown(
        """
    <div class="stats-container">
        <div class="stat-card">
            <div class="stat-icon">📊</div>
            <div class="stat-number">1,460</div>
            <div class="stat-label">Houses Analyzed</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">📐</div>
            <div class="stat-number">21</div>
            <div class="stat-label">Features Used</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">💰</div>
            <div class="stat-number">$180K</div>
            <div class="stat-label">Average Price</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">🎯</div>
            <div class="stat-number">89%</div>
            <div class="stat-label">Model Accuracy</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # REQUIREMENTS
    st.markdown(
        """
    <div class="requirement-container">
        <div class="requirement-card">
            <h3>📈 BR1: Correlation Analysis</h3>
            <p>
            Discover which house attributes correlate most with sale price.<br>
            </p>
            <p>
            Understand the key value drivers in the Ames real estate market.
            </p>
            <p><strong>✓ Outcome:</strong>
            Data-driven insights for property<br>
            valuation</p>
        </div>
        <div class="requirement-card">
            <h3>🎯 BR2: Price Prediction</h3>
            <p>Predict sale prices for any house in Ames,
            Iowa with high accuracy.
            Use machine learning for confident valuations.</p>
            <p><strong>✓ Outcome:</strong>
            Accurate price estimates for inherited<br>
            properties</p>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # PERFORMANCE
    st.markdown(
        """
    <div class="performance-container">
        <div class="performance-card success">
            <p>R² Score</p>
            <div class="performance-number">0.8897</div>
            <p>Target: 0.75</p>
            <p style="font-weight: bold; font-size: 1.1em;">✅ EXCEEDED!</p>
        </div>
        <div class="performance-card info">
            <p>Mean Absolute Error</p>
            <div class="performance-number">$17,200</div>
            <p>Prediction Accuracy</p>
        </div>
        <div class="performance-card info">
            <p>RMSE Error</p>
            <div class="performance-number">$29,091</div>
            <p>Root Mean Square Error</p>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # TOP FEATURES
    st.markdown(
        """
    <div class="insight-box">
        <h4>🏆 Top 5 Price Drivers</h4>
        <p><strong>1. Overall Quality (20%)</strong> - Quality is paramount!
        </p>
        <p><strong>2. Living Area (15%)</strong> - Size matters
        significantly</p>
        <p><strong>3. Garage Area (9%)</strong> - Parking adds value</p>
        <p><strong>4. Year Built (8%)</strong> - Newer houses command
        premium</p>
        <p><strong>5. Basement Area (8%)</strong> - Extra space is valuable</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # INSTRUCTIONS
    st.markdown(
        """
    <div class="instruction-box">
        <h4>📍 How to Use This Dashboard</h4>
        <ul>
            <li><strong>Correlation Study</strong>
            📊 - Explore which features drive house prices</li>
            <li><strong>Price Predictor</strong>
            🔮 - Get predictions for inherited houses or custom properties</li>
            <li><strong>Hypothesis</strong>
            🔬 - Review our data-driven hypotheses and validation</li>
            <li><strong>Model Performance</strong>
            📈 - Detailed metrics and feature importance analysis</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # FOOTER
    st.markdown(
        """
    <div class="footer-box">
        <p><strong>🤖 Machine Learning Model:</strong>
        Random Forest Regressor</p>
        <p><strong>⚙️ Optimization:
        </strong> 576 Hyperparameters Tuned with GridSearchCV</p>
        <p><strong>✅ Status:</strong> Production Ready | Live Deployment</p>
        <p style="font-size: 0.9em; opacity: 0.9;
        margin-top: 20px;">Heritage Housing Price Predictor |
        Powered by ML & Data Science</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ========== PAGE 2: CORRELATION STUDY ==========
elif page == "Correlation Study":
    st.title("📊 Correlation Analysis")

    st.write(
        """
    ## Feature Correlation with Sale Price

    This analysis shows which house attributes
    have the strongest relationship with sale price.
    """
    )

    # Load correlation data
    numeric_data = data.select_dtypes(include=[np.number])
    correlation = numeric_data.corr()["SalePrice"].sort_values(ascending=False)

    # Display top correlations
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Top 10 Correlated Features")
        top_10 = correlation.head(11)
        st.dataframe(top_10, use_container_width=True)

    with col2:
        st.write("### Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        correlation.head(11).plot(kind="barh", ax=ax, color="steelblue")
        ax.set_xlabel("Correlation Coefficient")
        ax.set_title("Features Most Correlated with SalePrice")
        st.pyplot(fig)

    st.write(
        """
    ### Key Insights
    - **OverallQual** (0.79): Overall quality is the strongest price driver
    - **GrLivArea** (0.71): Ground living area is very important
    - **GarageArea** (0.62): Garage size influences price
    - **YearBuilt** (0.52): Newer houses tend to be more expensive
    - **TotalBsmtSF** (0.61): Basement area matters
    """
    )

# ========== PAGE 3: PRICE PREDICTOR ==========
elif page == "Price Predictor":
    st.title("🔮 House Price Predictor")

    st.write(
        """
    ## Make Price Predictions

    Enter house attributes below to predict the sale price.
    """
    )

    # Hardcoded inherited houses data for Heroku
    inherited_df = pd.DataFrame(
        {
            "1stFlrSF": [896, 1262, 928, 926],
            "2ndFlrSF": [0.0, 0.0, 701.0, 678.0],
            "BedroomAbvGr": [3.0, 3.0, 3.0, 3.0],
            "BsmtExposure": [3, 1, 2, 3],
            "BsmtFinSF1": [706, 978, 486, 216],
            "BsmtFinType1": [2, 0, 2, 0],
            "BsmtUnfSF": [150, 284, 434, 540],
            "GarageArea": [548, 460, 608, 642],
            "GarageFinish": [1, 1, 1, 2],
            "GarageYrBlt": [2003.0, 1976.0, 2001.0, 1998.0],
            "GrLivArea": [856, 1262, 920, 961],
            "KitchenQual": [2, 3, 2, 2],
            "LotArea": [8450, 9600, 11250, 9550],
            "LotFrontage": [65.0, 80.0, 68.0, 60.0],
            "MasVnrArea": [196.0, 0.0, 162.0, 0.0],
            "OpenPorchSF": [61, 0, 42, 35],
            "OverallCond": [5, 8, 5, 5],
            "OverallQual": [7, 6, 7, 7],
            "TotalBsmtSF": [856, 1262, 920, 756],
            "YearBuilt": [2003, 1976, 2001, 1915],
            "YearRemodAdd": [2003, 1976, 2002, 1970],
        }
    )

    # Tabs for different sections
    tab1, tab2 = st.tabs(["Inherited Houses", "Custom Prediction"])

    # ========== TAB 1: INHERITED HOUSES ==========
    with tab1:
        st.write("### The 4 Inherited Houses")

        # Prepare inherited houses for prediction

        X_inherited = inherited_df.copy()

        # Remove features that don't exist in training data
        features_to_drop = ["EnclosedPorch", "WoodDeckSF"]
        X_inherited = X_inherited[
            [col for col in X_inherited.columns if col not in features_to_drop]
        ]

        # Reorder columns to match training data
        feature_names = data.drop("SalePrice", axis=1).columns.tolist()
        X_inherited = X_inherited[feature_names]

        predictions_inherited = model.predict(X_inherited)

        # Display inherited houses with predictions
        inherited_results = pd.DataFrame(
            {
                "House": ["House 1", "House 2", "House 3", "House 4"],
                "Year Built": inherited_df["YearBuilt"].values,
                "Overall Quality": inherited_df["OverallQual"].values,
                "Living Area (sqft)": inherited_df["GrLivArea"].values,
                "Predicted Price": predictions_inherited,
            }
        )

        st.dataframe(inherited_results, use_container_width=True)

        # Total predicted price
        total_price = predictions_inherited.sum()

        st.write("---")
        st.metric(
            label="Total Predicted Price for All 4 Houses",
            value=f"${total_price:,.2f}",
            delta=None,
        )

        st.write(
            f"""
        ### Summary
        - **House 1:** ${predictions_inherited[0]:,.2f}
        - **House 2:** ${predictions_inherited[1]:,.2f}
        - **House 3:** ${predictions_inherited[2]:,.2f}
        - **House 4:** ${predictions_inherited[3]:,.2f}
        - **TOTAL:** ${total_price:,.2f}
        """
        )

    # ========== TAB 2: CUSTOM PREDICTION ==========
    with tab2:
        st.write("### Predict Price for Any House")

        # Get feature names
        feature_names = data.drop("SalePrice", axis=1).columns.tolist()

        # Create input widgets
        st.write("#### Enter House Attributes")

        col1, col2 = st.columns(2)

        input_values = {}

        with col1:
            input_values["OverallQual"] = st.slider(
                "Overall Quality (1-10)", 1, 10, 6
            )
            input_values["GrLivArea"] = st.number_input(
                "Ground Living Area (sqft)", 300, 5000, 1500
            )
            input_values["GarageArea"] = st.number_input(
                "Garage Area (sqft)", 0, 2000, 500
            )
            input_values["TotalBsmtSF"] = st.number_input(
                "Total Basement Area (sqft)", 0, 5000, 1000
            )
            input_values["YearBuilt"] = st.slider(
                "Year Built", 1870, 2020, 2000)

        with col2:
            input_values["1stFlrSF"] = st.number_input(
                "1st Floor Area (sqft)", 300, 5000, 1200
            )
            input_values["2ndFlrSF"] = st.number_input(
                "2nd Floor Area (sqft)", 0, 3000, 0
            )
            input_values["BedroomAbvGr"] = st.slider("Bedrooms", 0, 8, 3)
            input_values["LotArea"] = st.number_input(
                "Lot Area (sqft)", 1000, 200000, 10000
            )
            input_values["LotFrontage"] = st.number_input(
                "Lot Frontage (ft)", 0, 300, 70
            )

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

            st.write(
                f"""
            Based on the entered attributes,
            this house would sell for
            approximately **${predicted_price:,.2f}**.

            **Note:** This prediction is based on the Random Forest model
            trained on 1,460 houses in Ames, Iowa.
            Actual prices may vary based on
            market conditions and other factors.
            """
            )

# ========== PAGE 4: HYPOTHESIS ==========
elif page == "Hypothesis":
    st.title("🔬 Project Hypothesis & Validation")

    st.write(
        """
    ## Project Hypotheses

    ### Hypothesis 1: Quality is the Primary Price Driver
    **Statement:** Overall quality has the
    strongest correlation with sale price.

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
    All hypotheses were validated.
    The machine learning model successfully predicts house prices
    with high accuracy, with overall quality being the most important factor.
    """
    )

# ========== PAGE 5: MODEL PERFORMANCE ==========
elif page == "Model Performance":
    st.title("📈 Model Performance Metrics")

    st.write(
        """
    ## Model Overview
    **Algorithm:** Random Forest Regressor
    **Training Data:** 1,168 houses (80%)
    **Test Data:** 292 houses (20%)
    """
    )

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

    st.write(
        """
    ## Performance Interpretation

    - **R² Score:** 0.8897 means the model explains 88.97% of price variance ✅
    - **MAE:** On average, predictions are off by $17,200
    - **RMSE:** Root mean squared error of $29,091
    - **Target Achievement:** R² ≥ 0.75 requirement **EXCEEDED** ✅

    ## Best Model Hyperparameters
    """
    )

    params = {
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "bootstrap": False,
    }

    for param, value in params.items():
        st.write(f"- **{param}:** {value}")

    st.write(
        """
    ## Top 5 Most Important Features
    """
    )

    feature_importance_data = {
        "Feature": [
            "OverallQual",
            "GrLivArea",
            "GarageArea",
            "YearBuilt",
            "TotalBsmtSF",
        ],
        "Importance (%)": [20.02, 15.14, 8.61, 8.47, 7.78],
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(
        feature_importance_data["Feature"],
        feature_importance_data["Importance (%)"],
        color="steelblue",
    )
    ax.set_xlabel("Importance (%)")
    ax.set_title("Top 5 Most Important Features")
    st.pyplot(fig)

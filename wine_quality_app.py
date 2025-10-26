import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pickle

# Page configuration
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #8B0000;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #A52A2A;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 2rem;
    }
    .good-wine {
        background-color: #90EE90;
        color: #006400;
        border: 3px solid #006400;
    }
    .bad-wine {
        background-color: #FFB6C1;
        color: #8B0000;
        border: 3px solid #8B0000;
    }
    .range-info {
        font-size: 12px;
        color: #666;
        font-style: italic;
        margin-top: -10px;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🍷 Wine Quality Prediction System")
st.markdown("""
    <div style='background-color: #f0f2f6; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h3 style='color: #8B0000; margin-top: 0;'>Welcome to the Wine Quality Predictor!</h3>
        <p style='font-size: 16px;'>This application uses Machine Learning to predict whether a wine is of <strong>Good Quality</strong> or <strong>Bad Quality</strong> based on its chemical properties.</p>
        <p style='font-size: 14px; margin-bottom: 0;'><em>Enter the wine features below and click "Predict Wine Quality" to get results.</em></p>
    </div>
    """, unsafe_allow_html=True)

# Load or train the model (you'll need to train and save the model first)
@st.cache_resource
def load_model():
    """
    Load the pre-trained model and scaler.
    If not available, return None and we'll show a message.
    """
    try:
        # Try to load saved model
        model = pickle.load(open('wine_quality_model.pkl', 'rb'))
        scaler = pickle.load(open('wine_quality_scaler.pkl', 'rb'))
        return model, scaler
    except:
        # If no saved model, return None (we'll handle training in the app)
        return None, None

# Initialize session state for model training
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
    st.session_state.model = None
    st.session_state.scaler = None

# Try to load existing model
if not st.session_state.model_trained:
    model, scaler = load_model()
    if model is not None:
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.model_trained = True

# Create three columns for input layout
st.markdown("### 📊 Enter Wine Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### Acidity Features")
    fixed_acidity = st.number_input(
        "Fixed Acidity", 
        min_value=4.6, 
        max_value=15.9, 
        value=7.4, 
        step=0.1,
        help="Tartaric acid content (g/dm³)"
    )
    st.markdown('<p class="range-info">Valid range: 4.6 – 15.9 g/dm³</p>', unsafe_allow_html=True)

    volatile_acidity = st.number_input(
        "Volatile Acidity", 
        min_value=0.12, 
        max_value=1.58, 
        value=0.70, 
        step=0.01,
        help="Acetic acid content (g/dm³)"
    )
    st.markdown('<p class="range-info">Valid range: 0.12 – 1.58 g/dm³</p>', unsafe_allow_html=True)

    citric_acid = st.number_input(
        "Citric Acid", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.0, 
        step=0.01,
        help="Citric acid content (g/dm³)"
    )
    st.markdown('<p class="range-info">Valid range: 0.0 – 1.0 g/dm³</p>', unsafe_allow_html=True)

    pH = st.number_input(
        "pH", 
        min_value=2.74, 
        max_value=4.01, 
        value=3.51, 
        step=0.01,
        help="pH value (0-14 scale)"
    )
    st.markdown('<p class="range-info">Valid range: 2.74 – 4.01</p>', unsafe_allow_html=True)

with col2:
    st.markdown("##### Chemical Properties")
    residual_sugar = st.number_input(
        "Residual Sugar", 
        min_value=0.9, 
        max_value=15.5, 
        value=1.9, 
        step=0.1,
        help="Sugar content (g/dm³)"
    )
    st.markdown('<p class="range-info">Valid range: 0.9 – 15.5 g/dm³</p>', unsafe_allow_html=True)

    chlorides = st.number_input(
        "Chlorides", 
        min_value=0.012, 
        max_value=0.611, 
        value=0.076, 
        step=0.001,
        help="Salt content (g/dm³)"
    )
    st.markdown('<p class="range-info">Valid range: 0.012 – 0.611 g/dm³</p>', unsafe_allow_html=True)

    sulphates = st.number_input(
        "Sulphates", 
        min_value=0.33, 
        max_value=2.0, 
        value=0.56, 
        step=0.01,
        help="Potassium sulphate content (g/dm³)"
    )
    st.markdown('<p class="range-info">Valid range: 0.33 – 2.0 g/dm³</p>', unsafe_allow_html=True)

    density = st.number_input(
        "Density", 
        min_value=0.99007, 
        max_value=1.00369, 
        value=0.9978, 
        step=0.0001,
        help="Density (g/cm³)"
    )
    st.markdown('<p class="range-info">Valid range: 0.99007 – 1.00369 g/cm³</p>', unsafe_allow_html=True)

with col3:
    st.markdown("##### Sulfur Dioxide & Alcohol")
    free_sulfur_dioxide = st.number_input(
        "Free Sulfur Dioxide", 
        min_value=1.0, 
        max_value=72.0, 
        value=11.0, 
        step=1.0,
        help="Free SO₂ content (mg/dm³)"
    )
    st.markdown('<p class="range-info">Valid range: 1 – 72 mg/dm³</p>', unsafe_allow_html=True)

    total_sulfur_dioxide = st.number_input(
        "Total Sulfur Dioxide", 
        min_value=6.0, 
        max_value=289.0, 
        value=34.0, 
        step=1.0,
        help="Total SO₂ content (mg/dm³)"
    )
    st.markdown('<p class="range-info">Valid range: 6 – 289 mg/dm³</p>', unsafe_allow_html=True)

    alcohol = st.number_input(
        "Alcohol", 
        min_value=8.4, 
        max_value=14.9, 
        value=9.4, 
        step=0.1,
        help="Alcohol content (%)"
    )
    st.markdown('<p class="range-info">Valid range: 8.4 – 14.9 %</p>', unsafe_allow_html=True)

    wine_id = st.number_input(
        "Wine ID", 
        min_value=0, 
        max_value=10000, 
        value=0, 
        step=1,
        help="Unique identifier for the wine"
    )
    st.markdown('<p class="range-info">Valid range: 0 – 10000</p>', unsafe_allow_html=True)

# Prepare input data
input_data = pd.DataFrame({
    'fixed acidity': [fixed_acidity],
    'volatile acidity': [volatile_acidity],
    'citric acid': [citric_acid],
    'residual sugar': [residual_sugar],
    'chlorides': [chlorides],
    'free sulfur dioxide': [free_sulfur_dioxide],
    'total sulfur dioxide': [total_sulfur_dioxide],
    'density': [density],
    'pH': [pH],
    'sulphates': [sulphates],
    'alcohol': [alcohol],
    'Id': [wine_id]
})

# Add some spacing
st.markdown("<br>", unsafe_allow_html=True)

# Create prediction button
col_left, col_center, col_right = st.columns([1, 2, 1])
with col_center:
    predict_button = st.button("🔮 Predict Wine Quality")

# Make prediction
if predict_button:
    if not st.session_state.model_trained:
        st.error("⚠️ Model not trained yet! Please upload your training data or use a pre-trained model.")

        # Show option to train model with sample data
        st.info("💡 **To use this application:**\n1. Train your model using the notebook\n2. Save the model using pickle\n3. Place the model files in the same directory as this app")

    else:
        try:
            # Scale the input data
            input_scaled = st.session_state.scaler.transform(input_data)

            # Make prediction
            prediction = st.session_state.model.predict(input_scaled)
            prediction_proba = st.session_state.model.predict_proba(input_scaled)

            # Display result
            st.markdown("<br>", unsafe_allow_html=True)

            if prediction[0] == 1:
                st.markdown(f"""
                    <div class='result-box good-wine'>
                        ✅ GOOD QUALITY WINE<br>
                        <span style='font-size: 18px;'>Confidence: {prediction_proba[0][1]*100:.2f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                    <div class='result-box bad-wine'>
                        ❌ BAD QUALITY WINE<br>
                        <span style='font-size: 18px;'>Confidence: {prediction_proba[0][0]*100:.2f}%</span>
                    </div>
                    """, unsafe_allow_html=True)

            # Show feature summary
            st.markdown("---")
            st.markdown("### 📋 Input Feature Summary")

            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(input_data.T.iloc[:6], use_container_width=True)
            with col2:
                st.dataframe(input_data.T.iloc[6:], use_container_width=True)

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

# Sidebar with information
with st.sidebar:
    st.markdown("## 📖 About")
    st.info("""
        This Wine Quality Prediction system uses a **Logistic Regression** model trained on wine chemical properties.

        **Model Performance:**
        - Training Accuracy: 76.26%
        - Testing Accuracy: 77.73%

        **Classification:**
        - **Good Wine**: Quality ≥ 6
        - **Bad Wine**: Quality < 6
    """)

    st.markdown("## 🎯 How to Use")
    st.markdown("""
        1. Enter all wine feature values
        2. Click "Predict Wine Quality"
        3. View the prediction result

        **Note:** All features are required for accurate prediction.
    """)

    st.markdown("## 📊 Feature Ranges")
    with st.expander("View typical ranges"):
        st.markdown("""
        - **Fixed Acidity**: 4.6 - 15.9 g/dm³
        - **Volatile Acidity**: 0.12 - 1.58 g/dm³
        - **Citric Acid**: 0.0 - 1.0 g/dm³
        - **Residual Sugar**: 0.9 - 15.5 g/dm³
        - **Chlorides**: 0.012 - 0.611 g/dm³
        - **Free SO₂**: 1 - 72 mg/dm³
        - **Total SO₂**: 6 - 289 mg/dm³
        - **Density**: 0.99007 - 1.00369 g/cm³
        - **pH**: 2.74 - 4.01
        - **Sulphates**: 0.33 - 2.0 g/dm³
        - **Alcohol**: 8.4 - 14.9 %
        """)

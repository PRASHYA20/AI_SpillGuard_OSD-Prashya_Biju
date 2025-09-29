import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Set page config
st.set_page_config(page_title="ML Model Deployment", layout="wide")

# Title
st.title("🚀 My ML Model Deployment")
st.write("Upload your data or input features to get predictions")

# Load your model (adjust the path/filename as needed)
@st.cache_resource
def load_model():
    try:
        # Try different common model formats
        model = joblib.load('model.pkl')
    except:
        try:
            model = pickle.load(open('model.pkl', 'rb'))
        except:
            st.error("Model file not found. Please ensure 'model.pkl' exists in the repo.")
            return None
    return model

model = load_model()

# Input methods
st.sidebar.header("Input Method")
input_method = st.sidebar.radio("Choose input method:", ["Manual Input", "File Upload"])

if input_method == "Manual Input":
    st.header("Manual Feature Input")
    
    # Example features - adjust based on your model
    col1, col2, col3 = st.columns(3)
    
    with col1:
        feature1 = st.number_input("Feature 1", value=0.0)
        feature2 = st.number_input("Feature 2", value=0.0)
    
    with col2:
        feature3 = st.number_input("Feature 3", value=0.0)
        feature4 = st.number_input("Feature 4", value=0.0)
    
    with col3:
        feature5 = st.number_input("Feature 5", value=0.0)
    
    # Create feature array
    features = np.array([[feature1, feature2, feature3, feature4, feature5]])
    
    if st.button("Predict"):
        if model is not None:
            try:
                prediction = model.predict(features)
                st.success(f"Prediction: {prediction[0]}")
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

else:  # File Upload
    st.header("Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            if st.button("Predict on Entire Dataset"):
                if model is not None:
                    predictions = model.predict(df)
                    df['predictions'] = predictions
                    
                    st.success("Predictions completed!")
                    st.write("Results:")
                    st.dataframe(df)
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Model info in sidebar
st.sidebar.header("Model Information")
if model is not None:
    st.sidebar.success("✅ Model loaded successfully")
    st.sidebar.write(f"Model type: {type(model).__name__}")
else:
    st.sidebar.error("❌ Model not loaded")

# Wine Quality Prediction App

A machine learning web application that predicts wine quality based on physicochemical properties. Built with Streamlit and deployed for real-time predictions.

## Quick Start

This app is already deployed and ready to use. Access it at: wine-quality-skylib.streamlit.io

## What This App Does

- Predicts wine quality using a trained Random Forest Classifier
- Validates inputs with proper chemical property ranges
- Displays confidence scores for each prediction
- Classifies wines as Good Quality (score >= 6) or Bad Quality (score < 6)

## How to Use the App

1. Enter wine features:
   - Input the 11 physicochemical properties of your wine
   - Valid ranges are displayed below each input field
   - All fields are required for accurate prediction

2. Click "Predict Wine Quality"

3. View results:
   - See if your wine is classified as Good or Bad quality
   - Check the confidence score (percentage)
   - Review your input feature summary

## Input Features

The app requires these 11 wine properties:

| Feature | Description | Valid Range |
|---------|-------------|-------------|
| Fixed Acidity | Tartaric acid in g/dm3 | 4.6 to 15.9 |
| Volatile Acidity | Acetic acid in g/dm3 | 0.12 to 1.58 |
| Citric Acid | Citric acid in g/dm3 | 0.0 to 1.0 |
| Residual Sugar | Sugar content in g/dm3 | 0.9 to 15.5 |
| Chlorides | Salt content in g/dm3 | 0.012 to 0.611 |
| Free Sulfur Dioxide | Free SO2 in mg/dm3 | 1 to 72 |
| Total Sulfur Dioxide | Total SO2 in mg/dm3 | 6 to 289 |
| Density | Density in g/cm3 | 0.99007 to 1.00369 |
| pH | pH value | 2.74 to 4.01 |
| Sulphates | Potassium sulphate in g/dm3 | 0.33 to 2.0 |
| Alcohol | Alcohol content in percent | 8.4 to 14.9 |

## Example Wines to Try

Example of Good Quality Wine:
Fixed Acidity: 7.8
Volatile Acidity: 0.3
Citric Acid: 0.4
Residual Sugar: 2.5
Chlorides: 0.065
Free Sulfur Dioxide: 20
Total Sulfur Dioxide: 80
Density: 0.9960
pH: 3.2
Sulphates: 0.75
Alcohol: 11.5

Expected Result: Good Quality Wine

Example of Bad Quality Wine:
Fixed Acidity: 10.5
Volatile Acidity: 1.2
Citric Acid: 0.0
Residual Sugar: 12.0
Chlorides: 0.3
Free Sulfur Dioxide: 5
Total Sulfur Dioxide: 220
Density: 1.0020
pH: 3.8
Sulphates: 0.4
Alcohol: 8.5

Expected Result: Bad Quality Wine

## Running Locally

If you want to run the app on your local machine:

Prerequisites:
- Python 3.8 or higher
- pip package manager

Installation Steps:

1. Clone the repository:
git clone https://github.com/skylib25/Wine-Quality-Prediction
cd Wine-Quality-Prediction

2. Install dependencies:
pip install streamlit pandas numpy scikit-learn

Or use requirements.txt if provided:
pip install -r requirements.txt

3. Run the app:
streamlit run wine_quality_app.py

4. Open your browser to http://localhost:8501

## Repository Contents

wine_quality_app.py - Main Streamlit application
wine_quality_model.pkl - Trained ML model (Random Forest)
wine_quality_scaler.pkl - Feature scaler (StandardScaler)
requirements.txt - Python dependencies
README.md - This file


## How It Works

1. User Input: You enter the wine's chemical properties
2. Preprocessing: Input is scaled using the saved scaler
3. Prediction: The trained Random Forest model classifies the wine
4. Confidence Score: Model returns probability of the prediction
5. Display: Results shown with color-coded output

## Dataset Reference

Dataset Source: https://www.kaggle.com/datasets/yasserh/wine-quality-dataset


## Deploy Your Own Version

Step 1: Fork this repository on GitHub

Step 2: Sign in to Streamlit Cloud at streamlit.io/cloud with your GitHub account

Step 3: Deploy the app

Click "New app" in Streamlit Cloud
Select your forked repository
Choose branch: main
Set main file path: wine_quality_app.py
Click "Deploy"

Your app will be live in minutes and will get its own URL.

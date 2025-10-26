"""
Wine Quality Model Training Script
Run this first to train and save your model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

print("="*70)
print("Wine Quality Model Training")
print("="*70)

# Load the dataset
print("\n1. Loading dataset...")
try:
    df = pd.read_csv("winequality.csv")
    print(f"   ✓ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
except FileNotFoundError:
    print("   ✗ Error: winequality.csv not found!")
    exit(1)

# Create binary classification target
print("\n2. Creating binary classification...")
df['quality_binary'] = df['quality'].apply(lambda x: 1 if x >= 6 else 0)
print(f"   ✓ Good Wine (≥6): {(df['quality_binary'] == 1).sum()}")
print(f"   ✓ Bad Wine (<6): {(df['quality_binary'] == 0).sum()}")

# Prepare features
print("\n3. Preparing features...")
X = df.drop(['quality', 'quality_binary'], axis=1)
y = df['quality_binary']
print(f"   ✓ Features: {X.shape}")

# Split data
print("\n4. Splitting data (80-20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   ✓ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# Scale features
print("\n5. Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("   ✓ Features scaled")

# Train model
print("\n6. Training Logistic Regression...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)
print("   ✓ Model trained")

# Evaluate
print("\n7. Evaluating model...")
test_acc = accuracy_score(y_test, model.predict(X_test_scaled))
print(f"   ✓ Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

# Save model and scaler
print("\n8. Saving model...")
with open('wine_quality_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("   ✓ Model saved: wine_quality_model.pkl")

with open('wine_quality_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("   ✓ Scaler saved: wine_quality_scaler.pkl")

print("\n" + "="*70)
print("✅ Training Complete! Run: streamlit run wine_quality_app.py")
print("="*70)

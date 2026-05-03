import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os

# src/ is one level inside root so paths go ../
PROCESSED_DIR = '../data'
ARTIFACTS_DIR = '../artifacts'
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def load_data():
    # cleaned.csv already has renamed columns (CustomerID no space)
    # columns: CustomerID, Recency, Frequency, Monetary, Country
    df = pd.read_csv(f'{PROCESSED_DIR}/cleaned.csv')
    print(f'loaded cleaned data: {df.shape}')
    print(df.head(3))
    return df


def scale_features(df):
    X = df[['Recency', 'Frequency', 'Monetary']].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    with open(f'{ARTIFACTS_DIR}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print(f'scaler saved → artifacts/scaler.pkl')
    print(f'X_scaled shape: {X_scaled.shape}')
    return X_scaled, scaler


if __name__ == '__main__':
    df = load_data()
    X_scaled, scaler = scale_features(df)
    print('feature engineering done')

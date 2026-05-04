from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import pickle
import io
import os

app = FastAPI(title='Customer Segmentation API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, '..', 'artifacts')

# load artifacts on startup
with open(f'{ARTIFACTS_DIR}/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open(f'{ARTIFACTS_DIR}/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open(f'{ARTIFACTS_DIR}/pca.pkl', 'rb') as f:
    pca = pickle.load(f)

df_segments = pd.read_csv(f'{ARTIFACTS_DIR}/segmented_customers.csv')


@app.get('/')
def home():
    return {'message': 'Customer Segmentation API is running'}


@app.get('/summary')
def summary():
    # segment counts
    counts = df_segments['Segment'].value_counts().to_dict()

    # avg RFM per segment
    avg_rfm = df_segments.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean().round(2)
    avg_rfm = avg_rfm.reset_index().to_dict(orient='records')

    return {
        'total_customers': len(df_segments),
        'segment_counts': counts,
        'avg_rfm_per_segment': avg_rfm
    }


@app.get('/segments')
def get_segments():
    data = df_segments[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Segment']].head(100)
    return data.to_dict(orient='records')


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    """
    upload a CSV with columns: Customer ID, InvoiceDate, Invoice, Quantity, Price
    returns segment for each customer
    """
    contents = await file.read()
    data = pd.read_csv(io.StringIO(contents.decode('utf-8')))

    # clean
    data = data.dropna(subset=['Customer ID'])
    data = data[~data['Invoice'].astype(str).str.startswith('C')]
    data = data[data['Quantity'] > 0]
    data = data[data['Price'] > 0]
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    data['Customer ID'] = data['Customer ID'].astype(int)
    data['TotalSpend']  = data['Quantity'] * data['Price']

    # build RFM
    snapshot_date = data['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = data.groupby('Customer ID').agg(
        Recency   = ('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
        Frequency = ('Invoice',     'nunique'),
        Monetary  = ('TotalSpend',  'sum')
    ).reset_index()
    rfm.rename(columns={'Customer ID': 'CustomerID'}, inplace=True)

    # scale and predict
    X = rfm[['Recency', 'Frequency', 'Monetary']]
    X_scaled = scaler.transform(X)
    rfm['Cluster'] = model.predict(X_scaled) if hasattr(model, 'predict') else model.fit_predict(X_scaled)

    return {
        'total_customers': len(rfm),
        'segments': rfm[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Cluster']].to_dict(orient='records')
    }

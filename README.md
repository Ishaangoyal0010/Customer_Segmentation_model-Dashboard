# Customer Segmentation Dashboard

## Project Structure
```
├── notebook/
│   └── eda.ipynb               # EDA + RFM feature engineering
├── data/
│   ├── raw/                    # put your kaggle CSV here
│   └── processed/              # cleaned.csv saved here by eda.ipynb
├── src/
│   ├── feature_engineering.py  # scaling, feature prep
│   └── train.py                # trains 4 models, picks best, saves artifacts
├── artifacts/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── pca.pkl
│   └── segmented_customers.csv
├── app/
│   ├── api.py                  # FastAPI backend
│   └── streamlit_app.py        # Streamlit dashboard
├── utils/
│   └── s3_utils.py             # upload/download to AWS S3
├── requirements.txt
└── mlflow.db                   # created when you run train.py
```

## How to run

### 1. install dependencies
```bash
pip install -r requirements.txt
```

### 2. put raw data
download UCI Online Retail dataset from Kaggle and put it in:
```
data/raw/ecommerce.csv
```

### 3. run EDA notebook
```bash
jupyter notebook notebook/eda.ipynb
```
this saves cleaned RFM data to `data/processed/cleaned.csv`

### 4. train models
```bash
cd src
python train.py
```
trains KMeans, Agglomerative, GaussianMixture, DBSCAN  
picks best model by silhouette score  
saves everything to `artifacts/`

### 5. check MLflow
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
open http://localhost:5000

### 6. run Streamlit dashboard
```bash
cd app
streamlit run streamlit_app.py
```

### 7. run FastAPI (optional)
```bash
cd app
uvicorn api:app --reload
```
open http://localhost:8000/docs

### 8. upload to S3 (optional)
```bash
# set env vars first
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret
export S3_BUCKET_NAME=your-bucket

cd utils
python s3_utils.py
```

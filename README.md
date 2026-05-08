# Customer Segmentation Dashboard

An end-to-end machine learning project that segments customers using RFM analysis and clustering algorithms, with a Streamlit dashboard for visualization and a FastAPI backend for predictions.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data & EDA | Pandas, NumPy, Matplotlib, Seaborn |
| ML | Scikit-learn (KMeans, Agglomerative, GMM, DBSCAN), PCA |
| Experiment Tracking | MLflow |
| Backend API | FastAPI, Uvicorn |
| Frontend | Streamlit, Plotly |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Cloud Storage | AWS S3 (optional) |

---

## Project Structure

```
customer-segmentation/
├── .github/
│   └── workflows/
│       └── deploy.yml          # CI/CD - test, build, push to Docker Hub
├── notebook/
│   └── eda.ipynb               # EDA + RFM feature engineering
├── data/
│   ├── online_retail_II.csv    # raw dataset (download from Kaggle)
│   └── cleaned.csv             # generated after running eda.ipynb
├── src/
│   ├── feature_engineering.py  # scaling + feature prep
│   └── train.py                # trains 4 models, picks best, saves artifacts
├── artifacts/
│   ├── best_model.pkl          # best clustering model
│   ├── scaler.pkl              # fitted StandardScaler
│   ├── pca.pkl                 # fitted PCA
│   ├── elbow_plot.png          # elbow curve for K selection
│   ├── pca_clusters.png        # PCA visualization of clusters
│   └── segmented_customers.csv # final labeled customer data
├── app/
│   ├── api.py                  # FastAPI backend
│   └── streamlit_app.py        # Streamlit dashboard
│ 
├── Dockerfile.backend          # Docker config for FastAPI
├── Dockerfile.frontend         # Docker config for Streamlit
├── docker-compose.yml          # runs both containers together
├── .dockerignore
├── requirements.txt
├── mlflow.db                   # auto created when train.py runs
└── README.md
```

---

## How It Works

### RFM Analysis
Each customer gets scored on 3 metrics:
- **Recency** — how many days since their last purchase (lower = better)
- **Frequency** — how many unique orders they placed
- **Monetary** — total amount they spent

### Models Trained
All 4 models run and are compared by silhouette score. Best one is saved automatically.
- KMeans
- Agglomerative Clustering
- Gaussian Mixture Model
- DBSCAN

### Business Segments
| Segment | Description | Action |
|---|---|---|
|  Champions | High spenders, bought recently | Premium offers, loyalty rewards |
|  Loyal Customers | Buy often, spend well | Upsell, cross-sell new products |
|  New Customers | Bought recently, not often | Welcome discounts, onboarding |
|  At Risk | Low spend, not recent | Win-back campaigns, special discounts |

---

## Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/customer-segmentation.git
cd customer-segmentation
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        
source venv/bin/activate     
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download dataset
Download **Online Retail II** dataset from Kaggle and place it at:
```
data/online_retail_II.csv
```

### 5. Run EDA notebook
```bash
jupyter notebook notebook/eda.ipynb
```
Run all cells — saves cleaned RFM data to `data/cleaned.csv`

### 6. Train models
```bash
python src/train.py
```
- Trains 4 clustering models
- Logs all runs to MLflow
- Picks best model by silhouette score
- Saves artifacts to `artifacts/`

### 7. Check MLflow experiments
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open `http://localhost:5000` to compare all model runs

### 8. Run the API
```bash
uvicorn app.api:app --reload
```
Open `http://localhost:8000/docs` to test endpoints

### 9. Run the dashboard
```bash
streamlit run app/streamlit_app.py
```
Open `http://localhost:8501`

---

## Run with Docker

```bash
# build and start both containers
docker-compose up --build

# backend  → http://localhost:8000/docs
# frontend → http://localhost:8501

# stop
docker-compose down
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/summary` | Segment counts + avg RFM per segment |
| GET | `/segments` | Top 100 customers with segment labels |
| POST | `/predict` | Upload new CSV → get predicted segments |

---

## CI/CD Pipeline

On every `git push` to `main`, GitHub Actions:

1. **Test job** — installs dependencies, checks all imports and syntax
2. **Build & Push job** (only if tests pass) — builds Docker images and pushes to Docker Hub

To set up, add these secrets in GitHub → Settings → Secrets:

| Secret | Value |
|---|---|
| `DOCKER_USERNAME` | your Docker Hub username |
| `DOCKER_PASSWORD` | your Docker Hub password |

Then on your local machine pull the latest images:
```bash
docker pull yourusername/segmentation-backend:latest
docker pull yourusername/segmentation-frontend:latest
docker-compose up -d
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from feature_engineering import load_data, scale_features

ARTIFACTS_DIR = '../artifacts'
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

BEST_K = 4  # change after looking at elbow plot


# load and scale
df = load_data()
X_scaled, scaler = scale_features(df)


# elbow method
inertia = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(range(2, 11), inertia, marker='o')
plt.title('Elbow Method - pick best K')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.savefig(f'{ARTIFACTS_DIR}/elbow_plot.png')
plt.show()
print('elbow plot saved → artifacts/elbow_plot.png')


# mlflow
mlflow.set_tracking_uri('sqlite:///../mlflow.db')
mlflow.set_experiment('customer_segmentation')

results = []


# 1 KMeans
with mlflow.start_run(run_name='KMeans'):
    model = KMeans(n_clusters=BEST_K, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    db  = davies_bouldin_score(X_scaled, labels)
    ch  = calinski_harabasz_score(X_scaled, labels)

    mlflow.log_params({'model': 'KMeans', 'n_clusters': BEST_K})
    mlflow.log_metrics({'silhouette': sil, 'davies_bouldin': db, 'calinski': ch})
    mlflow.sklearn.log_model(model, 'model')

    results.append({'model': 'KMeans', 'silhouette': sil, 'db': db, 'ch': ch, 'labels': labels, 'object': model})
    print(f'KMeans          | sil={sil:.3f} | db={db:.3f} | ch={ch:.1f}')


# 2 Agglomerative Clustering
with mlflow.start_run(run_name='Agglomerative'):
    model = AgglomerativeClustering(n_clusters=BEST_K)
    labels = model.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    db  = davies_bouldin_score(X_scaled, labels)
    ch  = calinski_harabasz_score(X_scaled, labels)

    mlflow.log_params({'model': 'Agglomerative', 'n_clusters': BEST_K})
    mlflow.log_metrics({'silhouette': sil, 'davies_bouldin': db, 'calinski': ch})

    results.append({'model': 'Agglomerative', 'silhouette': sil, 'db': db, 'ch': ch, 'labels': labels, 'object': model})
    print(f'Agglomerative   | sil={sil:.3f} | db={db:.3f} | ch={ch:.1f}')


# 3 Gaussian Mixture 
with mlflow.start_run(run_name='GaussianMixture'):
    model = GaussianMixture(n_components=BEST_K, random_state=42)
    model.fit(X_scaled)
    labels = model.predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    db  = davies_bouldin_score(X_scaled, labels)
    ch  = calinski_harabasz_score(X_scaled, labels)

    mlflow.log_params({'model': 'GaussianMixture', 'n_components': BEST_K})
    mlflow.log_metrics({'silhouette': sil, 'davies_bouldin': db, 'calinski': ch})
    mlflow.sklearn.log_model(model, 'model')

    results.append({'model': 'GaussianMixture', 'silhouette': sil, 'db': db, 'ch': ch, 'labels': labels, 'object': model})
    print(f'GaussianMixture | sil={sil:.3f} | db={db:.3f} | ch={ch:.1f}')


# 4 DBSCAN
with mlflow.start_run(run_name='DBSCAN'):
    model = DBSCAN(eps=0.5, min_samples=5)
    labels = model.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f'DBSCAN found {n_clusters} clusters')

    if n_clusters > 1:
        mask = labels != -1
        sil = silhouette_score(X_scaled[mask], labels[mask])
        db  = davies_bouldin_score(X_scaled[mask], labels[mask])
        ch  = calinski_harabasz_score(X_scaled[mask], labels[mask])
    else:
        sil, db, ch = 0, 99, 0
        print('DBSCAN only found 1 cluster, tune eps/min_samples')

    mlflow.log_params({'model': 'DBSCAN', 'eps': 0.5, 'min_samples': 5})
    mlflow.log_metrics({'silhouette': sil, 'davies_bouldin': db, 'calinski': ch, 'n_clusters_found': n_clusters})

    results.append({'model': 'DBSCAN', 'silhouette': sil, 'db': db, 'ch': ch, 'labels': labels, 'object': model})
    print(f'DBSCAN          | sil={sil:.3f} | db={db:.3f} | ch={ch:.1f}')


# compare models
results_df = pd.DataFrame(results)[['model', 'silhouette', 'db', 'ch']]
results_df = results_df.sort_values('silhouette', ascending=False)
print('\n--- model comparison ---')
print(results_df.to_string(index=False))


# pick best model
best        = max(results, key=lambda x: x['silhouette'])
best_labels = best['labels']
best_model  = best['object']
print(f"\nbest model : {best['model']}")
print(f"silhouette : {best['silhouette']:.3f}")


#assign segment labels
df['Cluster'] = best_labels

cluster_summary = df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
cluster_summary['MonetaryRank'] = cluster_summary['Monetary'].rank(ascending=False)
cluster_summary['RecencyRank']  = cluster_summary['Recency'].rank(ascending=True)

# label each customer directly based on their own RFM values
# using percentiles on actual data not cluster means(it caused imbalance before)

r_33  = df['Recency'].quantile(0.33)
r_66  = df['Recency'].quantile(0.66)
m_50  = df['Monetary'].quantile(0.50)
m_75  = df['Monetary'].quantile(0.75)
f_50  = df['Frequency'].quantile(0.50)

def assign_segment(row):
    if row['Monetary'] >= m_75 and row['Recency'] <= r_33:
        return 'Champions'
    elif row['Monetary'] >= m_50 and row['Frequency'] >= f_50:
        return 'Loyal Customers'
    elif row['Recency'] <= r_33:
        return 'New Customers'
    else:
        return 'At Risk'

df['Segment'] = df.apply(assign_segment, axis=1)
print('\nsegment counts:')
print(df['Segment'].value_counts())


# PCA plot
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
for seg in df['Segment'].unique():
    mask = df['Segment'] == seg
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=seg, alpha=0.5, s=15)
plt.title(f'Customer Segments PCA ({best["model"]})')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.tight_layout()
plt.savefig(f'{ARTIFACTS_DIR}/pca_clusters.png')
plt.show()


# saving fiea
with open(f'{ARTIFACTS_DIR}/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open(f'{ARTIFACTS_DIR}/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open(f'{ARTIFACTS_DIR}/pca.pkl', 'wb') as f:
    pickle.dump(pca, f)

df.to_csv(f'{ARTIFACTS_DIR}/segmented_customers.csv', index=False)

print('\nsaved:')
print('  artifacts/best_model.pkl')
print('  artifacts/scaler.pkl')
print('  artifacts/pca.pkl')
print('  artifacts/segmented_customers.csv')
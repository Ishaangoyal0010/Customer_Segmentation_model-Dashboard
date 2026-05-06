import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px

st.set_page_config(page_title='Customer Segmentation', layout='wide')

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, '..', 'artifacts')


# ── load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open(os.path.join(ARTIFACTS_DIR, 'best_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'pca.pkl'), 'rb') as f:
        pca = pickle.load(f)
    return model, scaler, pca


@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(ARTIFACTS_DIR, 'segmented_customers.csv'))


model, scaler, pca = load_artifacts()
df = load_data()


# ── sidebar filters ────────────────────────────────────────────────────────────
st.sidebar.title('Filters')

segments = ['All'] + list(df['Segment'].unique())
selected_segment = st.sidebar.selectbox('Select Segment', segments)

countries = ['All'] + sorted(df['Country'].dropna().unique().tolist())
selected_country = st.sidebar.selectbox('Select Country', countries)

filtered = df.copy()
if selected_segment != 'All':
    filtered = filtered[filtered['Segment'] == selected_segment]
if selected_country != 'All':
    filtered = filtered[filtered['Country'] == selected_country]


# ── header ─────────────────────────────────────────────────────────────────────
st.title('Customer Segmentation Dashboard')
st.markdown('Business-focused analysis using RFM + Clustering')
st.markdown('---')


# ── KPIs ───────────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric('Total Customers',    len(filtered))
col2.metric('Avg Recency (days)', f"{filtered['Recency'].mean():.0f}")
col3.metric('Avg Frequency',      f"{filtered['Frequency'].mean():.1f}")
col4.metric('Avg Spend',          f"£{filtered['Monetary'].mean():.0f}")
st.markdown('---')


# ── segment distribution ───────────────────────────────────────────────────────
st.subheader('Segment Distribution')
col1, col2 = st.columns(2)

seg_counts = df['Segment'].value_counts().reset_index()
seg_counts.columns = ['Segment', 'Count']

with col1:
    fig = px.pie(seg_counts, names='Segment', values='Count', title='Customer Segments')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.bar(seg_counts, x='Segment', y='Count', title='Segment Counts', color='Segment')
    st.plotly_chart(fig, use_container_width=True)


# ── RFM by segment ─────────────────────────────────────────────────────────────
st.subheader('RFM Breakdown by Segment')
rfm_avg = df.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean().round(2).reset_index()

col1, col2, col3 = st.columns(3)
with col1:
    fig = px.bar(rfm_avg, x='Segment', y='Recency', title='Avg Recency (lower = better)', color='Segment')
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.bar(rfm_avg, x='Segment', y='Frequency', title='Avg Frequency', color='Segment')
    st.plotly_chart(fig, use_container_width=True)
with col3:
    fig = px.bar(rfm_avg, x='Segment', y='Monetary', title='Avg Monetary (£)', color='Segment')
    st.plotly_chart(fig, use_container_width=True)


# ── PCA scatter ────────────────────────────────────────────────────────────────
st.subheader('Customer Clusters - PCA View')

X        = df[['Recency', 'Frequency', 'Monetary']]
X_scaled = scaler.transform(X)
X_pca    = pca.transform(X_scaled)

pca_df            = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
pca_df['Segment'] = df['Segment'].values

fig = px.scatter(pca_df, x='PCA1', y='PCA2', color='Segment',
                 title='Customer Segments (PCA)', opacity=0.6)
st.plotly_chart(fig, use_container_width=True)


# ── business insights ──────────────────────────────────────────────────────────
st.subheader('Business Insights')

insights = {
    'Champions':       '🏆 High spenders who bought recently → target with premium offers and loyalty rewards',
    'Loyal Customers': '💛 Buy often and spend well → upsell and cross-sell new products',
    'New Customers':   '🆕 Bought recently but not often → onboard them with welcome discounts',
    'At Risk':         '⚠️ Haven\'t bought in a while → win them back with special discount campaigns',
}

for seg, insight in insights.items():
    count = len(df[df['Segment'] == seg])
    pct   = count / len(df) * 100
    st.info(f'**{seg}** ({count} customers, {pct:.1f}%)  \n{insight}')


# ── upload new CSV and predict ─────────────────────────────────────────────────
st.markdown('---')
st.subheader('Predict Segments for New Data')

uploaded = st.file_uploader('Upload a CSV file (same format as training data)', type='csv')

if uploaded:
    new_df = pd.read_csv(uploaded, encoding='utf-8')
    st.write('uploaded data preview:')
    st.dataframe(new_df.head())

    new_df = new_df.dropna(subset=['Customer ID'])
    new_df = new_df[~new_df['Invoice'].astype(str).str.startswith('C')]
    new_df = new_df[new_df['Quantity'] > 0]
    new_df = new_df[new_df['Price'] > 0]
    new_df['InvoiceDate'] = pd.to_datetime(new_df['InvoiceDate'])
    new_df['Customer ID'] = new_df['Customer ID'].astype(int)
    new_df['TotalSpend']  = new_df['Quantity'] * new_df['Price']

    snapshot_date = new_df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm_new = new_df.groupby('Customer ID').agg(
        Recency   = ('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
        Frequency = ('Invoice',     'nunique'),
        Monetary  = ('TotalSpend',  'sum')
    ).reset_index()
    rfm_new.rename(columns={'Customer ID': 'CustomerID'}, inplace=True)

    X_new        = rfm_new[['Recency', 'Frequency', 'Monetary']]
    X_new_scaled = scaler.transform(X_new)

    if hasattr(model, 'predict'):
        rfm_new['Cluster'] = model.predict(X_new_scaled)
    else:
        rfm_new['Cluster'] = model.fit_predict(X_new_scaled)

    st.write('predicted segments:')
    st.dataframe(rfm_new)

    csv = rfm_new.to_csv(index=False).encode('utf-8')
    st.download_button('Download Results', csv, 'predicted_segments.csv', 'text/csv')


# ── raw data table ─────────────────────────────────────────────────────────────
st.markdown('---')
st.subheader('Customer Data')
st.dataframe(filtered[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Segment', 'Country']].reset_index(drop=True))
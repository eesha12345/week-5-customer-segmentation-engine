import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="Customer Segmentation Engine",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Segmentation Engine")

# Load Dataset
df = pd.read_csv("dataset.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Select Numeric Columns
numeric_df = df.select_dtypes(include=["number"])

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_df)

# Sidebar
st.sidebar.header("Clustering Settings")
n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)

# K-Means
kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# PCA Plot
st.subheader("Customer Segments (PCA)")

fig, ax = plt.subplots(figsize=(8,6))

scatter = ax.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=clusters,
    cmap="viridis"
)

ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_title("Customer Segmentation")

plt.colorbar(scatter)

st.pyplot(fig)

# Cluster Summary
st.subheader("Cluster Summary")

cluster_summary = df.groupby("Cluster").mean(numeric_only=True)

st.dataframe(cluster_summary)

# Download Button
csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Clustered Dataset",
    csv,
    "customer_segments.csv",
    "text/csv"
)

st.success("Customer Segmentation Completed Successfully!")
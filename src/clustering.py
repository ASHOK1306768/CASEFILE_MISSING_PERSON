
from sklearn.cluster import KMeans, DBSCAN
import numpy as np
import pandas as pd
import joblib

def fit_kmeans(df, n_clusters=5):
    """
    Fits K-Means clustering on GPS coordinates (PDF Module 4).
    """
    X = df[["Latitude", "Longitude"]].copy()
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    return model, labels

def fit_dbscan(df, eps_km=0.5, min_samples=5):
    """
    Fits DBSCAN clustering using Haversine distance metric (PDF Module 4).
    """
    # Convert lat/lon to radians for Haversine DBSCAN
    coords_rad = np.radians(df[["Latitude", "Longitude"]].values)
    kms_per_radian = 6371.0088
    epsilon = eps_km / kms_per_radian
    
    db = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm='ball_tree', metric='haversine')
    labels = db.fit_predict(coords_rad)
    return db, labels

def get_cluster_centroids(df, label_col="Location_Cluster"):
    """
    Computes mean lat/lon for each cluster hotspot.
    """
    if label_col not in df.columns:
        return pd.DataFrame()
    
    centroids = df.groupby(label_col).agg(
        Center_Latitude=("Latitude", "mean"),
        Center_Longitude=("Longitude", "mean"),
        Ping_Count=("Latitude", "count")
    ).reset_index()
    return centroids

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)


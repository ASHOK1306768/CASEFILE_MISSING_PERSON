
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import numpy as np
import pandas as pd
import joblib

def fit_isolation_forest(df, features, contamination=0.05):
    """
    Trains Isolation Forest anomaly detector (PDF Module 5).
    """
    X = df[features].fillna(0).copy()
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42
    )
    labels = model.fit_predict(X) # -1 for anomaly, 1 for normal
    raw_scores = model.decision_function(X) # lower score = more anomalous
    
    # Normalize score into a 0-100 anomaly severity index
    min_s, max_s = raw_scores.min(), raw_scores.max()
    if max_s > min_s:
        normalized_score = 100 * (1 - (raw_scores - min_s) / (max_s - min_s))
    else:
        normalized_score = np.zeros(len(df))
        
    return model, labels, normalized_score

def fit_local_outlier_factor(df, features, n_neighbors=20, contamination=0.05):
    """
    Fits Local Outlier Factor (LOF) model (PDF Module 5).
    """
    X = df[features].fillna(0).copy()
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    labels = lof.fit_predict(X)
    negative_factor = lof.negative_outlier_factor_
    return lof, labels, negative_factor

def fit_one_class_svm(df, features, nu=0.05):
    """
    Fits One-Class SVM model (PDF Module 5).
    """
    X = df[features].fillna(0).copy()
    oc_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=nu)
    labels = oc_svm.fit_predict(X)
    scores = oc_svm.decision_function(X)
    return oc_svm, labels, scores

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)


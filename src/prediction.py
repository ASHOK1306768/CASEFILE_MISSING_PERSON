
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def compute_top_k_accuracy(model, X_test, y_test, k=3):
    """
    Computes Top-K Accuracy for multi-class probable location classification (PDF Section 11).
    """
    probs = model.predict_proba(X_test)
    classes = model.classes_
    
    top_k_hits = 0
    for i, true_label in enumerate(y_test):
        top_k_idx = np.argsort(probs[i])[::-1][:k]
        top_k_classes = classes[top_k_idx]
        if true_label in top_k_classes:
            top_k_hits += 1
            
    return top_k_hits / len(y_test)

def train_location_model(df, features, target="Target_Area", model_type="random_forest"):
    """
    Trains supervised location prediction classifier and evaluates metrics (PDF Module 6 & Sec 11).
    Algorithms supported: Random Forest, XGBoost, Gradient Boosting, KNN.
    """
    data = df.dropna(subset=features + [target]).copy()
    
    # Encode categorical features if any
    X = data[features].copy()
    for col in X.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        
    y = data[target]
    
    min_class_count = y.value_counts().min()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if min_class_count >= 2 else None
    )
    
    if model_type == "xgb":
        # Label encode y for XGBoost
        le_y = LabelEncoder()
        y_train_enc = le_y.fit_transform(y_train)
        model = XGBClassifier(n_estimators=200, random_state=42, eval_metric="mlogloss")
        model.fit(X_train, y_train_enc)
        # Remap internal classes
        model.classes_ = le_y.classes_
        pred = le_y.inverse_transform(model.predict(X_test))
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(n_estimators=150, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
    elif model_type == "knn":
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
    else:
        model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
    # Metrics computation (PDF Section 11)
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, pred)
    
    top1 = compute_top_k_accuracy(model, X_test, y_test, k=1)
    top3 = compute_top_k_accuracy(model, X_test, y_test, k=min(3, len(model.classes_)))
    top5 = compute_top_k_accuracy(model, X_test, y_test, k=min(5, len(model.classes_)))
    
    metrics = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "Top-1 Accuracy": top1,
        "Top-3 Accuracy": top3,
        "Top-5 Accuracy": top5,
        "Confusion_Matrix": cm
    }
    
    # Feature Importances (PDF Module 9 - XAI)
    feature_importances = pd.Series(dtype=float)
    if hasattr(model, "feature_importances_"):
        feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
        
    return model, metrics, feature_importances

def predict_probabilities(model, sample_df, features):
    """
    Predicts probability distribution for target geographical areas.
    """
    X_sample = sample_df[features].copy()
    for col in X_sample.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X_sample[col] = le.fit_transform(X_sample[col].astype(str))
        
    probs = model.predict_proba(X_sample)[0]
    classes = model.classes_
    
    prob_df = pd.DataFrame({"Area": classes, "Probability": probs * 100})
    prob_df = prob_df.sort_values(by="Probability", ascending=False).reset_index(drop=True)
    return prob_df

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)



import pandas as pd
import numpy as np

def clean_gps_data(df, filter_india_bounds=True):
    """
    Cleans GPS trajectory dataset, filters invalid coordinates, and extracts temporal features.
    PDF Module 2 Compliance.
    """
    df = df.copy()
    df = df.drop_duplicates()
    
    # Standardize column names if needed
    if "User_ID" not in df.columns and "Person_ID" in df.columns:
        df["User_ID"] = df["Person_ID"]
        
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Latitude", "Longitude", "Timestamp"])
    
    # Coordinate filtering
    if filter_india_bounds:
        df = df[
            df["Latitude"].between(6.0, 37.5) &
            df["Longitude"].between(68.0, 97.5)
        ]
    else:
        df = df[
            df["Latitude"].between(-90.0, 90.0) &
            df["Longitude"].between(-180.0, 180.0)
        ]
        
    df = df.sort_values(["User_ID", "Timestamp"]).reset_index(drop=True)
    
    # Time-based Feature Extraction (PDF Module 2)
    df["Hour"] = df["Timestamp"].dt.hour
    df["Day"] = df["Timestamp"].dt.day
    df["Weekday"] = df["Timestamp"].dt.day_name()
    df["Month"] = df["Timestamp"].dt.month
    df["Is_Weekend"] = (df["Timestamp"].dt.dayofweek >= 5).astype(int)
    df["Is_Night"] = ((df["Hour"] >= 22) | (df["Hour"] <= 5)).astype(int)
    
    return df

def load_processed(path):
    return pd.read_csv(path, parse_dates=["Timestamp"])


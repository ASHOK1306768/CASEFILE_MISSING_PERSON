
import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6371000

def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    initial_bearing = np.arctan2(y, x)
    initial_bearing = np.degrees(initial_bearing)
    return (initial_bearing + 360) % 360

def add_movement_features(df):
    """
    Computes spatial-temporal movement features: Distance (m), Speed (km/h), 
    Acceleration (m/s²), Bearing (degrees), and Stationary Stops (PDF Module 3).
    """
    df = df.copy().sort_values(["User_ID", "Timestamp"]).reset_index(drop=True)
    
    df["Prev_Latitude"] = df.groupby("User_ID")["Latitude"].shift(1)
    df["Prev_Longitude"] = df.groupby("User_ID")["Longitude"].shift(1)
    df["Time_Diff_Sec"] = df.groupby("User_ID")["Timestamp"].diff().dt.total_seconds()
    
    df["Distance_M"] = haversine_m(
        df["Prev_Latitude"], df["Prev_Longitude"],
        df["Latitude"], df["Longitude"]
    )
    df.loc[df["Prev_Latitude"].isna(), "Distance_M"] = 0.0
    
    # Speed in km/h
    df["Speed_KMH"] = (df["Distance_M"] / df["Time_Diff_Sec"].replace(0, np.nan)) * 3.6
    df["Speed_KMH"] = df["Speed_KMH"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    # Acceleration in m/s²
    df["Prev_Speed_MS"] = (df.groupby("User_ID")["Speed_KMH"].shift(1) / 3.6).fillna(0.0)
    df["Current_Speed_MS"] = df["Speed_KMH"] / 3.6
    df["Acceleration_MS2"] = (df["Current_Speed_MS"] - df["Prev_Speed_MS"]) / df["Time_Diff_Sec"].replace(0, np.nan)
    df["Acceleration_MS2"] = df["Acceleration_MS2"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    # Bearing / Direction of movement
    df["Bearing"] = calculate_bearing(
        df["Prev_Latitude"].fillna(df["Latitude"]),
        df["Prev_Longitude"].fillna(df["Longitude"]),
        df["Latitude"],
        df["Longitude"]
    ).fillna(0.0)
    
    # Stationary Stop Flag (Speed < 2 km/h and time gap >= 5 mins)
    df["Is_Stop"] = ((df["Speed_KMH"] < 2.0) & (df["Time_Diff_Sec"] >= 300)).astype(int)
    
    return df

def user_summary(df):
    """
    Summarizes movement profile per user (PDF Module 3).
    """
    agg_dict = {
        "Total_Distance_M": ("Distance_M", "sum"),
        "Average_Speed_KMH": ("Speed_KMH", "mean"),
        "Maximum_Speed_KMH": ("Speed_KMH", "max"),
        "Total_Pings": ("Timestamp", "count")
    }
    if "Location_Cluster" in df.columns:
        agg_dict["Locations_Visited"] = ("Location_Cluster", "nunique")
        
    return df.groupby("User_ID").agg(**agg_dict).reset_index()

def daily_user_summary(df):
    """
    Aggregates daily trajectory features per user for ML Anomaly Detection (PDF Module 3 & 5).
    """
    df["Date_Str"] = df["Timestamp"].dt.strftime("%Y-%m-%d")
    daily = df.groupby(["User_ID", "Date_Str"]).agg(
        Total_Daily_Distance_KM=("Distance_M", lambda x: x.sum() / 1000.0),
        Avg_Daily_Speed_KMH=("Speed_KMH", "mean"),
        Max_Daily_Speed_KMH=("Speed_KMH", "max"),
        Night_Pings_Count=("Is_Night", "sum"),
        Stop_Count=("Is_Stop", "sum"),
        Total_Pings=("Timestamp", "count")
    ).reset_index()
    return daily


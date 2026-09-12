import numpy as np
import pandas as pd

def calculate_search_priority_score(
    ml_prob,
    historical_freq=0.5,
    route_sim=0.5,
    distance_rel=0.5,
    time_rel=0.5,
    anomaly_evid=0.5,
    weights=None
):
    """
    Calculates Search Priority Score (SPS 0-100) per PDF Module 8 Guidelines.
    Default Weights:
    - ML prediction probability: 30%
    - Historical visit frequency: 20%
    - Route similarity: 15%
    - Distance relevance: 15%
    - Time relevance: 10%
    - Anomaly evidence: 10%
    """
    if weights is None:
        weights = {
            'ml_prob': 0.30,
            'historical_freq': 0.20,
            'route_sim': 0.15,
            'distance_rel': 0.15,
            'time_rel': 0.10,
            'anomaly_evid': 0.10
        }
        
    w_score = (
        (ml_prob * weights['ml_prob']) +
        (historical_freq * 100 * weights['historical_freq']) +
        (route_sim * 100 * weights['route_sim']) +
        (distance_rel * 100 * weights['distance_rel']) +
        (time_rel * 100 * weights['time_rel']) +
        (anomaly_evid * 100 * weights['anomaly_evid'])
    )
    score = float(np.clip(w_score, 0, 100))
    
    if score <= 30:
        priority = 'Low'
    elif score <= 60:
        priority = 'Medium'
    elif score <= 80:
        priority = 'High'
    else:
        priority = 'Very High'
        
    return score, priority

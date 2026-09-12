# CASEFILE: Data Source Documentation (PDF Module 1 Compliance)

## Overview
This document logs all dataset sources, licenses, schemas, record counts, and usage guidelines as mandated by Module 1 of the Advanced Machine Learning Project Guidelines.

---

### 1. Indian GPS Trajectory Simulation Dataset
- **Dataset Name**: Indian Metros Human Movement GPS Trajectories (`indian_trajectories.csv`)
- **Source / Geographic Focus**: Simulated trajectories based on OpenStreetMap road corridors & public transport routes (Delhi NCR, Mumbai MMR, Bengaluru).
- **Number of Records**: 5,000+ trajectory GPS pings across 30 anonymous user profiles (`IND-USR-001` to `IND-USR-030`).
- **Features**: `User_ID`, `Latitude`, `Longitude`, `Altitude`, `Date`, `Time`, `Timestamp`.
- **Derived Features**: `Distance_M`, `Speed_KMH`, `Acceleration_MS2`, `Bearing`, `Is_Stop`, `Is_Night`, `Hour`, `Day`.
- **License / Ethical Note**: Fictional simulation. No real individual tracking or PII.
- **Purpose**: Training K-Means/DBSCAN movement pattern clustering, Isolation Forest trajectory anomaly detection, and Markov Chain route transition modeling.

---

### 2. Geographical Points of Interest (POI) & Map Layers
- **Source**: OpenStreetMap (OSM) & Data.gov.in public spatial layers.
- **URL**: https://www.openstreetmap.org/ & https://www.data.gov.in/
- **Key Categories**:
  - Interstate Bus Terminals (ISBT Kashmere Gate, Anand Vihar, Majestic)
  - Junction Railway Stations (New Delhi Railway Station, CST Mumbai, KSR Bengaluru)
  - Highway Toll Plazas & Border Checkpoints (Delhi-Gurgaon Expressway Toll, Vashi Plaza)
  - Police Station Jurisdiction Boundaries
- **Purpose**: Mapping missing person last seen locations, target area probability zones, and search radius buffers.

---

### 3. Fictional Indian Missing-Person Case Registry
- **Dataset Name**: Indian Missing-Person Synthetic Case Records (`synthetic_case_data.csv`)
- **Number of Records**: 100 benchmark fictional case records (`MP-DEL-2026-001` to `MP-BLR-2026-100`).
- **Features (PDF Section 5 Compliant)**:
  - `Case_ID`: Unique case tracking identifier.
  - `Person_ID`: Anonymous user link.
  - `Age_Group`: Demographics (`18–25`, `26–40`, `41–60`, `60+`).
  - `Gender`: Male / Female.
  - `Last_Latitude`, `Last_Longitude`: Last confirmed GPS signal coordinates.
  - `Last_Seen_Time`: Time of disappearance (IST).
  - `Day`: Day of week.
  - `Weather`: Weather conditions.
  - `Usual_Area`: Historical frequent stay location.
  - `Average_Distance`: Average historical travel distance (km).
  - `Average_Speed`: Average speed (km/h).
  - `Previous_Area`: Location visited prior to disappearance.
  - `Time_Since_Last_Seen`: Hours elapsed.
  - `Target_Area`: Simulated target geographical destination.
  - `FIR_Number` & `Police_Station`: Official Indian police case reference details.
- **Purpose**: Supervised Machine Learning (Random Forest, XGBoost) for Probable Location Prediction and Search Priority Score (SPS) evaluation.

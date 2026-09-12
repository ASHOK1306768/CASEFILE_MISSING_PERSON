# CASEFILE: An AI-Powered Missing Person Investigation and Probable Location Prediction System Using Advanced Machine Learning (Indian Version)

## Academic Simulation Warning
This project is a fictional academic simulation developed strictly per the **Advanced Machine Learning Project Guidelines**. It is designed as an investigative-support system and must **NEVER** be used to make real-world decisions or track actual individuals.

---

## 1. Project Overview & Pipeline
The **CASEFILE (Indian Version)** system analyzes historical GPS movement patterns, temporal indicators, and demographic context across major Indian metros (Delhi NCR, Mumbai MMR, Bengaluru) to assist missing person investigators in prioritizing search zones.

### End-to-End Pipeline (13 Phases)
1. **Data Collection**: Indian GPS trajectory datasets and OpenStreetMap POI layers (`data/DATA_SOURCES.md`).
2. **Data Cleaning**: Invalid coordinate filtering (Indian spatial bounds 6.0°N–37.5°N, 68.0°E–97.5°E), duplicate removal, missing value imputation.
3. **Feature Engineering**: Haversine distance (m), instantaneous speed (km/h), acceleration (m/s²), movement bearing, stationary stops, time features (`Is_Night`, `Is_Weekend`).
4. **Exploratory Data Analysis**: Trajectory distribution analysis, temporal movement heatmaps across Indian cities.
5. **Movement Pattern Analysis**: **K-Means Clustering** and **DBSCAN** stay-point hotspot identification.
6. **Anomaly Detection**: **Isolation Forest**, **Local Outlier Factor (LOF)**, and **One-Class SVM** detecting irregular movement deviations.
7. **Location Prediction**: Supervised **Random Forest**, **XGBoost**, **Gradient Boosting**, and **KNN** multi-class classification for probable destination areas.
8. **Route Prediction**: **Markov Chain State Transition Matrix** predicting sequential location movement corridors.
9. **Search Priority Score (SPS)**: Multi-criteria weighted decision score (0–100) assigning priority tiers (`Low`, `Medium`, `High`, `Very High`).
10. **Explainable AI (XAI)**: SHAP / Feature Importance attribution for predictive factors.
11. **Interactive Geospatial Map**: **Folium & GeoPandas** map with search priority radius buffers (1km, 5km, 10km), trajectory heatmaps, and last seen markers.
12. **Streamlit Tactical Dashboard**: 6-tab interactive web interface (`app/app.py`).
13. **Final Investigation Report**: Comprehensive documentation (`reports/investigation_report.pdf`).

---

## 2. Project Folder Structure (PDF Section 10 Compliant)

```text
CASEFILE_MISSING_PERSON/
│
├── data/
│   ├── raw/                  # Indian GPS trajectory raw logs
│   ├── processed/            # Cleaned trajectories with spatial & movement features
│   ├── synthetic/            # Indian missing person case registry (100 synthetic FIR records)
│   └── DATA_SOURCES.md       # Data Source Documentation (PDF Module 1)
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_clustering.ipynb
│   ├── 05_anomaly_detection.ipynb
│   ├── 06_location_prediction.ipynb
│   └── 07_route_prediction.ipynb
│
├── models/
│   ├── clustering_model.pkl
│   ├── anomaly_model.pkl
│   ├── location_model.pkl
│   ├── markov_route_model.pkl
│   └── trajectory_scaler.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── clustering.py
│   ├── anomaly_detection.py
│   ├── prediction.py
│   ├── search_priority.py
│   └── mapping.py
│
├── app/
│   └── app.py                # Full-featured Streamlit investigation dashboard
│
├── reports/
│   ├── investigation_report.pdf
│   └── presentation.pptx
│
├── requirements.txt           # Verified Python package dependencies
└── README.md
```

---

## 3. Quick Start & Execution

### Installation
```bash
pip install -r requirements.txt
```

### Run Jupyter Notebooks Pipeline
```bash
jupyter notebook
```
Or execute all phase notebooks under `notebooks/01_data_collection.ipynb` to `notebooks/07_route_prediction.ipynb`.

### Launch Streamlit Investigation Dashboard
```bash
streamlit run app/app.py
```

---

## 4. Minimum ML Requirements & Algorithms
- **Movement Clustering**: K-Means & DBSCAN
- **Anomaly Detection**: Isolation Forest, Local Outlier Factor (LOF), One-Class SVM
- **Location Prediction**: Random Forest & XGBoost (Evaluated via Accuracy, Precision, Recall, F1, Top-1, Top-3, Top-5 Accuracy)
- **Route Prediction**: Markov Chain Transition Matrix
- **Search Priority Score (SPS)**: 6-factor weighted calculation matrix

---

## 5. Ethical Guidelines (PDF Section 16)
- **Fictional Identities**: All case names, FIR numbers, and identities are completely synthetic.
- **No PII**: Zero personally identifiable information.
- **Probabilistic Nature**: Predictions are decision support probabilities, not proof.
- **Anomaly Limitation**: Movement anomalies indicate baseline deviations, not evidence of criminal behavior.


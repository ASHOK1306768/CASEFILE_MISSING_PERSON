
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
n = 500
areas = ["Area_A", "Area_B", "Area_C", "Area_D", "Area_E"]

centers = {
    "Area_A": (39.90, 116.38),
    "Area_B": (39.92, 116.42),
    "Area_C": (39.88, 116.35),
    "Area_D": (39.94, 116.34),
    "Area_E": (39.86, 116.45),
}

rows = []
for i in range(n):
    area = rng.choice(areas, p=[.34, .27, .18, .13, .08])
    lat, lon = centers[area]
    ts = pd.Timestamp("2008-10-01") + pd.to_timedelta(
        int(rng.integers(0, 30*24*3600)), unit="s"
    )
    rows.append([
        f"MP-{2026}-{i+1:03d}",
        f"P-{i%20+1:03d}",
        rng.choice(["18-25", "26-35", "36-50"]),
        rng.choice(["Male", "Female"]),
        lat + rng.normal(0, .008),
        lon + rng.normal(0, .008),
        ts,
        ts.day_name(),
        rng.choice(["Clear", "Cloudy", "Rain"]),
        area,
        float(rng.uniform(2, 15)),
        float(rng.uniform(3, 50)),
        area,
        float(rng.uniform(1, 24)),
        area
    ])

cols = [
    "Case_ID","Person_ID","Age_Group","Gender","Last_Latitude",
    "Last_Longitude","Last_Seen_Time","Day","Weather","Usual_Area",
    "Average_Distance","Average_Speed","Previous_Area",
    "Time_Since_Last_Seen","Target_Area"
]
out = pd.DataFrame(rows, columns=cols)
Path("data/synthetic").mkdir(parents=True, exist_ok=True)
out.to_csv("data/synthetic/synthetic_case_data.csv", index=False)
print(out.head())

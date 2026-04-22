"""
generate_data.py
================
Reproducible synthetic data generator for S2.T1.2 — Stunting Risk Heatmap.
Run:  python generate_data.py
Outputs:
  - data/households.csv          (2,500 rows)
  - data/gold_stunting_flag.csv  (300 labelled rows, 50/50 pos/neg)
  - data/districts.geojson       (5 Rwandan districts, simplified polygons)
"""

import json
import os
import random

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)

os.makedirs("data", exist_ok=True)

# ── 1. District / Sector definitions ─────────────────────────────────────────
DISTRICTS = {
    "Nyarugenge": {
        "sectors": ["Gitega", "Kigali", "Kimisagara", "Nyamirambo", "Rwezamenyo"],
        "urban_prob": 0.90,
        "center": (-1.945, 30.058),
    },
    "Gasabo": {
        "sectors": ["Gisozi", "Jabana", "Jali", "Kinyinya", "Ndera"],
        "urban_prob": 0.75,
        "center": (-1.895, 30.112),
    },
    "Kicukiro": {
        "sectors": ["Gahanga", "Gatenga", "Kagarama", "Kanombe", "Niboye"],
        "urban_prob": 0.70,
        "center": (-1.980, 30.103),
    },
    "Bugesera": {
        "sectors": ["Gashora", "Juru", "Kamabuye", "Mareba", "Ntarama"],
        "urban_prob": 0.20,
        "center": (-2.165, 30.245),
    },
    "Rulindo": {
        "sectors": ["Base", "Burega", "Bushoki", "Kinihira", "Masoro"],
        "urban_prob": 0.15,
        "center": (-1.730, 29.997),
    },
}

WATER_SOURCES = ["piped", "protected_well", "unprotected_well", "surface"]
WATER_RISK    = {"piped": 0, "protected_well": 0.2, "unprotected_well": 0.6, "surface": 1.0}

SANITATION_TIERS = ["high", "medium", "low"]
SANIT_RISK       = {"high": 0, "medium": 0.4, "low": 1.0}

INCOME_BANDS = ["high", "middle", "low"]
INCOME_RISK  = {"high": 0, "middle": 0.3, "low": 0.8}


def sample_location(center, spread=0.08):
    lat = center[0] + rng.uniform(-spread, spread)
    lon = center[1] + rng.uniform(-spread, spread)
    return round(lat, 6), round(lon, 6)


def stunting_probability(row):
    """Logistic-style ground truth used only during generation."""
    z = (
        -1.5
        + 1.8 * WATER_RISK[row["water_source"]]
        + 1.5 * SANIT_RISK[row["sanitation_tier"]]
        + 1.2 * INCOME_RISK[row["income_band"]]
        + 0.9 * max(0, (2 - row["avg_meal_count"]) / 2)
        + 0.4 * (row["children_under5"] / 5)
        + rng.normal(0, 0.3)
    )
    return float(1 / (1 + np.exp(-z)))


# ── 2. Generate households ────────────────────────────────────────────────────
rows = []
hid = 1
for district, meta in DISTRICTS.items():
    n_households = 500  # 5 districts × 500 = 2,500
    for _ in range(n_households):
        sector = random.choice(meta["sectors"])
        urban = rng.random() < meta["urban_prob"]

        water_probs = (
            [0.55, 0.30, 0.10, 0.05] if urban else [0.10, 0.25, 0.40, 0.25]
        )
        water = rng.choice(WATER_SOURCES, p=water_probs)

        sanit_probs = (
            [0.50, 0.35, 0.15] if urban else [0.10, 0.35, 0.55]
        )
        sanit = rng.choice(SANITATION_TIERS, p=sanit_probs)

        income_probs = (
            [0.25, 0.50, 0.25] if urban else [0.05, 0.30, 0.65]
        )
        income = rng.choice(INCOME_BANDS, p=income_probs)

        children = int(rng.integers(0, 6))
        meals = round(float(rng.uniform(1.0, 3.5)), 1) if income == "low" else round(float(rng.uniform(1.5, 3.5)), 1)

        lat, lon = sample_location(meta["center"])

        row = {
            "household_id": f"HH{hid:04d}",
            "lat": lat,
            "lon": lon,
            "district": district,
            "sector": sector,
            "urban": int(urban),
            "children_under5": children,
            "avg_meal_count": meals,
            "water_source": water,
            "sanitation_tier": sanit,
            "income_band": income,
        }
        rows.append(row)
        hid += 1

households = pd.DataFrame(rows)
households.to_csv("data/households.csv", index=False)
print(f"✓ households.csv — {len(households)} rows")

# ── 3. Gold labels (300 rows, 50/50 pos/neg) ─────────────────────────────────
households["_prob"] = households.apply(stunting_probability, axis=1)

# Pick 150 highest-prob (positives) + 150 lowest-prob (negatives)
sorted_df = households.sort_values("_prob")
negatives = sorted_df.head(150).copy()
positives = sorted_df.tail(150).copy()
negatives["stunting_flag"] = 0
positives["stunting_flag"] = 1

gold = pd.concat([positives, negatives]).sample(frac=1, random_state=SEED)
gold = gold[["household_id", "stunting_flag"]].reset_index(drop=True)
gold.to_csv("data/gold_stunting_flag.csv", index=False)
print(f"✓ gold_stunting_flag.csv — {len(gold)} rows "
      f"({gold['stunting_flag'].sum()} positive, {(gold['stunting_flag']==0).sum()} negative)")

households.drop(columns=["_prob"], inplace=True)

# ── 4. districts.geojson (approximate simplified polygons) ───────────────────
# Coordinates are manually simplified bounding boxes around each district.
geojson = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"district": "Nyarugenge"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [30.020, -1.970], [30.095, -1.970],
                    [30.095, -1.920], [30.020, -1.920], [30.020, -1.970]
                ]]
            }
        },
        {
            "type": "Feature",
            "properties": {"district": "Gasabo"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [30.070, -1.930], [30.170, -1.930],
                    [30.170, -1.850], [30.070, -1.850], [30.070, -1.930]
                ]]
            }
        },
        {
            "type": "Feature",
            "properties": {"district": "Kicukiro"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [30.060, -2.025], [30.150, -2.025],
                    [30.150, -1.945], [30.060, -1.945], [30.060, -2.025]
                ]]
            }
        },
        {
            "type": "Feature",
            "properties": {"district": "Bugesera"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [30.175, -2.250], [30.320, -2.250],
                    [30.320, -2.080], [30.175, -2.080], [30.175, -2.250]
                ]]
            }
        },
        {
            "type": "Feature",
            "properties": {"district": "Rulindo"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [29.940, -1.800], [30.060, -1.800],
                    [30.060, -1.650], [29.940, -1.650], [29.940, -1.800]
                ]]
            }
        }
    ]
}

with open("data/districts.geojson", "w") as f:
    json.dump(geojson, f, indent=2)
print("✓ districts.geojson — 5 districts")
print("\nAll data files written to data/")

"""
risk_scorer.py
==============
Stunting Risk Scorer for S2.T1.2.

Hybrid approach:
  1. Rule-based component  → fast, explainable, survives zero-data scenarios.
  2. Logistic regression   → calibrated on gold_stunting_flag.csv when available.

Public API
----------
    from risk_scorer import score, score_dataframe, train

    # Single household (dict or pd.Series)
    result = score(household)
    # result = {"risk_score": 0.72, "risk_label": "High", "top_drivers": [...]}

    # Full dataframe
    scored_df = score_dataframe(df)

    # (Re-)train logistic model
    train("data/households.csv", "data/gold_stunting_flag.csv")
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_PATH = Path("data/risk_model.pkl")

WATER_RISK = {
    "piped": 0.0,
    "protected_well": 0.2,
    "unprotected_well": 0.6,
    "surface": 1.0,
}
SANIT_RISK = {"high": 0.0, "medium": 0.4, "low": 1.0}
INCOME_RISK = {"high": 0.0, "middle": 0.3, "low": 0.8}

DRIVER_LABELS = {
    "meal_deficit": "Low meal frequency (< 2 meals/day)",
    "unsafe_water": "Unsafe water source",
    "poor_sanitation": "Poor sanitation",
    "low_income": "Low household income",
    "many_children": "High number of children under 5",
}

THRESHOLDS = {"High": 0.60, "Medium": 0.35, "Low": 0.0}


# ── Feature engineering ───────────────────────────────────────────────────────
def _features(h: Union[dict, pd.Series]) -> dict:
    """Return normalised numeric features from a household record."""
    meal_count = float(h.get("avg_meal_count", 2.0))
    children = float(h.get("children_under5", 0))
    water = str(h.get("water_source", "piped"))
    sanit = str(h.get("sanitation_tier", "medium"))
    income = str(h.get("income_band", "middle"))

    return {
        "meal_deficit": max(0.0, (3.0 - meal_count) / 3.0),
        "unsafe_water": WATER_RISK.get(water, 0.5),
        "poor_sanitation": SANIT_RISK.get(sanit, 0.4),
        "low_income": INCOME_RISK.get(income, 0.3),
        "many_children": min(children / 5.0, 1.0),
    }


def _rule_based_score(feats: dict) -> float:
    """Weighted rule-based risk score → [0, 1]."""
    weights = {
        "meal_deficit": 0.30,
        "unsafe_water": 0.25,
        "poor_sanitation": 0.25,
        "low_income": 0.12,
        "many_children": 0.08,
    }
    raw = sum(feats[k] * w for k, w in weights.items())
    return float(np.clip(raw, 0.0, 1.0))


def _top_drivers(feats: dict, n: int = 3) -> list[str]:
    """Return the top-n driver labels ordered by contribution."""
    weights = {
        "meal_deficit": 0.30,
        "unsafe_water": 0.25,
        "poor_sanitation": 0.25,
        "low_income": 0.12,
        "many_children": 0.08,
    }
    contributions = {k: feats[k] * weights[k] for k in feats}
    top = sorted(contributions, key=contributions.get, reverse=True)[:n]
    return [DRIVER_LABELS[k] for k in top if contributions[k] > 0]


def _risk_label(score_val: float) -> str:
    for label, thresh in THRESHOLDS.items():
        if score_val >= thresh:
            return label
    return "Low"


# ── Model training ────────────────────────────────────────────────────────────
def train(
    households_path: str = "data/households.csv",
    gold_path: str = "data/gold_stunting_flag.csv",
) -> None:
    """
    Train a logistic regression on the gold-labelled subset and persist to disk.
    The model learns to adjust the rule-based features against observed labels.
    """
    households = pd.read_csv(households_path)
    gold = pd.read_csv(gold_path)

    merged = households.merge(gold, on="household_id")

    feat_rows = merged.apply(lambda r: _features(r), axis=1)
    X = pd.DataFrame(list(feat_rows))
    y = merged["stunting_flag"].values

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=42)),
    ])
    pipeline.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    # Report basic accuracy
    preds = pipeline.predict(X)
    acc = (preds == y).mean()
    prec = ((preds == 1) & (y == 1)).sum() / max((preds == 1).sum(), 1)
    rec  = ((preds == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)
    print(f"✓ Model trained  | Accuracy={acc:.3f}  Precision={prec:.3f}  Recall={rec:.3f}")
    print(f"  Feature weights: {dict(zip(X.columns, pipeline['clf'].coef_[0].round(3)))}")
    print(f"  Model saved → {MODEL_PATH}")


def _load_model() -> Pipeline | None:
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None


# ── Public API ────────────────────────────────────────────────────────────────
def score(household: Union[dict, pd.Series]) -> dict:
    """
    Score a single household.

    Parameters
    ----------
    household : dict or pd.Series
        Must contain keys: avg_meal_count, water_source, sanitation_tier,
        income_band, children_under5.

    Returns
    -------
    dict with keys:
        risk_score  : float in [0, 1]
        risk_label  : "High" | "Medium" | "Low"
        top_drivers : list of str (up to 3)
        method      : "logistic+rule" | "rule-based"
    """
    feats = _features(household)
    rule_score = _rule_based_score(feats)

    model = _load_model()
    if model is not None:
        feat_vec = pd.DataFrame([feats])
        lr_prob = float(model.predict_proba(feat_vec)[0, 1])
        # Blend: 60% logistic, 40% rule-based for stability
        final_score = 0.6 * lr_prob + 0.4 * rule_score
        method = "logistic+rule"
    else:
        final_score = rule_score
        method = "rule-based"

    return {
        "risk_score": round(final_score, 4),
        "risk_label": _risk_label(final_score),
        "top_drivers": _top_drivers(feats),
        "method": method,
    }


def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score all rows in a households DataFrame.
    Adds columns: risk_score, risk_label, top_drivers.
    """
    results = df.apply(score, axis=1)
    df = df.copy()
    df["risk_score"] = results.apply(lambda r: r["risk_score"])
    df["risk_label"] = results.apply(lambda r: r["risk_label"])
    df["top_drivers"] = results.apply(lambda r: r["top_drivers"])
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("=== Stunting Risk Scorer ===\n")

    # Train model
    print("Training logistic regression on gold labels...")
    train()

    # Score all households
    print("\nScoring all households...")
    hh = pd.read_csv("data/households.csv")
    scored = score_dataframe(hh)
    scored.to_csv("data/scored_households.csv", index=False)

    # Summary
    print("\nRisk distribution:")
    print(scored["risk_label"].value_counts())
    print(f"\nMean risk score : {scored['risk_score'].mean():.3f}")
    print(f"High-risk count : {(scored['risk_label']=='High').sum()}")
    print("\n✓ Scored data saved → data/scored_households.csv")

    # Demo: single household
    print("\n--- Single household demo ---")
    demo = {
        "avg_meal_count": 1.5,
        "water_source": "surface",
        "sanitation_tier": "low",
        "income_band": "low",
        "children_under5": 4,
    }
    result = score(demo)
    print(f"Input  : {demo}")
    print(f"Output : {json.dumps(result, indent=2)}")

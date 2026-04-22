"""
dashboard.py
============
Streamlit dashboard for S2.T1.2 — Stunting Risk Heatmap.

Run:
    streamlit run dashboard.py

Requirements: streamlit, folium, streamlit-folium, pandas, plotly
"""

import json

import folium
import pandas as pd
import plotly.express as px
import streamlit as st
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

from risk_scorer import score_dataframe, train

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rwanda Stunting Risk Dashboard",
    page_icon="🏥",
    layout="wide",
)

# ── Load / score data (cached) ────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading and scoring households…")
def load_data():
    import os
    from pathlib import Path
    if not Path("data/scored_households.csv").exists():
        train()
        hh = pd.read_csv("data/households.csv")
        scored = score_dataframe(hh)
        scored.to_csv("data/scored_households.csv", index=False)
    else:
        scored = pd.read_csv("data/scored_households.csv")
        # top_drivers stored as string — parse back to list
        import ast
        scored["top_drivers"] = scored["top_drivers"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
    return scored


@st.cache_data(show_spinner=False)
def load_geojson():
    with open("data/districts.geojson") as f:
        return json.load(f)


df = load_data()
geojson = load_geojson()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.title("🔍 Filters")

districts = ["All"] + sorted(df["district"].unique().tolist())
selected_district = st.sidebar.selectbox("District", districts)

risk_threshold = st.sidebar.slider(
    "Risk score threshold (show households ≥)",
    min_value=0.0, max_value=1.0, value=0.5, step=0.05
)

show_labels = st.sidebar.radio("Risk labels to show", ["All", "High", "Medium", "Low"])

# ── Filter dataframe ──────────────────────────────────────────────────────────
filtered = df.copy()
if selected_district != "All":
    filtered = filtered[filtered["district"] == selected_district]
filtered = filtered[filtered["risk_score"] >= risk_threshold]
if show_labels != "All":
    filtered = filtered[filtered["risk_label"] == show_labels]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏥 Rwanda Childhood Stunting Risk Dashboard")
st.caption("S2.T1.2 · AIMS KTT Hackathon 2026 · Synthetic NISR-style data")

# ── KPI row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total households (filtered)", len(filtered))
c2.metric("High Risk", int((filtered["risk_label"] == "High").sum()))
c3.metric("Mean Risk Score", f"{filtered['risk_score'].mean():.2f}" if len(filtered) else "—")
c4.metric("Districts shown", filtered["district"].nunique())

st.divider()

# ── Layout: map + bar chart ───────────────────────────────────────────────────
left, right = st.columns([3, 2])

# ── Choropleth map ────────────────────────────────────────────────────────────
with left:
    st.subheader("📍 Sector-Level Risk Choropleth")

    # Aggregate to sector
    sector_agg = (
        df.groupby(["district", "sector"])["risk_score"]
        .mean()
        .reset_index()
        .rename(columns={"risk_score": "avg_risk"})
    )

    # Build Folium map
    center_lat = -1.95 if selected_district == "All" else filtered["lat"].mean()
    center_lon = 30.10 if selected_district == "All" else filtered["lon"].mean()
    zoom = 10 if selected_district == "All" else 11

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom,
                   tiles="CartoDB positron")

    # District choropleth
    folium.Choropleth(
        geo_data=geojson,
        name="District Risk",
        data=df.groupby("district")["risk_score"].mean().reset_index(),
        columns=["district", "risk_score"],
        key_on="feature.properties.district",
        fill_color="YlOrRd",
        fill_opacity=0.65,
        line_opacity=0.4,
        legend_name="Mean Stunting Risk Score",
        nan_fill_color="white",
    ).add_to(m)

    # Household markers (filtered, capped at 500 for performance)
    display_df = filtered.head(500)
    cluster = MarkerCluster(name="Households").add_to(m)

    color_map = {"High": "red", "Medium": "orange", "Low": "green"}
    for _, row in display_df.iterrows():
        drivers = row["top_drivers"]
        if isinstance(drivers, list):
            drivers_html = "<br>".join(f"• {d}" for d in drivers)
        else:
            drivers_html = str(drivers)
        popup_html = f"""
        <b>ID:</b> {row['household_id']}<br>
        <b>District:</b> {row['district']}<br>
        <b>Sector:</b> {row['sector']}<br>
        <b>Risk Score:</b> {row['risk_score']:.2f} ({row['risk_label']})<br>
        <b>Top Drivers:</b><br>{drivers_html}
        """
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=5,
            color=color_map.get(row["risk_label"], "blue"),
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=280),
        ).add_to(cluster)

    folium.LayerControl().add_to(m)
    st_folium(m, width=700, height=480)

# ── Bar + pie charts ──────────────────────────────────────────────────────────
with right:
    st.subheader("📊 Risk by District")
    dist_risk = (
        filtered.groupby("district")["risk_score"].mean().reset_index().sort_values("risk_score", ascending=False)
    )
    fig_bar = px.bar(
        dist_risk, x="risk_score", y="district", orientation="h",
        color="risk_score", color_continuous_scale="YlOrRd",
        labels={"risk_score": "Avg Risk", "district": ""},
        height=250,
    )
    fig_bar.update_layout(margin=dict(l=0, r=10, t=10, b=10), showlegend=False,
                          coloraxis_showscale=False)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("🍩 Risk Label Distribution")
    label_counts = filtered["risk_label"].value_counts().reset_index()
    label_counts.columns = ["label", "count"]
    color_seq = ["#e74c3c", "#f39c12", "#2ecc71"]
    fig_pie = px.pie(
        label_counts, names="label", values="count",
        color="label",
        color_discrete_map={"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71"},
        height=220,
    )
    fig_pie.update_layout(margin=dict(l=0, r=0, t=10, b=10))
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("📋 Top Risk Drivers (all filtered HH)")
    driver_counts: dict[str, int] = {}
    for drivers in filtered["top_drivers"]:
        if isinstance(drivers, list):
            for d in drivers:
                driver_counts[d] = driver_counts.get(d, 0) + 1
    if driver_counts:
        dr_df = pd.DataFrame(list(driver_counts.items()), columns=["Driver", "Count"]).sort_values("Count", ascending=False)
        fig_dr = px.bar(dr_df, x="Count", y="Driver", orientation="h", height=220)
        fig_dr.update_layout(margin=dict(l=0, r=10, t=10, b=10))
        st.plotly_chart(fig_dr, use_container_width=True)

st.divider()

# ── Sector-level table ────────────────────────────────────────────────────────
st.subheader("🏘️ Sector-Level Summary")
sector_table = (
    filtered.groupby(["district", "sector"])
    .agg(
        households=("household_id", "count"),
        avg_risk=("risk_score", "mean"),
        high_risk_count=("risk_label", lambda x: (x == "High").sum()),
    )
    .reset_index()
    .sort_values("avg_risk", ascending=False)
)
sector_table["avg_risk"] = sector_table["avg_risk"].round(3)
st.dataframe(sector_table, use_container_width=True, height=220)

st.divider()

# ── Raw data table ────────────────────────────────────────────────────────────
with st.expander("📂 Raw household data (filtered)"):
    show_cols = ["household_id", "district", "sector", "risk_score",
                 "risk_label", "avg_meal_count", "water_source",
                 "sanitation_tier", "income_band", "children_under5"]
    st.dataframe(filtered[show_cols].sort_values("risk_score", ascending=False),
                 use_container_width=True)

st.caption("Data: Synthetic NISR-style generator · Model: Logistic Regression + Rule-based · Built for AIMS KTT Hackathon 2026")

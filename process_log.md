# process_log.md — AIMS KTT Hackathon 2026 · S2.T1.2

## Hour-by-hour timeline

| Time (CAT) | Activity |
|---|---|
| 10:00 – 10:20 | Read brief end-to-end. Identified key deliverables and scoring weights. Decided on rule-based + logistic hybrid over deep ML (speed + explainability). |
| 10:20 – 10:40 | Set up repo structure. Created `generate_data.py` with reproducible NISR-style synthetic data (SEED=42). Verified output: 2,500 rows, 5 districts, correct stunting probability function. |
| 10:40 – 11:20 | Built `risk_scorer.py`: feature engineering, rule-based weights, logistic regression pipeline with StandardScaler, blended scoring, top-driver extraction. Tested `score()` on edge cases. |
| 11:20 – 12:00 | Built `dashboard.py`: Streamlit layout, Folium choropleth, sector aggregation, district/threshold filters, KPI metrics, bar and pie charts. |
| 12:00 – 12:30 | Built `generate_printables.py`: ReportLab A4 layout with anonymised IDs, top-3 drivers, action guidance table, privacy footer. Generated 5 PDFs. |
| 12:30 – 13:00 | Wrote README (2-command Colab setup, data table, model card, product section with cost and language plan). Wrote process_log and SIGNED.md. |
| 13:00 – 13:30 | Recorded 4-minute video. Ran `streamlit run dashboard.py` live on screen. Walked `risk_scorer.py::score()`. Answered the three questions. |
| 13:30 – 13:45 | Final checks: incognito link verification, README links, repo public status. |

---

## LLM / tool use

| Tool | Version | Purpose |
|---|---|---|
| GitHub Copilot (Claude Sonnet 4.6) | April 2026 | Primary coding assistant — used throughout to generate boilerplate, suggest feature weights, draft ReportLab layout, and structure README. All output was reviewed, debugged, and adapted. |

---

## Three sample prompts I actually sent

**Prompt 1** (used):
> "Create a Streamlit dashboard with a Folium choropleth map aggregated at sector level. Include a district dropdown filter, a risk threshold slider, and KPI metrics at the top."

**Prompt 2** (used):
> "Write a risk_scorer.py with a score(household) function that uses a hybrid of rule-based weights and logistic regression. The function must return risk_score, risk_label, and top_drivers."

**Prompt 3** (used):
> "Generate a ReportLab A4 PDF showing the top 10 anonymised high-risk households in a sector with a colour-coded table, a summary box, and a privacy footer compliant with Rwanda data protection."

**Prompt I discarded:**
> "Train a random forest with SHAP values for feature importance on the stunting dataset."

Reason discarded: The gold set has only 300 rows (150 positive, 150 negative). A random forest risks overfitting and is harder to explain to a village chief during Live Defense. The rule-based + logistic blend is more defensible, faster on CPU, and produces identical user-facing output.

---

## Hardest decision

The single hardest decision was **feature weighting** in the rule-based component. I had to choose whether to weight `water_source` higher than `avg_meal_count`, or vice versa. Rwanda DHS data and UNICEF literature consistently rank **dietary diversity and meal frequency** as the strongest proxies for stunting — more so than water source alone (which correlates with diarrhoea/wasting, a related but distinct pathway). I therefore gave `meal_deficit` the highest weight (30%) and `unsafe_water` second (25%), even though the logistic regression component would rebalance these during calibration. This decision is defensible at Live Defense because it reflects real public-health evidence.

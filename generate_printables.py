"""
generate_printables.py
======================
Generates printable A4 PDF pages for S2.T1.2.
One PDF per sector sample (at least 5 total).
Each page shows top-10 anonymised high-risk households + top-3 drivers.

Run:
    python generate_printables.py

Output: printable/sector_<SectorName>.pdf
"""

import ast
import os
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

os.makedirs("printable", exist_ok=True)

# ── Load scored data ──────────────────────────────────────────────────────────
scored = pd.read_csv("data/scored_households.csv")
scored["top_drivers"] = scored["top_drivers"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

# ── Styles ────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

TITLE_STYLE = ParagraphStyle(
    "Title",
    parent=styles["Heading1"],
    fontSize=16,
    textColor=colors.HexColor("#1a5276"),
    spaceAfter=4,
)
SUBTITLE_STYLE = ParagraphStyle(
    "Subtitle",
    parent=styles["Normal"],
    fontSize=9,
    textColor=colors.HexColor("#555555"),
    spaceAfter=10,
)
BODY_STYLE = ParagraphStyle(
    "Body",
    parent=styles["Normal"],
    fontSize=9,
    spaceAfter=6,
    leading=13,
)
SECTION_STYLE = ParagraphStyle(
    "Section",
    parent=styles["Heading2"],
    fontSize=11,
    textColor=colors.HexColor("#1a5276"),
    spaceBefore=10,
    spaceAfter=4,
)
FOOTER_STYLE = ParagraphStyle(
    "Footer",
    parent=styles["Normal"],
    fontSize=7,
    textColor=colors.grey,
    spaceAfter=0,
)

TABLE_HEADER_COLOR = colors.HexColor("#1a5276")
HIGH_RISK_COLOR    = colors.HexColor("#fadbd8")
MED_RISK_COLOR     = colors.HexColor("#fef9e7")


def risk_color(label: str) -> colors.Color:
    return {"High": colors.HexColor("#e74c3c"),
            "Medium": colors.HexColor("#f39c12"),
            "Low": colors.HexColor("#27ae60")}.get(label, colors.black)


def anonymise(household_id: str, idx: int) -> str:
    """Replace real ID with positional reference — no names."""
    return f"HH-{idx+1:02d}"


def build_pdf(sector_name: str, district: str, sector_df: pd.DataFrame, out_path: str):
    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    elements = []

    # ── Header ────────────────────────────────────────────────────────────────
    elements.append(Paragraph("🇷🇼  MINISANTE / NISR  ·  Umudugudu Risk Report", TITLE_STYLE))
    elements.append(Paragraph(
        f"District: <b>{district}</b>  |  Sector: <b>{sector_name}</b>  |  "
        f"Report Date: April 2026  |  CONFIDENTIAL — IDs anonymised",
        SUBTITLE_STYLE,
    ))

    # ── Summary box ───────────────────────────────────────────────────────────
    total = len(sector_df)
    high_n = (sector_df["risk_label"] == "High").sum()
    med_n  = (sector_df["risk_label"] == "Medium").sum()
    avg_r  = sector_df["risk_score"].mean()

    summary_data = [
        ["Total households", f"{total}"],
        ["High-risk households", f"{high_n}  ({100*high_n/max(total,1):.0f}%)"],
        ["Medium-risk households", f"{med_n}  ({100*med_n/max(total,1):.0f}%)"],
        ["Average risk score", f"{avg_r:.2f}  /  1.00"],
    ]
    summary_table = Table(summary_data, colWidths=[8 * cm, 7 * cm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), colors.HexColor("#eaf2ff")),
        ("TEXTCOLOR",   (0, 0), (0, -1), colors.HexColor("#1a5276")),
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#eaf2ff"), colors.white]),
        ("BOX",         (0, 0), (-1, -1), 0.5, colors.HexColor("#aed6f1")),
        ("INNERGRID",   (0, 0), (-1, -1), 0.3, colors.HexColor("#aed6f1")),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 0.4 * cm))

    # ── Top 10 high-risk households table ─────────────────────────────────────
    elements.append(Paragraph("Top 10 Highest-Risk Households", SECTION_STYLE))
    elements.append(Paragraph(
        "Households are ranked by risk score. All identifiers are anonymised — "
        "no personal names appear on this page.",
        BODY_STYLE,
    ))

    top10 = sector_df.sort_values("risk_score", ascending=False).head(10).reset_index(drop=True)

    table_data = [["Ref", "Risk Score", "Risk Level", "Meals/day",
                   "Water Source", "Sanitation", "Income", "Top Drivers"]]

    for idx, row in top10.iterrows():
        drivers = row["top_drivers"]
        if isinstance(drivers, list):
            driver_str = "\n".join(f"• {d[:30]}" for d in drivers[:3])
        else:
            driver_str = str(drivers)[:60]

        risk_label = row["risk_label"]
        table_data.append([
            anonymise(row["household_id"], idx),
            f"{row['risk_score']:.2f}",
            risk_label,
            f"{row['avg_meal_count']:.1f}",
            row["water_source"].replace("_", " ").title(),
            row["sanitation_tier"].title(),
            row["income_band"].title(),
            Paragraph(driver_str, ParagraphStyle("d", fontSize=7, leading=10)),
        ])

    col_widths = [1.3*cm, 1.5*cm, 1.6*cm, 1.5*cm, 2.0*cm, 1.7*cm, 1.5*cm, 5.9*cm]
    hh_table = Table(table_data, colWidths=col_widths, repeatRows=1)

    ts = [
        ("BACKGROUND",    (0, 0), (-1, 0), TABLE_HEADER_COLOR),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("ALIGN",         (7, 1), (7, -1), "LEFT"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ("BOX",           (0, 0), (-1, -1), 0.5, colors.grey),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
    ]
    # Color-code risk level column
    for i, row in enumerate(top10.itertuples(), start=1):
        bg = HIGH_RISK_COLOR if row.risk_label == "High" else (MED_RISK_COLOR if row.risk_label == "Medium" else colors.white)
        ts.append(("BACKGROUND", (2, i), (2, i), bg))

    hh_table.setStyle(TableStyle(ts))
    elements.append(hh_table)
    elements.append(Spacer(1, 0.4 * cm))

    # ── Action guidance ───────────────────────────────────────────────────────
    elements.append(Paragraph("Recommended Actions for Community Leaders", SECTION_STYLE))

    CELL_STYLE = ParagraphStyle("cell", fontSize=8, leading=11)
    CELL_BOLD  = ParagraphStyle("cellb", fontSize=8, leading=11, fontName="Helvetica-Bold")

    action_data = [
        [
            Paragraph("Risk Level", CELL_BOLD),
            Paragraph("Immediate Action", CELL_BOLD),
            Paragraph("Who escalates to", CELL_BOLD),
        ],
        [
            Paragraph("🔴 High", CELL_STYLE),
            Paragraph("Visit household within 7 days. Verify meal count and water source. Issue nutrition kit referral if needed.", CELL_STYLE),
            Paragraph("Sector Health Post → MINISANTE", CELL_STYLE),
        ],
        [
            Paragraph("🟡 Medium", CELL_STYLE),
            Paragraph("Schedule home visit next monthly cycle. Advise on WASH practices.", CELL_STYLE),
            Paragraph("Sector coordinator", CELL_STYLE),
        ],
        [
            Paragraph("🟢 Low", CELL_STYLE),
            Paragraph("Monitor at next quarterly community meeting.", CELL_STYLE),
            Paragraph("Village chief log", CELL_STYLE),
        ],
    ]
    action_table = Table(action_data, colWidths=[2.5*cm, 9.5*cm, 5.0*cm])
    action_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), TABLE_HEADER_COLOR),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [HIGH_RISK_COLOR, MED_RISK_COLOR, colors.HexColor("#eafaf1")]),
        ("BOX",           (0, 0), (-1, -1), 0.5, colors.grey),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
    ]))
    elements.append(action_table)
    elements.append(Spacer(1, 0.3 * cm))

    # ── Privacy notice ────────────────────────────────────────────────────────
    elements.append(Paragraph(
        "PRIVACY: This document contains no personal names. Household IDs are positional references "
        "valid only within this report. The matching register is held by the district health office "
        "under Rwanda Data Protection Law.",
        FOOTER_STYLE,
    ))
    elements.append(Paragraph(
        "Generated by: AIMS KTT S2.T1.2 Stunting Risk Dashboard · Synthetic data for demonstration",
        FOOTER_STYLE,
    ))

    doc.build(elements)
    print(f"  ✓ {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating printable A4 sector reports…\n")

    # Pick one sector per district (the sector with highest avg risk)
    sector_ranking = (
        scored.groupby(["district", "sector"])["risk_score"]
        .mean()
        .reset_index()
        .sort_values("risk_score", ascending=False)
    )

    seen_districts: set[str] = set()
    selected: list[dict] = []
    for _, row in sector_ranking.iterrows():
        if row["district"] not in seen_districts:
            selected.append({"district": row["district"], "sector": row["sector"]})
            seen_districts.add(row["district"])
        if len(selected) >= 5:
            break

    for item in selected:
        district = item["district"]
        sector   = item["sector"]
        sector_df = scored[(scored["district"] == district) & (scored["sector"] == sector)]
        safe_name = sector.replace(" ", "_")
        out_path  = f"printable/sector_{safe_name}.pdf"
        build_pdf(sector, district, sector_df, out_path)

    print(f"\n✓ {len(selected)} PDFs written to printable/")

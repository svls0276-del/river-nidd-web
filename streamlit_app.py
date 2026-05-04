from __future__ import annotations

import json
from pathlib import Path

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from branca.colormap import LinearColormap
from pyproj import Transformer
from streamlit_folium import st_folium


ROOT = Path(__file__).resolve().parent
DATA_JS = ROOT / "web" / "data.js"
st.set_page_config(page_title="River Nidd Streamlit Backup", layout="wide")

SITE_PALETTE = {
    "Pateley Bridge": "#2d9cdb",
    "Summerbridge": "#38a169",
    "Hampsthwaite": "#dd8a2d",
    "Scotton Mill": "#d85f49",
    "Knaresborough Lido": "#7a5af8",
}
CLASS_PALETTE = {
    "Excellent": "#2d9cdb",
    "Good": "#2fb35c",
    "Sufficient": "#f1c644",
    "Poor": "#d86a2b",
}
MST_PALETTE = {
    "Hubac": "#23695a",
    "Rubac": "#d67f33",
    "Mixed": "#90836d",
}
TRANSFORMER = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)


def load_payload() -> dict:
    text = DATA_JS.read_text(encoding="utf-8")
    return json.loads(text.split("=", 1)[1].rsplit(";", 1)[0].strip())


@st.cache_data
def load_frames():
    payload = load_payload()
    locations = pd.DataFrame(payload["locations"]).sort_values("y", ascending=False).reset_index(drop=True)
    lon, lat = TRANSFORMER.transform(locations["x"].to_numpy(), locations["y"].to_numpy())
    locations["lon"] = lon
    locations["lat"] = lat

    standards = pd.DataFrame(payload["analysis"]["standardsBySite"]).sort_values("siteRank").reset_index(drop=True)
    spatial = pd.DataFrame(payload["analysis"]["spatialGradient"]).sort_values("siteRank").reset_index(drop=True)
    rainfall = pd.DataFrame(payload["analysis"]["rainfallResponseBySite"]).sort_values("siteRank").reset_index(drop=True)
    river_level = pd.DataFrame(payload["analysis"]["riverLevelBySite"]).sort_values("siteRank").reset_index(drop=True)
    mst = pd.DataFrame(payload["analysis"]["mstBySite"]).sort_values("siteRank").reset_index(drop=True)
    site_summaries = pd.DataFrame(payload["siteSummaries"]).sort_values("siteRank").reset_index(drop=True)
    samples = pd.DataFrame(payload["samples"])
    if not samples.empty:
        samples["date_dt"] = pd.to_datetime(samples["date"], dayfirst=True)
    lido_case = pd.DataFrame(payload["analysis"]["lidoCaseTimeline"])
    if not lido_case.empty:
        lido_case["date_dt"] = pd.to_datetime(lido_case["date"], dayfirst=True)
    return payload, locations, standards, spatial, rainfall, river_level, mst, site_summaries, samples, lido_case


def indicator_label(indicator: str) -> str:
    return "E. coli" if indicator == "eColi" else "IE"


def audience_class_label(value: str) -> str:
    return {
        "Excellent": "lowest bacteria-threshold band",
        "Good": "within the good bacteria band",
        "Sufficient": "within the sufficient bacteria band",
        "Poor": "above the bacteria threshold",
    }.get(value, value)


def short_class_label(value: str) -> str:
    return {
        "Excellent": "Excellent",
        "Good": "Good",
        "Sufficient": "Sufficient",
        "Poor": "Above threshold",
    }.get(value, value)


def format_unit_value(value, unit: str = "", digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, (int, np.integer)) and digits == 0:
        text = f"{int(value)}"
    else:
        text = f"{float(value):.{digits}f}"
    return f"{text} {unit}".strip()


def merge_map_df(locations: pd.DataFrame, frame: pd.DataFrame, field: str) -> pd.DataFrame:
    return locations.merge(frame[["site", field]], left_on="site", right_on="site", how="left")


def scale_color(value: float | None, values: np.ndarray, diverging: bool) -> str:
    if value is None or pd.isna(value):
        return "#adb9b1"
    if diverging:
        limit = np.nanmax(np.abs(values))
        limit = 1 if not np.isfinite(limit) or limit == 0 else limit
        cmap = LinearColormap(["#7fd0b3", "#dbe86f", "#b24a36"], vmin=-limit, vmax=limit)
    else:
        vmin = np.nanmin(values)
        vmax = np.nanmax(values)
        if not np.isfinite(vmin):
            vmin = 0
        if not np.isfinite(vmax) or vmax == vmin:
            vmax = vmin + 1
        cmap = LinearColormap(["#7fd0b3", "#dbe86f", "#b24a36"], vmin=vmin, vmax=vmax)
    return cmap(value)


def continuous_colormap(values: np.ndarray, diverging: bool, caption: str) -> LinearColormap:
    if diverging:
        limit = np.nanmax(np.abs(values))
        limit = 1 if not np.isfinite(limit) or limit == 0 else limit
        cmap = LinearColormap(["#7fd0b3", "#dbe86f", "#b24a36"], vmin=-limit, vmax=limit)
    else:
        vmin = np.nanmin(values)
        vmax = np.nanmax(values)
        if not np.isfinite(vmin):
            vmin = 0
        if not np.isfinite(vmax) or vmax == vmin:
            vmax = vmin + 1
        cmap = LinearColormap(["#7fd0b3", "#dbe86f", "#b24a36"], vmin=vmin, vmax=vmax)
    cmap.caption = caption
    return cmap


def make_map(locations: pd.DataFrame, title: str) -> folium.Map:
    fmap = folium.Map(
        location=[locations["lat"].mean(), locations["lon"].mean()],
        zoom_start=11,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    folium.PolyLine(
        locations.sort_values("y", ascending=False)[["lat", "lon"]].values.tolist(),
        color="#2f6f97",
        weight=4,
        opacity=0.85,
        dash_array="10 8",
    ).add_to(fmap)
    title_html = f"""
    <div style="position: fixed; top: 10px; left: 50px; z-index: 9999;
    background: rgba(255,255,255,0.92); padding: 8px 12px; border-radius: 10px;
    border: 1px solid #d9d2c5; font-weight: 700;">{title}</div>
    """
    fmap.get_root().html.add_child(folium.Element(title_html))
    return fmap


def add_marker_number(fmap: folium.Map, lat: float, lon: float, number: int):
    folium.Marker(
        [lat, lon],
        icon=folium.DivIcon(
            html=(
                "<div style='width:18px;height:18px;line-height:18px;text-align:center;"
                "border-radius:50%;background:white;border:1px solid #333;font-size:10px;"
                "font-weight:700;color:#111;transform: translate(-6px, -26px);'>"
                f"{number}</div>"
            )
        ),
    ).add_to(fmap)


def add_categorical_map(fmap: folium.Map, merged: pd.DataFrame, field: str, palette: dict[str, str], highlight_site: str | None = None):
    for number, row in enumerate(merged.sort_values("y", ascending=False).to_dict("records"), start=1):
        color = palette.get(row[field], "#adb9b1")
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=11 if row["site"] == highlight_site else 9,
            color="white",
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.92,
            tooltip=row["site"],
            popup=f"{row['site']}<br>{field}: {row[field]}",
        ).add_to(fmap)
        if row["site"] == highlight_site:
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=15,
                color="#1c1c1c",
                weight=3,
                fill=False,
            ).add_to(fmap)
        add_marker_number(fmap, row["lat"], row["lon"], number)
    legend_rows = "".join(
        f"<div style='display:flex;align-items:center;gap:8px;margin:4px 0;'>"
        f"<span style='display:inline-block;width:12px;height:12px;border-radius:50%;background:{color};'></span>"
        f"<span>{label}</span></div>"
        for label, color in palette.items()
    )
    legend_html = (
        "<div style='position: fixed; bottom: 58px; left: 28px; z-index: 9999; "
        "background: rgba(255,255,255,0.94); border: 1px solid #cfc7ba; border-radius: 10px; "
        "padding: 10px 12px; font-size: 12px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);'>"
        "<div style='font-weight:700;margin-bottom:6px;'>Legend</div>"
        f"{legend_rows}</div>"
    )
    fmap.get_root().html.add_child(folium.Element(legend_html))


def add_continuous_map(fmap: folium.Map, merged: pd.DataFrame, field: str, diverging: bool, caption: str, highlight_site: str | None = None):
    values = pd.to_numeric(merged[field], errors="coerce").to_numpy(dtype=float)
    cmap = continuous_colormap(values, diverging=diverging, caption=caption)
    for number, row in enumerate(merged.sort_values("y", ascending=False).to_dict("records"), start=1):
        value = row[field]
        color = "#adb9b1" if pd.isna(value) else cmap(value)
        unit = "cfu / 100 ml" if ("EColi" in field or "IE" in field or "eColi" in field or field.endswith("ie")) else "mm"
        shown = "n/a" if pd.isna(value) else format_unit_value(value, unit)
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=11 if row["site"] == highlight_site else 9,
            color="white",
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.92,
            tooltip=row["site"],
            popup=f"{row['site']}<br>{field}: {shown}",
        ).add_to(fmap)
        if row["site"] == highlight_site:
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=15,
                color="#1c1c1c",
                weight=3,
                fill=False,
            ).add_to(fmap)
        add_marker_number(fmap, row["lat"], row["lon"], number)
    cmap.add_to(fmap)


def hbar(frame: pd.DataFrame, label_col: str, value_col: str, title: str, color: str, unit_label: str, note: str):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ordered = frame.sort_values(value_col)
    ax.barh(ordered[label_col], ordered[value_col], color=color)
    ax.set_title(title)
    ax.axvline(0, color="#222222", linewidth=1)
    ax.set_xlabel(unit_label)
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(note)


def standards_scatter(samples: pd.DataFrame, site_order: list[str]):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, value_col, title, good_line, sufficient_line in [
        (axes[0], "eColi", "E. coli by site", np.log10(1000), np.log10(900)),
        (axes[1], "ie", "IE by site", np.log10(400), np.log10(330)),
    ]:
        for idx, site in enumerate(site_order):
            g = samples.loc[samples["site"] == site, value_col].dropna()
            if g.empty:
                continue
            x = np.full(len(g), idx) + np.linspace(-0.18, 0.18, len(g))
            ax.scatter(x, np.log10(g), s=32, alpha=0.8, color=SITE_PALETTE[site])
            ax.scatter(idx - 0.1, np.log10(g.quantile(0.90)), s=90, facecolors="white", edgecolors="black", linewidths=1.5, zorder=4)
            ax.scatter(idx + 0.1, np.log10(g.quantile(0.95)), s=90, color="black", marker="s", zorder=4)
        ax.axhline(good_line, color="#2f6f97", linestyle="--", linewidth=2)
        ax.axhline(sufficient_line, color="#d67f33", linestyle="--", linewidth=2)
        ax.set_title(title)
        ax.set_ylabel("log10 concentration")
        ax.grid(alpha=0.25, linestyle="--")
    axes[1].set_xticks(range(len(site_order)))
    axes[1].set_xticklabels(site_order, rotation=20)
    legend_handles = [
        plt.Line2D([0], [0], color="#2f6f97", linestyle="--", linewidth=2, label="Good threshold"),
        plt.Line2D([0], [0], color="#d67f33", linestyle="--", linewidth=2, label="Sufficient threshold"),
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white", markeredgecolor="black", markersize=7, label="90th percentile"),
        plt.Line2D([0], [0], marker="s", linestyle="", color="black", markersize=7, label="95th percentile"),
    ]
    axes[0].legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def line_and_bar_river(spatial: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.ravel()
    axes[0].plot(spatial["site"], spatial["meanEColi"], marker="o", linewidth=2.4, color="#23695a")
    axes[0].set_title("Mean E. coli from upstream to downstream")
    axes[1].plot(spatial["site"], spatial["meanIE"], marker="o", linewidth=2.4, color="#2f6f97")
    axes[1].set_title("Mean IE from upstream to downstream")
    axes[2].bar(spatial["site"], spatial["deltaFromUpstreamEColi"], color="#d85f49")
    axes[2].axhline(0, color="#222222", linewidth=1)
    axes[2].set_title("E. coli relative to upstream baseline")
    axes[3].bar(spatial["site"], spatial["deltaFromUpstreamIE"], color="#7a5af8")
    axes[3].axhline(0, color="#222222", linewidth=1)
    axes[3].set_title("IE relative to upstream baseline")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.25, linestyle="--", axis="y")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def case_charts(lido_case: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    peak_idx = lido_case[["eColi", "ie"]].max(axis=1).idxmax()
    peak_date = lido_case.loc[peak_idx, "date_dt"]
    axes[0].plot(lido_case["date_dt"], lido_case["eColi"], color="#d67f33", marker="o", linewidth=2.2, label="E. coli daily mean")
    axes[0].plot(lido_case["date_dt"], lido_case["ie"], color="#2f6f97", marker="o", linewidth=2.2, label="IE daily mean")
    episode = lido_case.loc[lido_case["date"] == "16/09/2025"]
    if not episode.empty:
        axes[0].scatter(episode["date_dt"], episode["eColi"], s=110, marker="s", color="#d67f33", edgecolor="white", zorder=5)
        axes[0].scatter(episode["date_dt"], episode["ie"], s=110, marker="s", color="#2f6f97", edgecolor="white", zorder=5)
    axes[0].axvline(peak_date, color="#c62828", linestyle="--", linewidth=1.8, label="Peak day")
    axes[0].set_title("Knaresborough Lido daily mean bacteria, Aug-Sep 2025")
    axes[0].set_ylabel("Concentration (cfu / 100 ml)")
    axes[0].grid(alpha=0.25, linestyle="--")
    axes[0].legend(loc="upper left")

    axes[1].plot(lido_case["date_dt"], lido_case["rain72h"], color="#23695a", marker="o", linewidth=2.2)
    if not episode.empty:
        axes[1].scatter(episode["date_dt"], episode["rain72h"], s=110, marker="s", color="#23695a", edgecolor="white", zorder=5)
    axes[1].axvline(peak_date, color="#c62828", linestyle="--", linewidth=1.8)
    axes[1].set_title("3-day rainfall across the same Lido period")
    axes[1].set_ylabel("Rainfall (mm)")
    axes[1].grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def mst_bars(mst: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(mst))
    ax.barh(y - 0.18, mst["meanHubac"], height=0.34, color="#23695a", label="Hubac")
    ax.barh(y + 0.18, mst["meanRubac"], height=0.34, color="#d67f33", label="Rubac")
    ax.set_yticks(y)
    ax.set_yticklabels(mst["site"])
    ax.set_title("MST key site comparison")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render():
    payload, locations, standards, spatial, rainfall, river_level, mst, site_summaries, samples, lido_case = load_frames()
    site_order = locations["site"].tolist()
    st.title("How does rainfall affect bacteria levels and swimming safety at local bathing sites?")
    st.caption("Backup Streamlit version of the River Nidd interactive webpage")
    st.markdown("**Case study:** The River Nidd")
    st.write(payload["story"]["researchQuestion"])

    with st.sidebar:
        st.header("Controls")
        mode = st.radio(
            "Analysis mode",
            ["Bacteria thresholds", "Along the river", "Environmental drivers", "Location snapshot", "Source clues"],
        )
        indicator = st.radio("Bacteria indicator", ["E. coli", "IE"], horizontal=True)
        indicator_key = "eColi" if indicator == "E. coli" else "ie"
        driver = None
        rainfall_condition = None
        if mode == "Environmental drivers":
            driver = st.radio("Environmental driver", ["Recent rainfall", "River level"], horizontal=True)
            if driver == "Recent rainfall":
                rainfall_condition = st.radio("Rainfall condition", ["Some rain", "High rain"], horizontal=True)
        mst_date = None
        if mode == "Source clues":
            dates = [item["date"] for item in payload["mstSnapshots"]]
            mst_date = st.selectbox("MST date focus", dates)

        st.subheader("Method")
        for line in [
            "Bacteria thresholds mode uses inland bathing-water thresholds from the provided guide.",
            "Sufficient uses the 90th percentile; Good and Excellent use the 95th percentile.",
            f"Only samples on or after {payload['story']['dateCutoff']} are included.",
            "Rainfall response uses the first available 3-day rainfall field.",
        ]:
            st.write(f"- {line}")

    left, right = st.columns([1.55, 1])

    with left:
        if mode == "Bacteria thresholds":
            field = "eColiClass" if indicator_key == "eColi" else "ieClass"
            p95 = "eColiP95" if indicator_key == "eColi" else "ieP95"
            merged = merge_map_df(locations, standards.rename(columns={field: "value"}), "value")
            fmap = make_map(locations, f"{indicator} against bacteria bathing-water thresholds")
            add_categorical_map(fmap, merged.rename(columns={"value": field}), field, CLASS_PALETTE)
            st_folium(fmap, use_container_width=True, height=470)
            standards_scatter(samples, site_order)
            hbar(
                standards[["site", p95]].rename(columns={p95: "value"}),
                "site",
                "value",
                f"Key site comparison: {indicator} 95th percentile",
                "#d67f33" if indicator_key == "eColi" else "#2f6f97",
                "cfu / 100 ml",
                f"Bars show the {indicator} 95th percentile at each site in cfu / 100 ml.",
            )

        elif mode == "Along the river":
            field = "deltaFromUpstreamEColi" if indicator_key == "eColi" else "deltaFromUpstreamIE"
            merged = merge_map_df(locations, spatial, field)
            fmap = make_map(locations, f"{indicator} change from upstream baseline")
            add_continuous_map(fmap, merged, field, diverging=True, caption="Delta from upstream")
            st_folium(fmap, use_container_width=True, height=470)
            line_and_bar_river(spatial)
            hbar(
                spatial[["site", field]],
                "site",
                field,
                f"Key site comparison: {indicator} delta from Pateley Bridge",
                "#d85f49" if indicator_key == "eColi" else "#7a5af8",
                "cfu / 100 ml",
                "Bars show each site's mean bacteria value minus the Pateley Bridge baseline.",
            )

        elif mode == "Environmental drivers":
            if driver == "Recent rainfall":
                field = {
                    ("eColi", "Some rain"): "someRainDeltaEColi",
                    ("eColi", "High rain"): "highRainDeltaEColi",
                    ("ie", "Some rain"): "someRainDeltaIE",
                    ("ie", "High rain"): "highRainDeltaIE",
                }[(indicator_key, rainfall_condition)]
                title = f"{indicator} {rainfall_condition.lower()} response relative to dry"
                merged = merge_map_df(locations, rainfall, field)
                fmap = make_map(locations, title)
                add_continuous_map(fmap, merged, field, diverging=True, caption="Delta from dry baseline")
                st_folium(fmap, use_container_width=True, height=470)
                hbar(
                    rainfall[["site", field]],
                    "site",
                    field,
                    f"Key site comparison: {indicator} {rainfall_condition.lower()} minus dry",
                    "#d67f33" if indicator_key == "eColi" else "#2f6f97",
                    "cfu / 100 ml",
                    f"Bars show mean {indicator} under {rainfall_condition.lower()} conditions minus the dry baseline.",
                )
            else:
                field = "highLowDeltaEColi" if indicator_key == "eColi" else "highLowDeltaIE"
                merged = merge_map_df(locations, river_level, field)
                fmap = make_map(locations, f"{indicator} response to river level")
                add_continuous_map(fmap, merged, field, diverging=True, caption="Delta between flow groups")
                st_folium(fmap, use_container_width=True, height=470)
                hbar(
                    river_level[["site", field]],
                    "site",
                    field,
                    f"Key site comparison: {indicator} high-level minus low-level",
                    "#23695a" if indicator_key == "eColi" else "#2f6f97",
                    "cfu / 100 ml",
                    f"Bars show mean {indicator} under higher river level minus lower river level conditions.",
                )

        elif mode == "Location snapshot":
            case_df = pd.DataFrame({"site": site_order, "focus": ["Case site" if site == "Knaresborough Lido" else "Context site" for site in site_order]})
            merged = merge_map_df(locations, case_df, "focus")
            fmap = make_map(locations, "Location snapshot: Knaresborough Lido in River Nidd context")
            add_categorical_map(fmap, merged, "focus", {"Case site": "#d67f33", "Context site": "#8fa2a0"}, highlight_site="Knaresborough Lido")
            st_folium(fmap, use_container_width=True, height=470)
            case_charts(lido_case)
            compare = payload["analysis"]["lidoCaseCompare"]
            compare_frame = pd.DataFrame([
                {"label": "20 Aug E. coli", "value": compare["20/08/2025"]["eColi"]},
                {"label": "16 Sep E. coli", "value": compare["16/09/2025"]["eColi"]},
                {"label": "20 Aug IE", "value": compare["20/08/2025"]["ie"]},
                {"label": "16 Sep IE", "value": compare["16/09/2025"]["ie"]},
            ])
            hbar(
                compare_frame,
                "label",
                "value",
                "Key site comparison: Knaresborough Lido on 20 Aug vs 16 Sep",
                "#d67f33",
                "cfu / 100 ml",
                "Bars compare the measured bacteria levels at Knaresborough Lido on 20 Aug and 16 Sep.",
            )

        else:
            merged = merge_map_df(locations, mst.rename(columns={"dominantMarker": "value"}), "value")
            fmap = make_map(locations, "MST source clues")
            add_categorical_map(fmap, merged.rename(columns={"value": "dominantMarker"}), "dominantMarker", MST_PALETTE)
            st_folium(fmap, use_container_width=True, height=470)
            mst_bars(mst)

    with right:
        st.subheader("Supporting summary")
        if mode == "Bacteria thresholds":
            class_col = "eColiClass" if indicator_key == "eColi" else "ieClass"
            p95_col = "eColiP95" if indicator_key == "eColi" else "ieP95"
            class_counts = standards[class_col].value_counts()
            top_band = class_counts.idxmax()
            top_count = int(class_counts.max())
            max_row = standards.sort_values(p95_col, ascending=False).iloc[0]
            st.metric("Most common threshold band", short_class_label(top_band))
            st.caption(audience_class_label(top_band))
            st.metric("Sites in that band", top_count)
            st.metric("Highest site 95th percentile", max_row["site"])
            st.metric(
                f"Highest {indicator} 95th percentile",
                format_unit_value(max_row[p95_col], "cfu / 100 ml", 0),
            )
            standards_table = standards[["site", "eColiClass", "ieClass", "eColiP95", "ieP95"]].rename(
                columns={
                    "eColiClass": "E. coli threshold status",
                    "ieClass": "IE threshold status",
                    "eColiP95": "E. coli p95 (cfu / 100 ml)",
                    "ieP95": "IE p95 (cfu / 100 ml)",
                }
            )
            standards_table["E. coli threshold status"] = standards_table["E. coli threshold status"].map(audience_class_label)
            standards_table["IE threshold status"] = standards_table["IE threshold status"].map(audience_class_label)
            st.dataframe(standards_table, use_container_width=True, hide_index=True)

        elif mode == "Along the river":
            hotspot = spatial.loc[spatial["meanEColi"].idxmax(), "site"]
            st.metric("Most upstream baseline", site_order[0])
            st.metric("Strongest E. coli hotspot", hotspot)
            lido = spatial.loc[spatial["site"] == "Knaresborough Lido"].iloc[0]
            st.metric("Lido E. coli delta", format_unit_value(lido["deltaFromUpstreamEColi"], "cfu / 100 ml"))
            st.metric("Lido IE delta", format_unit_value(lido["deltaFromUpstreamIE"], "cfu / 100 ml"))
            spatial_table = spatial.rename(
                columns={
                    "meanEColi": "Mean E. coli (cfu / 100 ml)",
                    "meanIE": "Mean IE (cfu / 100 ml)",
                    "deltaFromUpstreamEColi": "E. coli delta (cfu / 100 ml)",
                    "deltaFromUpstreamIE": "IE delta (cfu / 100 ml)",
                }
            )
            st.dataframe(spatial_table, use_container_width=True, hide_index=True)

        elif mode == "Environmental drivers":
            if driver == "Recent rainfall":
                st.write("Response here means selected rain-condition mean minus dry-condition mean.")
                field = {
                    ("eColi", "Some rain"): "someRainDeltaEColi",
                    ("eColi", "High rain"): "highRainDeltaEColi",
                    ("ie", "Some rain"): "someRainDeltaIE",
                    ("ie", "High rain"): "highRainDeltaIE",
                }[(indicator_key, rainfall_condition)]
                strongest = rainfall.sort_values(field, ascending=False).iloc[0]
                count_key = "someRainCount" if rainfall_condition == "Some rain" else "highRainCount"
                st.metric("Strongest rain response", strongest["site"])
                st.metric(f"{rainfall_condition} samples at strongest site", int(strongest[count_key]))
                st.metric(f"{indicator} delta at strongest site", format_unit_value(strongest[field], "cfu / 100 ml"))
                rain_table = rainfall[["site", "dryCount", "someRainCount", "highRainCount", field]].rename(
                    columns={field: f"{indicator} delta (cfu / 100 ml)"}
                )
                st.dataframe(rain_table, use_container_width=True, hide_index=True)
            else:
                field = "highLowDeltaEColi" if indicator_key == "eColi" else "highLowDeltaIE"
                strongest = river_level.sort_values(field, ascending=False).iloc[0]
                st.metric("Strongest river-level response", strongest["site"])
                st.metric("High-level samples at strongest site", int(strongest["highLevelCount"]))
                st.metric(f"{indicator} delta at strongest site", format_unit_value(strongest[field], "cfu / 100 ml"))
                level_table = river_level[["site", "lowLevelCount", "highLevelCount", field]].rename(
                    columns={field: f"{indicator} delta (cfu / 100 ml)"}
                )
                st.dataframe(level_table, use_container_width=True, hide_index=True)

        elif mode == "Location snapshot":
            compare = payload["analysis"]["lidoCaseCompare"]
            ecoli_delta = compare["16/09/2025"]["eColi"] - compare["20/08/2025"]["eColi"]
            ie_delta = compare["16/09/2025"]["ie"] - compare["20/08/2025"]["ie"]
            st.caption("Short-term risk can shift fast, even at one site.")
            st.metric("E. coli change", format_unit_value(ecoli_delta, "cfu / 100 ml"))
            st.metric("IE change", format_unit_value(ie_delta, "cfu / 100 ml"))
            case_table = lido_case[["date", "eColi", "ie", "rain72h", "riverLevel"]].rename(
                columns={
                    "eColi": "E. coli (cfu / 100 ml)",
                    "ie": "IE (cfu / 100 ml)",
                    "rain72h": "3-day rainfall (mm)",
                    "riverLevel": "River level (m)",
                }
            )
            st.dataframe(case_table, use_container_width=True, hide_index=True)

        else:
            snapshot = next(item for item in payload["mstSnapshots"] if item["date"] == mst_date)
            st.metric("Selected MST date", snapshot["date"])
            st.metric("Sites in snapshot", snapshot["siteCount"])
            st.metric("Mean Hubac", format_unit_value(snapshot["meanHubac"], "marker units"))
            st.metric("Mean Rubac", format_unit_value(snapshot["meanRubac"], "marker units"))
            st.write("MST adds source clues, not proof.")
            mst_table = mst[["site", "meanHubac", "meanRubac", "dominantMarker"]].rename(
                columns={"meanHubac": "Mean Hubac (marker units)", "meanRubac": "Mean Rubac (marker units)"}
            )
            st.dataframe(mst_table, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    render()

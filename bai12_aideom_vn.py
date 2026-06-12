# bai12_aideom_vn.py

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


BRAND = "#053151"

MULTI_COLORS = {
    "S1. Truyền thống": "#053151",
    "S2. Số hóa nhanh": "#E76F51",
    "S3. AI dẫn dắt": "#2A9D8F",
    "S4. Bao trùm số": "#F4A261",
    "S5. Tối ưu cân bằng": "#8E44AD",
    "K": "#053151",
    "D": "#E76F51",
    "AI": "#2A9D8F",
    "H": "#F4A261",
}


YEARS = np.arange(2026, 2031)


MODULES = [
    {
        "Module": "M1",
        "Tên": "Dự báo kinh tế",
        "Đầu vào": "Macro 2020-2025",
        "Đầu ra": "GDP, TFP, lao động 2026-2030",
        "Kỹ thuật chính": "Cobb-Douglas + Bài 1",
    },
    {
        "Module": "M2",
        "Tên": "Đánh giá sẵn sàng số",
        "Đầu vào": "Sectors, Regions",
        "Đầu ra": "Digital Index + AI Readiness",
        "Kỹ thuật chính": "Bài 6 TOPSIS",
    },
    {
        "Module": "M3",
        "Tên": "Tối ưu phân bổ",
        "Đầu vào": "Budget, beta-matrix",
        "Đầu ra": "Phân bổ ngành - vùng - thời gian",
        "Kỹ thuật chính": "Bài 4 + Bài 8",
    },
    {
        "Module": "M4",
        "Tên": "Mô phỏng lao động",
        "Đầu vào": "AI, H plans",
        "Đầu ra": "NetJob từng ngành",
        "Kỹ thuật chính": "Bài 9",
    },
    {
        "Module": "M5",
        "Tên": "Đánh giá rủi ro",
        "Đầu vào": "Risk parameters",
        "Đầu ra": "Cyber, môi trường, phụ thuộc",
        "Kỹ thuật chính": "Bài 7 + Bài 10",
    },
    {
        "Module": "M6",
        "Tên": "Dashboard ra quyết định",
        "Đầu vào": "Outputs M1-M5",
        "Đầu ra": "Kịch bản, cảnh báo, khuyến nghị",
        "Kỹ thuật chính": "Streamlit / Plotly",
    },
]


BASE_SCENARIOS = [
    {
        "Kịch bản": "S1. Truyền thống",
        "Mô tả ngắn": "Tập trung vốn vật chất, FDI, hạ tầng truyền thống, xuất khẩu",
        "Đặc điểm phân bổ": "70% K + 10% mỗi loại D/AI/H",
        "K": 0.70,
        "D": 0.10,
        "AI": 0.10,
        "H": 0.10,
    },
    {
        "Kịch bản": "S2. Số hóa nhanh",
        "Mô tả ngắn": "Tăng chính phủ số, doanh nghiệp số, thanh toán số",
        "Đặc điểm phân bổ": "25% K + 45% D + 15% AI + 15% H",
        "K": 0.25,
        "D": 0.45,
        "AI": 0.15,
        "H": 0.15,
    },
    {
        "Kịch bản": "S3. AI dẫn dắt",
        "Mô tả ngắn": "Ưu tiên AI, dữ liệu lớn, bán dẫn, trung tâm dữ liệu",
        "Đặc điểm phân bổ": "20% K + 20% D + 45% AI + 15% H",
        "K": 0.20,
        "D": 0.20,
        "AI": 0.45,
        "H": 0.15,
    },
    {
        "Kịch bản": "S4. Bao trùm số",
        "Mô tả ngắn": "Ưu tiên vùng yếu, SME, giáo dục số, nông nghiệp số",
        "Đặc điểm phân bổ": "30% K + 20% D + 10% AI + 40% H",
        "K": 0.30,
        "D": 0.20,
        "AI": 0.10,
        "H": 0.40,
    },
]


def find_file(filename):
    paths = [
        filename,
        f"./{filename}",
        f"data/{filename}",
        f"./data/{filename}",
        f"datasets/{filename}",
        f"./datasets/{filename}",
    ]

    for path in paths:
        if os.path.exists(path):
            return path

    return None


def fallback_macro_data():
    return pd.DataFrame(
        {
            "year": [2020, 2021, 2022, 2023, 2024, 2025],
            "GDP_trillion_VND": [6345, 6480, 7200, 7550, 8075, 8550],
            "GDP_growth_pct": [2.9, 2.6, 8.0, 5.1, 7.1, 6.5],
            "digital_economy_share_GDP_pct": [12.0, 13.5, 14.8, 16.0, 17.2, 18.5],
        }
    )


def fallback_region_data():
    return pd.DataFrame(
        {
            "region_name_vi": [
                "Trung du và miền núi Bắc Bộ",
                "Đồng bằng sông Hồng",
                "Bắc Trung Bộ và Duyên hải miền Trung",
                "Tây Nguyên",
                "Đông Nam Bộ",
                "Đồng bằng sông Cửu Long",
            ],
            "digital_index_0_100": [48, 78, 55, 42, 82, 52],
            "ai_readiness_0_100": [41, 77, 49, 36, 84, 45],
            "trained_labor_pct": [24, 42, 29, 22, 45, 25],
        }
    )


def fallback_sector_data():
    return pd.DataFrame(
        {
            "sector_name_vi": [
                "Nông - Lâm - Thủy sản",
                "Công nghiệp chế biến chế tạo",
                "Xây dựng",
                "Bán buôn bán lẻ",
                "Tài chính - Ngân hàng",
                "Vận tải - Logistics",
                "Thông tin - Truyền thông",
                "Giáo dục - Y tế - Xã hội",
            ],
            "automation_risk_pct": [18, 42, 25, 38, 52, 35, 28, 22],
            "ai_readiness_0_100": [38, 62, 44, 55, 82, 58, 88, 52],
        }
    )


def load_macro_data():
    path = find_file("vietnam_macro_2020_2025.csv")

    if path is None:
        return fallback_macro_data()

    try:
        df = pd.read_csv(path)

        if "year" not in df.columns or "GDP_trillion_VND" not in df.columns:
            return fallback_macro_data()

        for col in df.columns:
            if col != "year":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df.dropna(subset=["year", "GDP_trillion_VND"]).reset_index(drop=True)

    except Exception:
        return fallback_macro_data()


def load_region_data():
    path = find_file("vietnam_regions_2024.csv")

    if path is None:
        return fallback_region_data()

    try:
        df = pd.read_csv(path)

        need = ["digital_index_0_100", "ai_readiness_0_100"]

        if any(col not in df.columns for col in need):
            return fallback_region_data()

        for col in need + ["trained_labor_pct"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df.dropna(subset=need).reset_index(drop=True)

    except Exception:
        return fallback_region_data()


def load_sector_data():
    path = find_file("vietnam_sectors_2024.csv")

    if path is None:
        return fallback_sector_data()

    try:
        df = pd.read_csv(path)

        if "automation_risk_pct" not in df.columns:
            return fallback_sector_data()

        df["automation_risk_pct"] = pd.to_numeric(df["automation_risk_pct"], errors="coerce")

        return df.dropna(subset=["automation_risk_pct"]).reset_index(drop=True)

    except Exception:
        return fallback_sector_data()


def get_base_values(macro, regions, sectors):
    macro = macro.sort_values("year").reset_index(drop=True)
    last = macro.iloc[-1]

    gdp0 = float(last.get("GDP_trillion_VND", 8550))

    if "digital_economy_share_GDP_pct" in macro.columns:
        digital_share0 = float(last.get("digital_economy_share_GDP_pct", 18.5))
    else:
        digital_share0 = 18.5

    digital_index0 = float(regions["digital_index_0_100"].mean())
    ai_ready0 = float(regions["ai_readiness_0_100"].mean())

    if "trained_labor_pct" in regions.columns:
        trained0 = float(regions["trained_labor_pct"].mean())
    else:
        trained0 = 32.0

    risk0 = float(sectors["automation_risk_pct"].mean())

    return {
        "GDP": gdp0,
        "Digital Index": digital_index0,
        "AI Readiness": ai_ready0,
        "Training": trained0,
        "Digital share": digital_share0,
        "Risk": risk0,
    }


def simulate_one_scenario(scenario, base):
    rows = []

    gdp = base["GDP"]
    digital_index = base["Digital Index"]
    ai_ready = base["AI Readiness"]
    training = base["Training"]
    digital_share = base["Digital share"]
    risk_base = base["Risk"]

    for year in YEARS:
        k = scenario["K"]
        d = scenario["D"]
        ai = scenario["AI"]
        h = scenario["H"]

        growth = (
            4.5
            + 1.0 * k
            + 1.4 * d
            + 1.7 * ai
            + 0.9 * h
            + 0.015 * (digital_index - 55)
            + 0.010 * (ai_ready - 55)
        )

        gdp = gdp * (1 + growth / 100)

        digital_index = min(100, digital_index + 3.6 * d + 1.3 * k + 0.9 * h)
        ai_ready = min(100, ai_ready + 4.3 * ai + 1.1 * d + 1.2 * h)
        training = min(100, training + 3.4 * h + 0.7 * d)
        digital_share = min(45, digital_share + 1.2 * d + 1.0 * ai + 0.4 * h)

        displaced = risk_base * (0.50 * ai + 0.22 * d + 0.10 * k) * 16
        new_job = 520 * ai + 270 * d + 160 * h
        net_job = new_job - displaced + training * h * 12

        cyber_risk = np.clip(35 + 18 * ai + 8 * d - 7 * h, 0, 100)
        env_risk = np.clip(35 + 14 * k + 17 * ai - 7 * d - 6 * h, 0, 100)
        labor_risk = np.clip(35 + displaced / 25 - 0.25 * training * h, 0, 100)

        risk_score = np.clip(0.35 * cyber_risk + 0.30 * env_risk + 0.35 * labor_risk, 0, 100)

        rows.append(
            {
                "Năm": year,
                "Kịch bản": scenario["Kịch bản"],
                "GDP": gdp,
                "GDP growth (%)": growth,
                "Digital Index": digital_index,
                "AI Readiness": ai_ready,
                "Lao động đào tạo (%)": training,
                "Kinh tế số/GDP (%)": digital_share,
                "NetJob": net_job,
                "Risk score": risk_score,
                "Cyber risk": cyber_risk,
                "Environmental risk": env_risk,
                "Labor risk": labor_risk,
            }
        )

    return pd.DataFrame(rows)


def normalize(x):
    arr = np.asarray(x, dtype=float)
    min_v = np.nanmin(arr)
    max_v = np.nanmax(arr)

    if np.isclose(max_v, min_v):
        return np.ones_like(arr) * 0.5

    return (arr - min_v) / (max_v - min_v)


def score_terminal(df2030):
    score = (
        0.25 * normalize(df2030["GDP"])
        + 0.20 * normalize(df2030["Digital Index"])
        + 0.20 * normalize(df2030["AI Readiness"])
        + 0.20 * normalize(df2030["NetJob"])
        + 0.15 * normalize(-df2030["Risk score"])
    )

    return score


def optimize_s5(base):
    candidates = []

    grid = np.arange(0.10, 0.501, 0.05)

    for k in grid:
        for d in grid:
            for ai in grid:
                h = 1 - k - d - ai

                if h < 0.10 or h > 0.50:
                    continue

                if k < 0.15 or d < 0.15 or ai < 0.10:
                    continue

                scenario = {
                    "Kịch bản": "Candidate",
                    "K": k,
                    "D": d,
                    "AI": ai,
                    "H": h,
                }

                sim = simulate_one_scenario(scenario, base)
                terminal = sim[sim["Năm"] == 2030].copy()

                raw = (
                    0.25 * min(1, terminal["GDP"].iloc[0] / (base["GDP"] * 1.45))
                    + 0.20 * terminal["Digital Index"].iloc[0] / 100
                    + 0.20 * terminal["AI Readiness"].iloc[0] / 100
                    + 0.20 * np.clip((terminal["NetJob"].iloc[0] + 300) / 1400, 0, 1)
                    + 0.15 * (1 - terminal["Risk score"].iloc[0] / 100)
                )

                balance_penalty = 0.04 * (
                    abs(k - 0.25) + abs(d - 0.25) + abs(ai - 0.25) + abs(h - 0.25)
                )

                candidates.append(
                    {
                        "K": k,
                        "D": d,
                        "AI": ai,
                        "H": h,
                        "Score": raw - balance_penalty,
                    }
                )

    cand = pd.DataFrame(candidates).sort_values("Score", ascending=False).reset_index(drop=True)
    best = cand.iloc[0]

    return {
        "Kịch bản": "S5. Tối ưu cân bằng",
        "Mô tả ngắn": "Kết quả mô hình AIDEOM-VN tự chạy ra",
        "Đặc điểm phân bổ": f"{best['K']:.0%} K + {best['D']:.0%} D + {best['AI']:.0%} AI + {best['H']:.0%} H",
        "K": float(best["K"]),
        "D": float(best["D"]),
        "AI": float(best["AI"]),
        "H": float(best["H"]),
    }


def build_scenarios(base):
    s5 = optimize_s5(base)
    return BASE_SCENARIOS + [s5]


def simulate_all(scenarios, base):
    frames = [simulate_one_scenario(sc, base) for sc in scenarios]
    return pd.concat(frames, ignore_index=True)


def kpi_2030(sim_df):
    df = sim_df[sim_df["Năm"] == 2030].copy().reset_index(drop=True)

    df["AIDEOM score"] = score_terminal(df)
    df["Xếp hạng"] = df["AIDEOM score"].rank(ascending=False, method="min").astype(int)

    cols = [
        "Kịch bản",
        "GDP",
        "GDP growth (%)",
        "Digital Index",
        "AI Readiness",
        "Lao động đào tạo (%)",
        "Kinh tế số/GDP (%)",
        "NetJob",
        "Risk score",
        "AIDEOM score",
        "Xếp hạng",
    ]

    return df[cols].sort_values("Xếp hạng").reset_index(drop=True)


def risk_alerts(sim_df):
    df = sim_df[sim_df["Năm"] == 2030].copy()

    rows = []

    for _, row in df.iterrows():
        alerts = []

        if row["Risk score"] >= 70:
            alerts.append("Rủi ro tổng hợp cao")

        if row["Cyber risk"] >= 70:
            alerts.append("An ninh mạng cao")

        if row["Environmental risk"] >= 70:
            alerts.append("Áp lực môi trường cao")

        if row["Labor risk"] >= 70:
            alerts.append("Rủi ro lao động cao")

        if row["NetJob"] < 0:
            alerts.append("NetJob âm")

        if len(alerts) == 0:
            level = "Xanh"
            text = "Không có cảnh báo lớn"
        elif row["Risk score"] >= 70 or row["NetJob"] < 0:
            level = "Đỏ"
            text = "; ".join(alerts)
        else:
            level = "Vàng"
            text = "; ".join(alerts)

        rows.append(
            {
                "Kịch bản": row["Kịch bản"],
                "Mức cảnh báo": level,
                "Risk score": row["Risk score"],
                "Cyber risk": row["Cyber risk"],
                "Environmental risk": row["Environmental risk"],
                "Labor risk": row["Labor risk"],
                "NetJob": row["NetJob"],
                "Cảnh báo": text,
            }
        )

    return pd.DataFrame(rows)


def recommendations(kpi_df, risk_df):
    merged = kpi_df.merge(
        risk_df[["Kịch bản", "Mức cảnh báo", "Cảnh báo"]],
        on="Kịch bản",
        how="left",
    )

    rows = []

    for _, row in merged.iterrows():
        sc = row["Kịch bản"]

        if sc.startswith("S1"):
            rec = "Không nên chọn làm trục chính; chỉ phù hợp làm kịch bản đối chứng."
        elif sc.startswith("S2"):
            rec = "Phù hợp để tăng tốc số hóa, nhưng cần tăng an ninh dữ liệu và đào tạo kỹ năng số."
        elif sc.startswith("S3"):
            rec = "Tạo năng lực AI cao, nhưng cần kiểm soát rủi ro lao động và phụ thuộc công nghệ."
        elif sc.startswith("S4"):
            rec = "Tốt cho bao trùm và an sinh, nhưng cần bổ sung động lực tăng trưởng."
        else:
            rec = "Nên chọn làm kịch bản nền vì cân bằng giữa tăng trưởng, số hóa, AI, việc làm và rủi ro."

        rows.append(
            {
                "Kịch bản": sc,
                "Xếp hạng": row["Xếp hạng"],
                "Mức cảnh báo": row["Mức cảnh báo"],
                "Khuyến nghị ngắn": rec,
            }
        )

    return pd.DataFrame(rows).sort_values("Xếp hạng").reset_index(drop=True)


def make_styled_table(df, decimals=3):
    show_df = df.copy()

    format_dict = {}

    for col in show_df.columns:
        if pd.api.types.is_numeric_dtype(show_df[col]):
            if str(col).lower() in ["năm", "xếp hạng"]:
                format_dict[col] = "{:.0f}"
            else:
                format_dict[col] = "{:." + str(decimals) + "f}"

    styler = show_df.style.format(format_dict)

    try:
        styler = styler.hide(axis="index")
    except Exception:
        try:
            styler = styler.hide_index()
        except Exception:
            pass

    styler = styler.set_properties(
        **{
            "background-color": "#ffffff",
            "color": BRAND,
            "border": f"1px solid {BRAND}",
            "padding": "10px 12px",
            "font-size": "16px",
        }
    )

    styler = styler.set_table_styles(
        [
            {
                "selector": "thead th",
                "props": [
                    ("background-color", "#ffffff"),
                    ("color", BRAND),
                    ("font-weight", "800"),
                    ("border", f"1px solid {BRAND}"),
                    ("padding", "12px 12px"),
                    ("font-size", "16px"),
                    ("text-align", "left"),
                ],
            },
            {
                "selector": "tbody td",
                "props": [
                    ("background-color", "#ffffff"),
                    ("color", BRAND),
                    ("border", f"1px solid {BRAND}"),
                ],
            },
            {
                "selector": "table",
                "props": [
                    ("border-collapse", "collapse"),
                    ("width", "100%"),
                    ("background-color", "#ffffff"),
                    ("color", BRAND),
                ],
            },
        ]
    )

    return styler


def show_table(df, decimals=3):
    st.table(make_styled_table(df, decimals=decimals))


def style_base_fig(fig, height=430):
    fig.update_layout(
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color=BRAND, size=15),
        title_font=dict(color=BRAND, size=20),
        xaxis=dict(
            showline=True,
            linecolor=BRAND,
            tickfont=dict(color=BRAND),
            title_font=dict(color=BRAND),
        ),
        yaxis=dict(
            showline=True,
            linecolor=BRAND,
            gridcolor="rgba(5,49,81,0.18)",
            tickfont=dict(color=BRAND),
            title_font=dict(color=BRAND),
        ),
        legend=dict(font=dict(color=BRAND)),
    )

    return fig


def render():
    st.title("Bài 12. AIDEOM-VN tích hợp")
    st.caption("Dashboard tinh gọn: kịch bản, KPI 2030, cảnh báo rủi ro và khuyến nghị chính sách")

    macro = load_macro_data()
    regions = load_region_data()
    sectors = load_sector_data()

    base = get_base_values(macro, regions, sectors)
    scenarios = build_scenarios(base)
    scenarios_df = pd.DataFrame(scenarios)

    sim_df = simulate_all(scenarios, base)
    kpi_df = kpi_2030(sim_df)
    risk_df = risk_alerts(sim_df)
    rec_df = recommendations(kpi_df, risk_df)

    best_scenario = kpi_df.iloc[0]["Kịch bản"]
    best_score = kpi_df.iloc[0]["AIDEOM score"]

    st.header("12.1. Yêu cầu chức năng")
    st.markdown("Mô hình **AIDEOM-VN** gồm 6 module liên kết theo cấu trúc Mục 14 của bài báo nguồn.")
    show_table(pd.DataFrame(MODULES), decimals=3)

    st.header("12.2. Năm kịch bản chính sách")
    show_table(
        scenarios_df[
            ["Kịch bản", "Mô tả ngắn", "Đặc điểm phân bổ", "K", "D", "AI", "H"]
        ],
        decimals=3,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Số module", "6")
    c2.metric("Kịch bản tốt nhất", best_scenario)
    c3.metric("AIDEOM score", f"{best_score:.3f}")

    tabs = st.tabs(
        [
            "Đường kịch bản",
            "KPI năm 2030",
            "Cảnh báo rủi ro",
            "Thảo luận chính sách",
            "Khuyến nghị chính sách",
        ]
    )

    with tabs[0]:
        st.header("Đường kịch bản 2026-2030")

        metric = st.selectbox(
            "Chọn chỉ tiêu",
            [
                "GDP",
                "GDP growth (%)",
                "Digital Index",
                "AI Readiness",
                "NetJob",
                "Risk score",
            ],
            index=0,
        )

        fig = px.line(
            sim_df,
            x="Năm",
            y=metric,
            color="Kịch bản",
            markers=True,
            title=f"Đường kịch bản theo {metric}",
            color_discrete_map=MULTI_COLORS,
        )
        fig.update_traces(line=dict(width=4), marker=dict(size=8))
        fig.update_layout(xaxis_title="Năm", yaxis_title=metric)
        style_base_fig(fig, height=480)
        st.plotly_chart(fig, use_container_width=True)

        alloc_long = scenarios_df.melt(
            id_vars="Kịch bản",
            value_vars=["K", "D", "AI", "H"],
            var_name="Hạng mục",
            value_name="Tỷ trọng",
        )

        fig_alloc = px.bar(
            alloc_long,
            x="Kịch bản",
            y="Tỷ trọng",
            color="Hạng mục",
            title="Cấu trúc phân bổ K/D/AI/H",
            color_discrete_map=MULTI_COLORS,
        )
        fig_alloc.update_traces(marker_line_color="white", marker_line_width=1)
        fig_alloc.update_layout(barmode="stack", xaxis_title="Kịch bản", yaxis_title="Tỷ trọng")
        style_base_fig(fig_alloc, height=430)
        st.plotly_chart(fig_alloc, use_container_width=True)

    with tabs[1]:
        st.header("KPI năm 2030")

        show_table(kpi_df, decimals=3)

        score_plot = kpi_df.sort_values("AIDEOM score", ascending=True)

        fig = px.bar(
            score_plot,
            x="AIDEOM score",
            y="Kịch bản",
            orientation="h",
            text=score_plot["AIDEOM score"].round(3),
            title="Xếp hạng AIDEOM score năm 2030",
        )
        fig.update_traces(marker_color=BRAND, textposition="outside", textfont=dict(color=BRAND))
        fig.update_layout(xaxis_title="AIDEOM score", yaxis_title="Kịch bản")
        style_base_fig(fig, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.header("Cảnh báo rủi ro")

        show_table(risk_df, decimals=3)

        risk_long = risk_df[
            ["Kịch bản", "Risk score", "Cyber risk", "Environmental risk", "Labor risk"]
        ].melt(
            id_vars="Kịch bản",
            var_name="Loại rủi ro",
            value_name="Điểm rủi ro",
        )

        fig = px.bar(
            risk_long,
            x="Kịch bản",
            y="Điểm rủi ro",
            color="Loại rủi ro",
            barmode="group",
            title="Rủi ro theo kịch bản năm 2030",
        )
        fig.update_traces(marker_line_color="white", marker_line_width=1)
        fig.update_layout(xaxis_title="Kịch bản", yaxis_title="Điểm rủi ro")
        style_base_fig(fig, height=470)
        st.plotly_chart(fig, use_container_width=True)

        red = risk_df[risk_df["Mức cảnh báo"] == "Đỏ"]

        if len(red) > 0:
            st.warning("Có kịch bản cần can thiệp: " + ", ".join(red["Kịch bản"].tolist()))
        else:
            st.success("Không có kịch bản nào ở mức cảnh báo Đỏ.")

    with tabs[3]:
        st.header("Thảo luận chính sách")

        st.markdown(
            f"""
            **1. Kịch bản nổi bật:**  
            Theo điểm tổng hợp, kịch bản tốt nhất là **{best_scenario}**.

            **2. Ý nghĩa của S5:**  
            S5 không cố định trước mà được mô hình tự chọn theo tiêu chí cân bằng giữa GDP, số hóa, AI, NetJob và rủi ro.

            **3. Đánh đổi chính sách:**  
            S3 thường mạnh về AI nhưng có thể làm tăng rủi ro lao động và phụ thuộc công nghệ.  
            S4 tốt cho bao trùm nhưng tốc độ tăng trưởng có thể thấp hơn.  
            S5 là phương án trung dung, phù hợp làm baseline chính sách.

            **4. Nguyên tắc sử dụng mô hình:**  
            AIDEOM-VN chỉ là công cụ hỗ trợ ra quyết định, không thay thế đánh giá chính trị - xã hội.
            """
        )

    with tabs[4]:
        st.header("Khuyến nghị chính sách")

        show_table(rec_df, decimals=3)

        st.success(f"Khuyến nghị chính: chọn **{best_scenario}** làm kịch bản nền giai đoạn 2026-2030.")

        st.markdown(
            """
            **Khuyến nghị cô đọng:**

            - Không nên chỉ đầu tư theo hướng truyền thống S1.
            - Nếu chọn S2 hoặc S3, cần tăng mạnh an ninh dữ liệu và đào tạo lại lao động.
            - Nếu ưu tiên công bằng xã hội, S4 là kịch bản hỗ trợ tốt.
            - S5 nên được dùng làm phương án nền vì cân bằng giữa tăng trưởng, AI, việc làm và rủi ro.
            - Cần theo dõi hằng năm các KPI: GDP, Digital Index, AI Readiness, NetJob và Risk score.
            """
        )

        st.download_button(
            "Tải bảng KPI 2030 CSV",
            data=kpi_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="bai12_kpi_2030.csv",
            mime="text/csv",
        )


def run():
    render()

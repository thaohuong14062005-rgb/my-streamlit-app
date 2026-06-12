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
    "GDP": "#053151",
    "Digital Index": "#E76F51",
    "AI Readiness": "#2A9D8F",
    "NetJob": "#F4A261",
    "Risk": "#BC4749",
}


# =========================================================
# BÀI 12 — AIDEOM-VN INTEGRATED DASHBOARD
# =========================================================


YEARS = np.arange(2026, 2031)

MODULES = [
    {
        "Module": "M1",
        "Tên": "Dự báo kinh tế",
        "Đầu vào": "Macro 2020-2025",
        "Đầu ra": "GDP, TFP, lao động 2026-2030",
        "Kỹ thuật chính": "Cobb-Douglas + Bài 1, có thể bổ sung VAR/LSTM",
    },
    {
        "Module": "M2",
        "Tên": "Đánh giá sẵn sàng số",
        "Đầu vào": "Sectors, Regions",
        "Đầu ra": "Bản đồ Digital Index + AI Readiness",
        "Kỹ thuật chính": "Bài 6 TOPSIS + entropy weight",
    },
    {
        "Module": "M3",
        "Tên": "Tối ưu phân bổ",
        "Đầu vào": "Budget, beta-matrix",
        "Đầu ra": "Phân bổ ngành - vùng - thời gian",
        "Kỹ thuật chính": "Bài 4 LP + Bài 8 dynamic",
    },
    {
        "Module": "M4",
        "Tên": "Mô phỏng lao động",
        "Đầu vào": "AI, H plans",
        "Đầu ra": "NetJob từng ngành",
        "Kỹ thuật chính": "Bài 9 + bổ sung Markov chain",
    },
    {
        "Module": "M5",
        "Tên": "Đánh giá rủi ro",
        "Đầu vào": "Risk parameters",
        "Đầu ra": "Cyber, environmental, dependency",
        "Kỹ thuật chính": "Bài 7 đa mục tiêu + Bài 10 SP",
    },
    {
        "Module": "M6",
        "Tên": "Dashboard ra quyết định",
        "Đầu vào": "Outputs M1-M5",
        "Đầu ra": "Trực quan kịch bản, cảnh báo, khuyến nghị",
        "Kỹ thuật chính": "Streamlit / Dash / Plotly",
    },
]

SCENARIOS = [
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
        "Mô tả ngắn": "Tăng đầu tư chính phủ số, doanh nghiệp số, thanh toán số",
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
    {
        "Kịch bản": "S5. Tối ưu cân bằng",
        "Mô tả ngắn": "Kết quả mô hình AIDEOM-VN tự chạy ra",
        "Đặc điểm phân bổ": "Tối ưu cân bằng K/D/AI/H",
        "K": 0.35,
        "D": 0.25,
        "AI": 0.22,
        "H": 0.18,
    },
]


def find_file(filename):
    possible_paths = [
        filename,
        f"./{filename}",
        f"data/{filename}",
        f"./data/{filename}",
        f"datasets/{filename}",
        f"./datasets/{filename}",
    ]

    for path in possible_paths:
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
            "labor_productivity_million_VND": [150, 158, 170, 178, 188, 198],
            "inflation_CPI_pct": [3.2, 1.8, 3.2, 3.3, 3.8, 3.5],
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
            "gini_coef": [0.39, 0.36, 0.37, 0.40, 0.38, 0.35],
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
            "growth_rate_2024_pct": [3.3, 9.6, 7.2, 8.1, 7.8, 6.9, 10.5, 5.8],
            "ai_readiness_0_100": [38, 62, 44, 55, 82, 58, 88, 52],
            "automation_risk_pct": [18, 42, 25, 38, 52, 35, 28, 22],
        }
    )


def load_macro_data():
    path = find_file("vietnam_macro_2020_2025.csv")

    if path is None:
        return fallback_macro_data()

    try:
        df = pd.read_csv(path)

        needed = ["year", "GDP_trillion_VND", "GDP_growth_pct"]

        for col in needed:
            if col not in df.columns:
                return fallback_macro_data()

        for col in df.columns:
            if col != "year":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df.dropna(subset=needed).reset_index(drop=True)

    except Exception:
        return fallback_macro_data()


def load_region_data():
    path = find_file("vietnam_regions_2024.csv")

    if path is None:
        return fallback_region_data()

    try:
        df = pd.read_csv(path)

        needed = ["region_name_vi", "digital_index_0_100", "ai_readiness_0_100"]

        for col in needed:
            if col not in df.columns:
                return fallback_region_data()

        for col in df.columns:
            if col != "region_name_vi" and df[col].dtype != "object":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df.dropna(subset=needed).reset_index(drop=True)

    except Exception:
        return fallback_region_data()


def load_sector_data():
    path = find_file("vietnam_sectors_2024.csv")

    if path is None:
        return fallback_sector_data()

    try:
        df = pd.read_csv(path)

        needed = ["sector_name_vi", "ai_readiness_0_100"]

        for col in needed:
            if col not in df.columns:
                return fallback_sector_data()

        for col in df.columns:
            if col != "sector_name_vi" and df[col].dtype != "object":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df.dropna(subset=needed).reset_index(drop=True)

    except Exception:
        return fallback_sector_data()


def scenario_df():
    return pd.DataFrame(SCENARIOS)


def modules_df():
    return pd.DataFrame(MODULES)


def base_values_from_data(macro, regions, sectors):
    last = macro.sort_values("year").iloc[-1]

    gdp0 = float(last.get("GDP_trillion_VND", 8550))
    growth0 = float(last.get("GDP_growth_pct", 6.5))

    if "digital_economy_share_GDP_pct" in macro.columns:
        digital_share0 = float(last.get("digital_economy_share_GDP_pct", 18.5))
    else:
        digital_share0 = 18.5

    if "labor_productivity_million_VND" in macro.columns:
        productivity0 = float(last.get("labor_productivity_million_VND", 198))
    else:
        productivity0 = 198.0

    digital_index0 = float(regions["digital_index_0_100"].mean())
    ai_ready0 = float(regions["ai_readiness_0_100"].mean())

    if "trained_labor_pct" in regions.columns:
        trained0 = float(regions["trained_labor_pct"].mean())
    else:
        trained0 = 32.0

    if "automation_risk_pct" in sectors.columns:
        automation_risk0 = float(sectors["automation_risk_pct"].mean())
    else:
        automation_risk0 = 32.5

    return {
        "gdp0": gdp0,
        "growth0": growth0,
        "digital_share0": digital_share0,
        "productivity0": productivity0,
        "digital_index0": digital_index0,
        "ai_ready0": ai_ready0,
        "trained0": trained0,
        "automation_risk0": automation_risk0,
    }


def simulate_scenarios(macro, regions, sectors, annual_budget=1000):
    base = base_values_from_data(macro, regions, sectors)

    rows = []

    for sc in SCENARIOS:
        gdp = base["gdp0"]
        digital_index = base["digital_index0"]
        ai_ready = base["ai_ready0"]
        trained = base["trained0"]
        risk = base["automation_risk0"]
        digital_share = base["digital_share0"]

        for idx, year in enumerate(YEARS):
            k = sc["K"]
            d = sc["D"]
            ai = sc["AI"]
            h = sc["H"]

            growth = (
                4.6
                + 1.00 * k
                + 1.45 * d
                + 1.70 * ai
                + 0.95 * h
                + 0.018 * (digital_index - 50)
                + 0.012 * (ai_ready - 50)
            )

            inclusion_bonus = 1.8 * h + 0.8 * d
            environment_pressure = 14.0 * k + 18.0 * ai - 10.0 * d - 8.0 * h
            cyber_pressure = 20.0 * ai + 10.0 * d - 6.0 * h
            dependency_pressure = 16.0 * ai + 8.0 * d - 4.0 * h

            displaced = risk * (0.50 * ai + 0.20 * d) * 100
            upgrade = trained * h * 18
            new_job = 550 * ai + 260 * d + 180 * h
            net_job = new_job + upgrade - displaced

            risk_score = np.clip(
                35
                + 0.35 * environment_pressure
                + 0.32 * cyber_pressure
                + 0.22 * dependency_pressure
                - 0.20 * inclusion_bonus,
                0,
                100,
            )

            gdp = gdp * (1 + growth / 100)
            digital_index = min(100, digital_index + 3.8 * d + 1.4 * k + 1.1 * h)
            ai_ready = min(100, ai_ready + 4.2 * ai + 1.4 * d + 1.7 * h)
            trained = min(100, trained + 3.6 * h + 0.8 * d)
            digital_share = min(45, digital_share + 1.3 * d + 1.0 * ai + 0.4 * h)

            rows.append(
                {
                    "Năm": int(year),
                    "Kịch bản": sc["Kịch bản"],
                    "GDP": gdp,
                    "GDP growth (%)": growth,
                    "Digital Index": digital_index,
                    "AI Readiness": ai_ready,
                    "Lao động đào tạo (%)": trained,
                    "Kinh tế số/GDP (%)": digital_share,
                    "NetJob": net_job,
                    "Risk score": risk_score,
                    "Cyber risk": np.clip(35 + cyber_pressure, 0, 100),
                    "Environmental risk": np.clip(35 + environment_pressure, 0, 100),
                    "Dependency risk": np.clip(35 + dependency_pressure, 0, 100),
                    "K": k * annual_budget,
                    "D": d * annual_budget,
                    "AI": ai * annual_budget,
                    "H": h * annual_budget,
                    "Tổng ngân sách": annual_budget,
                }
            )

    return pd.DataFrame(rows)


def kpi_2030_table(sim_df):
    df = sim_df[sim_df["Năm"] == 2030].copy()

    df["Xếp hạng GDP"] = df["GDP"].rank(ascending=False, method="min").astype(int)
    df["Xếp hạng AI"] = df["AI Readiness"].rank(ascending=False, method="min").astype(int)
    df["Xếp hạng Risk"] = df["Risk score"].rank(ascending=True, method="min").astype(int)

    score = (
        0.30 * normalize_series(df["GDP"])
        + 0.20 * normalize_series(df["Digital Index"])
        + 0.20 * normalize_series(df["AI Readiness"])
        + 0.15 * normalize_series(df["NetJob"])
        + 0.15 * normalize_series(-df["Risk score"])
    )

    df["AIDEOM score"] = score
    df["Xếp hạng tổng hợp"] = df["AIDEOM score"].rank(ascending=False, method="min").astype(int)

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
        "Xếp hạng tổng hợp",
    ]

    return df[cols].sort_values("Xếp hạng tổng hợp").reset_index(drop=True)


def normalize_series(x):
    arr = np.asarray(x, dtype=float)
    min_v = np.nanmin(arr)
    max_v = np.nanmax(arr)

    if np.isclose(max_v, min_v):
        return np.ones_like(arr) * 0.5

    return (arr - min_v) / (max_v - min_v)


def risk_alert_table(sim_df):
    df = sim_df[sim_df["Năm"] == 2030].copy()

    rows = []

    for _, row in df.iterrows():
        alerts = []

        if row["Risk score"] >= 70:
            alerts.append("Rủi ro tổng hợp cao")

        if row["Cyber risk"] >= 70:
            alerts.append("Rủi ro an ninh mạng cao")

        if row["Environmental risk"] >= 70:
            alerts.append("Áp lực môi trường cao")

        if row["Dependency risk"] >= 70:
            alerts.append("Rủi ro phụ thuộc công nghệ cao")

        if row["NetJob"] < 0:
            alerts.append("NetJob âm")

        if row["AI Readiness"] >= 75 and row["Lao động đào tạo (%)"] < 40:
            alerts.append("AI nhanh hơn năng lực nhân lực")

        if len(alerts) == 0:
            level = "Thấp"
            alert_text = "Không có cảnh báo lớn"
        elif row["Risk score"] >= 70 or row["NetJob"] < 0:
            level = "Cao"
            alert_text = "; ".join(alerts)
        else:
            level = "Trung bình"
            alert_text = "; ".join(alerts)

        rows.append(
            {
                "Kịch bản": row["Kịch bản"],
                "Mức cảnh báo": level,
                "Risk score": row["Risk score"],
                "Cyber risk": row["Cyber risk"],
                "Environmental risk": row["Environmental risk"],
                "Dependency risk": row["Dependency risk"],
                "NetJob": row["NetJob"],
                "Cảnh báo": alert_text,
            }
        )

    return pd.DataFrame(rows)


def recommendation_table(kpi_df, risk_df):
    merged = kpi_df.merge(risk_df[["Kịch bản", "Mức cảnh báo", "Cảnh báo"]], on="Kịch bản", how="left")

    rows = []

    for _, row in merged.iterrows():
        scenario = row["Kịch bản"]

        if scenario.startswith("S1"):
            rec = "Không nên chọn làm chiến lược trung tâm; chỉ giữ vai trò nền tảng hạ tầng vì chuyển đổi số và AI readiness thấp."
            priority = "Giảm tỷ trọng K, tăng D và H"
        elif scenario.startswith("S2"):
            rec = "Phù hợp để tăng tốc chính phủ số và kinh tế số; cần bổ sung an ninh dữ liệu và đào tạo kỹ năng số."
            priority = "Tăng cyber governance và kỹ năng số"
        elif scenario.startswith("S3"):
            rec = "Tạo tăng trưởng và AI readiness cao nhưng cần kiểm soát rủi ro tự động hóa, phụ thuộc công nghệ và an ninh mạng."
            priority = "AI đi kèm H, an ninh mạng và chuẩn dữ liệu"
        elif scenario.startswith("S4"):
            rec = "Phù hợp mục tiêu bao trùm và an sinh; tốc độ tăng trưởng có thể thấp hơn nhưng giúp giảm phân hóa vùng và lao động."
            priority = "Ưu tiên vùng yếu, SME, giáo dục số"
        else:
            rec = "Khuyến nghị chọn làm kịch bản chính vì cân bằng giữa tăng trưởng, số hóa, AI, lao động và rủi ro."
            priority = "Triển khai làm baseline chính sách"

        rows.append(
            {
                "Kịch bản": scenario,
                "Xếp hạng tổng hợp": row["Xếp hạng tổng hợp"],
                "Mức cảnh báo": row["Mức cảnh báo"],
                "Ưu tiên điều chỉnh": priority,
                "Khuyến nghị": rec,
            }
        )

    return pd.DataFrame(rows).sort_values("Xếp hạng tổng hợp").reset_index(drop=True)


def make_styled_table(df, decimals=3):
    show_df = df.copy()

    format_dict = {}

    for col in show_df.columns:
        if pd.api.types.is_numeric_dtype(show_df[col]):
            if str(col).lower() in ["năm", "xếp hạng tổng hợp", "xếp hạng gdp", "xếp hạng ai", "xếp hạng risk"]:
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
    st.caption("Dashboard ra quyết định chính sách: mô-đun, kịch bản, KPI 2030, cảnh báo rủi ro và khuyến nghị")

    macro = load_macro_data()
    regions = load_region_data()
    sectors = load_sector_data()

    sim_df = simulate_scenarios(macro, regions, sectors, annual_budget=1000)
    kpi_df = kpi_2030_table(sim_df)
    risk_df = risk_alert_table(sim_df)
    rec_df = recommendation_table(kpi_df, risk_df)

    best_scenario = kpi_df.iloc[0]["Kịch bản"]
    best_score = kpi_df.iloc[0]["AIDEOM score"]

    tabs = st.tabs(
        [
            "Tổng quan ",
            "Đường kịch bản",
            "KPI năm 2030",
            "Cảnh báo rủi ro",
            "Thảo luận chính sách",
            "Khuyến nghị chính sách",
        ]
    )

    # =====================================================
    # 12.1 + 12.2
    # =====================================================
    with tabs[0]:
   

        st.markdown(
            """
            Mô hình **AIDEOM-VN** gồm 6 module liên kết theo cấu trúc Mục 14 của bài báo nguồn.
            Mục tiêu là tích hợp các kết quả từ Bài 1 đến Bài 11 thành một dashboard ra quyết định chính sách.
            """
        )

        show_table(modules_df(), decimals=3)

        st.header("12.2. Năm kịch bản chính sách")

        scenario_display = scenario_df()[
            ["Kịch bản", "Mô tả ngắn", "Đặc điểm phân bổ", "K", "D", "AI", "H"]
        ]
        show_table(scenario_display, decimals=3)

        c1, c2, c3 = st.columns(3)
        c1.metric("Số module", "6")
        c2.metric("Số kịch bản", "5")
        c3.metric("Kịch bản tốt nhất", best_scenario)

        alloc_long = scenario_df().melt(
            id_vars=["Kịch bản"],
            value_vars=["K", "D", "AI", "H"],
            var_name="Hạng mục",
            value_name="Tỷ trọng",
        )

        fig_alloc = px.bar(
            alloc_long,
            x="Kịch bản",
            y="Tỷ trọng",
            color="Hạng mục",
            title="Cấu trúc phân bổ K/D/AI/H theo 5 kịch bản",
        )
        fig_alloc.update_traces(marker_line_color="white", marker_line_width=1)
        fig_alloc.update_layout(
            barmode="stack",
            xaxis_title="Kịch bản",
            yaxis_title="Tỷ trọng ngân sách",
        )
        style_base_fig(fig_alloc, height=470)
        st.plotly_chart(fig_alloc, use_container_width=True)

        st.success(
            "Dashboard tích hợp M1-M6 và mô phỏng 5 kịch bản chính sách đến năm 2030."
        )

    # =====================================================
    # Đường kịch bản
    # =====================================================
    with tabs[1]:
     

        metric_choice = st.selectbox(
            "Chọn chỉ tiêu để vẽ đường kịch bản",
            [
                "GDP",
                "GDP growth (%)",
                "Digital Index",
                "AI Readiness",
                "Lao động đào tạo (%)",
                "Kinh tế số/GDP (%)",
                "NetJob",
                "Risk score",
            ],
            index=0,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Kịch bản tốt nhất", best_scenario)
        c2.metric("AIDEOM score", f"{best_score:.3f}")
        c3.metric("Năm đích", "2030")

        fig_line = px.line(
            sim_df,
            x="Năm",
            y=metric_choice,
            color="Kịch bản",
            markers=True,
            title=f"Đường kịch bản theo chỉ tiêu: {metric_choice}",
            color_discrete_map=MULTI_COLORS,
        )
        fig_line.update_traces(line=dict(width=4), marker=dict(size=8))
        fig_line.update_layout(
            xaxis_title="Năm",
            yaxis_title=metric_choice,
        )
        style_base_fig(fig_line, height=500)
        st.plotly_chart(fig_line, use_container_width=True)

        st.subheader("Bảng dữ liệu mô phỏng")
        show_table(sim_df, decimals=3)

        st.success(
            "Đường kịch bản giúp so sánh động thái của tăng trưởng, số hóa, AI readiness, việc làm và rủi ro theo thời gian."
        )

    # =====================================================
    # KPI 2030
    # =====================================================
    with tabs[2]:
       

        show_table(kpi_df, decimals=3)

        kpi_long = kpi_df[
            [
                "Kịch bản",
                "GDP",
                "Digital Index",
                "AI Readiness",
                "Lao động đào tạo (%)",
                "Kinh tế số/GDP (%)",
                "NetJob",
                "Risk score",
            ]
        ].melt(
            id_vars="Kịch bản",
            var_name="KPI",
            value_name="Giá trị",
        )

        fig_kpi = px.bar(
            kpi_long,
            x="Kịch bản",
            y="Giá trị",
            color="KPI",
            barmode="group",
            title="So sánh KPI năm 2030 theo kịch bản",
        )
        fig_kpi.update_traces(marker_line_color="white", marker_line_width=1)
        fig_kpi.update_layout(
            xaxis_title="Kịch bản",
            yaxis_title="Giá trị KPI",
        )
        style_base_fig(fig_kpi, height=540)
        st.plotly_chart(fig_kpi, use_container_width=True)

        fig_score = px.bar(
            kpi_df.sort_values("AIDEOM score", ascending=True),
            x="AIDEOM score",
            y="Kịch bản",
            orientation="h",
            text=kpi_df.sort_values("AIDEOM score", ascending=True)["AIDEOM score"].round(3),
            title="Xếp hạng tổng hợp AIDEOM score năm 2030",
        )
        fig_score.update_traces(
            marker_color=BRAND,
            textposition="outside",
            textfont=dict(color=BRAND),
        )
        fig_score.update_layout(
            xaxis_title="AIDEOM score",
            yaxis_title="Kịch bản",
        )
        style_base_fig(fig_score, height=430)
        st.plotly_chart(fig_score, use_container_width=True)

        st.success(
            f"Kịch bản có điểm tổng hợp cao nhất năm 2030 là {best_scenario}."
        )

    # =====================================================
    # Cảnh báo rủi ro
    # =====================================================
    with tabs[3]:
    

        show_table(risk_df, decimals=3)

        risk_long = risk_df[
            [
                "Kịch bản",
                "Risk score",
                "Cyber risk",
                "Environmental risk",
                "Dependency risk",
            ]
        ].melt(
            id_vars="Kịch bản",
            var_name="Loại rủi ro",
            value_name="Điểm rủi ro",
        )

        fig_risk = px.bar(
            risk_long,
            x="Kịch bản",
            y="Điểm rủi ro",
            color="Loại rủi ro",
            barmode="group",
            title="Bản đồ rủi ro theo kịch bản năm 2030",
        )
        fig_risk.update_traces(marker_line_color="white", marker_line_width=1)
        fig_risk.update_layout(
            xaxis_title="Kịch bản",
            yaxis_title="Điểm rủi ro 0-100",
        )
        style_base_fig(fig_risk, height=500)
        st.plotly_chart(fig_risk, use_container_width=True)

        high_alerts = risk_df[risk_df["Mức cảnh báo"] == "Cao"]

        if len(high_alerts) > 0:
            st.warning(
                "Có kịch bản cảnh báo cao: "
                + ", ".join(high_alerts["Kịch bản"].tolist())
            )
        else:
            st.success("Không có kịch bản nào ở mức cảnh báo cao theo ngưỡng hiện tại.")

        st.markdown(
            """
            Cảnh báo rủi ro trong AIDEOM-VN không chỉ phản ánh mức AI cao,
            mà còn xét sự mất cân đối giữa AI, nhân lực số, dữ liệu, an ninh mạng và môi trường.
            """
        )

    # =====================================================
    # Thảo luận chính sách
    # =====================================================
    with tabs[4]:
 

        st.markdown("### 1. AIDEOM-VN tích hợp các bài trước như thế nào?")

        st.markdown(
            """
            AIDEOM-VN hoạt động như một khung tích hợp. Bài 1 cung cấp nền dự báo tăng trưởng;
            Bài 4, 7, 8 cung cấp tối ưu phân bổ tĩnh - động - đa mục tiêu;
            Bài 6 đánh giá sẵn sàng vùng; Bài 9 mô phỏng lao động;
            Bài 10 đưa bất định vào quyết định; Bài 11 bổ sung tư duy học chính sách thích ứng.
            """
        )

        st.markdown("### 2. Tại sao cần so sánh nhiều kịch bản?")

        st.markdown(
            """
            Một kịch bản đơn lẻ dễ tạo cảm giác chắc chắn giả tạo.
            Trong thực tế, chuyển đổi số Việt Nam chịu ảnh hưởng của tăng trưởng, FDI, kỹ năng lao động,
            rủi ro tự động hóa, an ninh dữ liệu, thiên tai, biến động chuỗi cung ứng và thay đổi công nghệ.
            Vì vậy, dashboard cần so sánh nhiều đường kịch bản để thấy đánh đổi giữa tăng trưởng, bao trùm và rủi ro.
            """
        )

        st.markdown("### 3. Kịch bản nào nên là trục chính sách?")

        st.success(
            f"Theo AIDEOM score năm 2030, kịch bản nên ưu tiên là **{best_scenario}**."
        )

        st.markdown(
            """
            Tuy nhiên, kết quả mô hình không nên được hiểu là mệnh lệnh tự động.
            Đây là cơ sở kỹ thuật để hỗ trợ ra quyết định. Nhà hoạch định chính sách vẫn cần cân nhắc:
            năng lực ngân sách, địa - chính trị, công bằng vùng, an sinh lao động,
            an ninh dữ liệu và tính khả thi thể chế.
            """
        )

        st.markdown("### 4. AI có nên được ưu tiên tuyệt đối không?")

        st.markdown(
            """
            AI là động lực tăng năng suất, nhưng nếu AI tăng nhanh hơn dữ liệu, nhân lực và quản trị rủi ro,
            nền kinh tế có thể đối mặt với phụ thuộc công nghệ, rủi ro an ninh mạng và dịch chuyển lao động.
            Vì vậy, AI nên đi cùng ba điều kiện: dữ liệu tốt, nhân lực đủ và khung quản trị an toàn.
            """
        )

    # =====================================================
    # Khuyến nghị chính sách
    # =====================================================
    with tabs[5]:
        st.header("Khuyến nghị chính sách")

        show_table(rec_df, decimals=3)

        st.markdown("### Khuyến nghị trung tâm")

        st.success(
            f"Chọn **{best_scenario}** làm kịch bản chính sách nền cho giai đoạn 2026-2030."
        )

        st.markdown(
            """
            **Một là, triển khai kịch bản tối ưu cân bằng làm baseline.**
            Không nên cực đoan theo hướng chỉ đầu tư hạ tầng truyền thống hoặc chỉ chạy theo AI.
            Chính sách cần cân bằng giữa K, D, AI và H.

            **Hai là, đặt nhân lực số là điều kiện bắt buộc của đầu tư AI.**
            Mọi dự án AI quy mô lớn nên có cấu phần đào tạo lại, nâng kỹ năng và chuyển đổi việc làm.

            **Ba là, xây dựng dashboard cảnh báo sớm rủi ro.**
            Cần theo dõi đồng thời cyber risk, environmental risk, dependency risk và NetJob.
            Kịch bản có GDP cao nhưng Risk score quá cao không nên được chọn nguyên trạng.

            **Bốn là, ưu tiên vùng yếu và SME trong chính sách bao trùm.**
            Nếu chỉ ưu tiên vùng có năng lực hấp thụ cao, chuyển đổi số có thể làm tăng chênh lệch vùng miền.

            **Năm là, dùng mô hình như công cụ hỗ trợ, không thay thế quyết định chính trị - xã hội.**
            AIDEOM-VN nên được đặt trong quy trình có chuyên gia, địa phương, doanh nghiệp và đại diện người lao động tham gia.
            """
        )

        st.markdown("### Bộ KPI theo dõi hằng năm")

        kpi_tracking = pd.DataFrame(
            {
                "Nhóm KPI": [
                    "Tăng trưởng",
                    "Số hóa",
                    "AI",
                    "Lao động",
                    "Rủi ro",
                    "Bao trùm",
                ],
                "Chỉ tiêu đề xuất": [
                    "GDP, TFP, năng suất lao động",
                    "Digital Index, tỷ trọng kinh tế số/GDP",
                    "AI Readiness, số dự án AI công - tư",
                    "NetJob, tỷ lệ lao động được đào tạo lại",
                    "Cyber risk, dependency risk, environmental risk",
                    "Khoảng cách số vùng, SME digital adoption",
                ],
                "Tần suất": [
                    "Hằng năm",
                    "Hằng năm",
                    "6 tháng - 1 năm",
                    "6 tháng",
                    "Quý - 6 tháng",
                    "Hằng năm",
                ],
            }
        )

        show_table(kpi_tracking, decimals=3)

        st.info(
            "Kết luận: Bài 12 đóng vai trò dashboard tích hợp, chuyển kết quả định lượng từ các bài trước thành công cụ hỗ trợ ra quyết định chính sách."
        )


def run():
    render()

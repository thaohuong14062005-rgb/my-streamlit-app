# bai01_cobb_douglas.py

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


BRAND = "#053151"


# =========================================================
# BÀI 1 — HÀM SẢN XUẤT COBB-DOUGLAS MỞ RỘNG
# Trình bày gọn, đúng từng mục 1.4.1 -> 1.5
# Không dùng mô hình 3D
# =========================================================


def assignment_base_data():
    return pd.DataFrame(
        {
            "Year": [2020, 2021, 2022, 2023, 2024, 2025],
            "GDP_trillion_VND": [8044.4, 8487.5, 9513.3, 10221.8, 11511.9, 12847.6],
            "K": [16500.0, 17800.0, 19600.0, 21300.0, 23500.0, 25900.0],
            "L": [53.6, 50.5, 51.7, 52.4, 52.9, 53.4],
            "D": [12.0, 12.7, 14.3, 16.5, 18.3, 19.5],
            "AI": [55.6, 60.2, 65.4, 67.0, 73.8, 80.1],
            "H": [24.1, 26.1, 26.2, 27.0, 28.4, 29.2],
        }
    )


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


def load_data():
    """
    Ưu tiên lấy GDP và D từ vietnam_macro_2020_2025.csv.
    K, L, AI, H lấy theo bảng đề bài.
    """
    base = assignment_base_data()
    path = find_file("vietnam_macro_2020_2025.csv")

    if path is None:
        return base

    try:
        macro = pd.read_csv(path)

        if "year" in macro.columns:
            macro = macro.rename(columns={"year": "Year"})

        if "Year" not in macro.columns:
            return base

        macro["Year"] = pd.to_numeric(macro["Year"], errors="coerce")
        macro = macro.dropna(subset=["Year"]).copy()
        macro["Year"] = macro["Year"].astype(int)

        if "GDP_trillion_VND" in macro.columns:
            macro["GDP_csv"] = pd.to_numeric(macro["GDP_trillion_VND"], errors="coerce")
        else:
            macro["GDP_csv"] = np.nan

        if "digital_economy_share_GDP_pct" in macro.columns:
            macro["D_csv"] = pd.to_numeric(
                macro["digital_economy_share_GDP_pct"], errors="coerce"
            )
        else:
            macro["D_csv"] = np.nan

        macro_small = macro[["Year", "GDP_csv", "D_csv"]].copy()

        df = base.merge(macro_small, on="Year", how="left")

        df["GDP_trillion_VND"] = df["GDP_csv"].fillna(df["GDP_trillion_VND"])
        df["D"] = df["D_csv"].fillna(df["D"])

        df = df[["Year", "GDP_trillion_VND", "K", "L", "D", "AI", "H"]]
        df = df.dropna().sort_values("Year").reset_index(drop=True)

        return df

    except Exception:
        return base


def compute_model(df, alpha, beta, gamma, delta, theta):
    out = df.copy()

    Y = out["GDP_trillion_VND"].astype(float).values
    K = out["K"].astype(float).values
    L = out["L"].astype(float).values
    D = out["D"].astype(float).values
    AI = out["AI"].astype(float).values
    H = out["H"].astype(float).values

    production_part = (
        (K ** alpha)
        * (L ** beta)
        * (D ** gamma)
        * (AI ** delta)
        * (H ** theta)
    )

    A = Y / production_part
    A_mean = A.mean()
    Y_hat = A_mean * production_part
    mape = np.mean(np.abs((Y - Y_hat) / Y)) * 100

    out["A_t"] = A
    out["A_growth_pct"] = out["A_t"].pct_change() * 100
    out["Y_hat"] = Y_hat
    out["Error"] = Y - Y_hat
    out["APE_pct"] = np.abs((Y - Y_hat) / Y) * 100

    return out, A_mean, mape


def growth_decomposition(df, alpha, beta, gamma, delta, theta):
    years = df["Year"].values

    dlnY = np.diff(np.log(df["GDP_trillion_VND"].astype(float).values))
    dlnK = np.diff(np.log(df["K"].astype(float).values))
    dlnL = np.diff(np.log(df["L"].astype(float).values))
    dlnD = np.diff(np.log(df["D"].astype(float).values))
    dlnAI = np.diff(np.log(df["AI"].astype(float).values))
    dlnH = np.diff(np.log(df["H"].astype(float).values))
    dlnA = np.diff(np.log(df["A_t"].astype(float).values))

    detail = pd.DataFrame(
        {
            "Giai đoạn": [f"{years[i]}-{years[i + 1]}" for i in range(len(years) - 1)],
            "GDP (%)": dlnY * 100,
            "K (%)": alpha * dlnK * 100,
            "L (%)": beta * dlnL * 100,
            "D (%)": gamma * dlnD * 100,
            "AI (%)": delta * dlnAI * 100,
            "H (%)": theta * dlnH * 100,
            "TFP (%)": dlnA * 100,
        }
    )

    avg_growth = detail["GDP (%)"].mean()

    avg = pd.DataFrame(
        {
            "Yếu tố": ["K", "L", "D", "AI", "H", "TFP"],
            "Đóng góp bình quân (% điểm)": [
                detail["K (%)"].mean(),
                detail["L (%)"].mean(),
                detail["D (%)"].mean(),
                detail["AI (%)"].mean(),
                detail["H (%)"].mean(),
                detail["TFP (%)"].mean(),
            ],
        }
    )

    avg["Tỷ trọng tăng trưởng (%)"] = (
        avg["Đóng góp bình quân (% điểm)"] / avg_growth * 100
    )

    return detail, avg, avg_growth


def forecast_2030(
    df,
    alpha,
    beta,
    gamma,
    delta,
    theta,
    target_D,
    target_AI,
    target_H,
    gK,
    gL,
    gTFP,
):
    last = df.iloc[-1]

    years = np.arange(int(last["Year"]) + 1, 2031)
    n = len(years)

    K = np.array([last["K"] * (1 + gK) ** i for i in range(1, n + 1)])
    L = np.array([last["L"] * (1 + gL) ** i for i in range(1, n + 1)])
    A = np.array([last["A_t"] * (1 + gTFP) ** i for i in range(1, n + 1)])

    D = np.linspace(last["D"], target_D, n + 1)[1:]
    AI = np.linspace(last["AI"], target_AI, n + 1)[1:]
    H = np.linspace(last["H"], target_H, n + 1)[1:]

    Y = A * (K ** alpha) * (L ** beta) * (D ** gamma) * (AI ** delta) * (H ** theta)

    return pd.DataFrame(
        {
            "Year": years,
            "K": K,
            "L": L,
            "D": D,
            "AI": AI,
            "H": H,
            "A_t": A,
            "GDP_forecast": Y,
        }
    )


def tfp_trend(df):
    change = (df["A_t"].iloc[-1] / df["A_t"].iloc[0] - 1) * 100

    if change > 3:
        trend = "tăng rõ rệt"
    elif change > 0:
        trend = "tăng nhẹ"
    elif change < -3:
        trend = "giảm rõ rệt"
    elif change < 0:
        trend = "giảm nhẹ"
    else:
        trend = "đi ngang"

    return trend, change


def render_clean_html_table(df, decimals=4):
    show_df = df.copy()

    for col in show_df.columns:
        if pd.api.types.is_numeric_dtype(show_df[col]):
            if col.lower() in ["year", "năm"]:
                show_df[col] = show_df[col].map(lambda x: f"{int(x)}" if pd.notna(x) else "")
            else:
                show_df[col] = show_df[col].map(
                    lambda x: f"{x:.{decimals}f}" if pd.notna(x) else ""
                )

    css = f"""
    <style>
    .clean-table-wrap {{
        width: 100%;
        overflow-x: auto;
        margin-top: 8px;
        margin-bottom: 16px;
    }}
    .clean-table {{
        width: 100%;
        border-collapse: collapse;
        background: #ffffff !important;
        color: {BRAND} !important;
        border: 1.5px solid {BRAND};
        font-size: 16px;
    }}
    .clean-table thead th {{
        background: #ffffff !important;
        color: {BRAND} !important;
        font-weight: 800 !important;
        text-align: left;
        padding: 13px 15px;
        border-right: 1px solid {BRAND};
        border-bottom: 1.5px solid {BRAND};
        white-space: nowrap;
    }}
    .clean-table thead th:last-child {{
        border-right: none;
    }}
    .clean-table tbody td {{
        background: #ffffff !important;
        color: {BRAND} !important;
        padding: 12px 15px;
        border-right: 1px solid {BRAND};
        border-bottom: 1px solid {BRAND};
        vertical-align: middle;
    }}
    .clean-table tbody tr:last-child td {{
        border-bottom: none;
    }}
    .clean-table tbody td:last-child {{
        border-right: none;
    }}
    </style>
    """

    table_html = show_df.to_html(index=False, border=0, escape=False, classes="clean-table")

    st.markdown(
        css + f'<div class="clean-table-wrap">{table_html}</div>',
        unsafe_allow_html=True,
    )


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
    st.markdown(
        f"""
        <style>
        .bai1-card {{
            padding: 34px 42px;
            border-radius: 0 0 22px 22px;
            border-left: 6px solid #3b82f6;
            background: white;
            box-shadow: 0 12px 35px rgba(15, 23, 42, 0.08);
            margin-bottom: 28px;
        }}
        .bai1-title {{
            font-size: 2.1rem;
            font-weight: 850;
            color: {BRAND};
            margin-bottom: 12px;
        }}
        .bai1-sub {{
            color: #64748b;
            font-size: 1.05rem;
        }}
        div[data-testid="stMetricValue"] {{
            font-size: 1.42rem;
            color: {BRAND};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="bai1-card">
            <div class="bai1-title">Bài 1. Hàm sản xuất Cobb-Douglas mở rộng</div>
            <div class="bai1-sub">Ước lượng TFP, kiểm định dự báo, phân rã tăng trưởng và mô phỏng GDP 2030</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =========================
    # Sidebar
    # =========================
    st.sidebar.header("⚙️ Bài 1")

    st.sidebar.markdown("### Hệ số")
    alpha = st.sidebar.slider("α - K", 0.00, 0.80, 0.33, 0.01)
    beta = st.sidebar.slider("β - L", 0.00, 0.80, 0.42, 0.01)
    gamma = st.sidebar.slider("γ - D", 0.00, 0.50, 0.10, 0.01)
    delta = st.sidebar.slider("δ - AI", 0.00, 0.50, 0.08, 0.01)
    theta = st.sidebar.slider("θ - H", 0.00, 0.50, 0.07, 0.01)

    total_coef = alpha + beta + gamma + delta + theta
    st.sidebar.caption(f"Tổng hệ số: {total_coef:.2f}")

    st.sidebar.markdown("### Kịch bản 2030")
    target_D = st.sidebar.slider("D 2030 (%)", 19.5, 45.0, 30.0, 0.5)
    target_AI = st.sidebar.slider("AI 2030", 80.1, 180.0, 100.0, 1.0)
    target_H = st.sidebar.slider("H 2030 (%)", 29.2, 60.0, 35.0, 0.5)
    gK = st.sidebar.slider("Tăng K/năm (%)", 0.0, 15.0, 6.0, 0.1) / 100
    gL = st.sidebar.slider("Tăng L/năm (%)", -2.0, 10.0, 6.0, 0.1) / 100
    gTFP = st.sidebar.slider("Tăng TFP/năm (%)", -2.0, 5.0, 1.2, 0.1) / 100

    # =========================
    # Data + model
    # =========================
    raw_df = load_data()
    model_df, A_mean, mape = compute_model(raw_df, alpha, beta, gamma, delta, theta)
    detail_df, avg_df, avg_growth = growth_decomposition(
        model_df, alpha, beta, gamma, delta, theta
    )
    forecast_df = forecast_2030(
        model_df,
        alpha,
        beta,
        gamma,
        delta,
        theta,
        target_D,
        target_AI,
        target_H,
        gK,
        gL,
        gTFP,
    )

    trend, tfp_change = tfp_trend(model_df)

    new_factors = avg_df[avg_df["Yếu tố"].isin(["D", "AI", "H"])].copy()
    largest_new = (
        new_factors.assign(abs_share=lambda x: x["Tỷ trọng tăng trưởng (%)"].abs())
        .sort_values("abs_share", ascending=False)
        .iloc[0]
    )

    row_2030 = forecast_df[forecast_df["Year"] == 2030].iloc[0]
    y_2025 = model_df["GDP_trillion_VND"].iloc[-1]
    y_2030 = row_2030["GDP_forecast"]
    growth_2030 = (y_2030 / y_2025 - 1) * 100

    # =========================
    # Tabs
    # =========================
    tabs = st.tabs(
        [
            "1.4.1 TFP",
            "1.4.2 MAPE",
            "1.4.3 Phân rã",
            "1.4.4 Dự báo 2030",
            "1.5 Chính sách",
        ]
    )

    # =====================================================
    # 1.4.1
    # =====================================================
    with tabs[0]:
        st.header("1.4.1. Kết quả TFP")

        c1, c2, c3 = st.columns(3)
        c1.metric("A_t 2020", f"{model_df['A_t'].iloc[0]:,.2f}")
        c2.metric("A_t 2025", f"{model_df['A_t'].iloc[-1]:,.2f}")
        c3.metric("Xu hướng", trend)

        table_141 = model_df[
            ["Year", "GDP_trillion_VND", "K", "L", "D", "AI", "H", "A_t"]
        ].copy()

        table_141 = table_141.rename(
            columns={
                "Year": "Năm",
                "GDP_trillion_VND": "Y",
                "A_t": "TFP A_t",
            }
        )

        render_clean_html_table(table_141, decimals=3)

        fig = px.line(
            model_df,
            x="Year",
            y="A_t",
            markers=True,
            title="Xu hướng TFP A_t",
        )
        fig.update_traces(line=dict(color=BRAND, width=4), marker=dict(color=BRAND, size=9))
        fig.update_layout(xaxis_title="Năm", yaxis_title="A_t")
        style_base_fig(fig, height=430)
        st.plotly_chart(fig, use_container_width=True)

        st.success(
            f"TFP giai đoạn 2020-2025 {trend}, thay đổi khoảng {tfp_change:.2f}%."
        )

    # =====================================================
    # 1.4.2
    # =====================================================
    with tabs[1]:
        st.header("1.4.2. GDP dự báo và MAPE")

        c1, c2, c3 = st.columns(3)
        c1.metric("A trung bình", f"{A_mean:,.2f}")
        c2.metric("MAPE", f"{mape:.2f}%")
        c3.metric("Sai số TB", f"{model_df['Error'].abs().mean():,.1f}")

        table_142 = model_df[
            ["Year", "GDP_trillion_VND", "Y_hat", "Error", "APE_pct"]
        ].copy()

        table_142 = table_142.rename(
            columns={
                "Year": "Năm",
                "GDP_trillion_VND": "GDP thực tế",
                "Y_hat": "GDP dự báo",
                "Error": "Sai số",
                "APE_pct": "APE (%)",
            }
        )

        render_clean_html_table(table_142, decimals=3)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=model_df["Year"],
                y=model_df["GDP_trillion_VND"],
                mode="lines+markers",
                name="GDP thực tế",
                line=dict(color=BRAND, width=4),
                marker=dict(color=BRAND, size=9),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=model_df["Year"],
                y=model_df["Y_hat"],
                mode="lines+markers",
                name="GDP dự báo",
                line=dict(color="#7a8fa3", width=4, dash="dash"),
                marker=dict(color="#7a8fa3", size=9),
            )
        )
        fig.update_layout(
            title="GDP thực tế và GDP dự báo",
            xaxis_title="Năm",
            yaxis_title="GDP, nghìn tỷ VND",
        )
        style_base_fig(fig, height=430)
        st.plotly_chart(fig, use_container_width=True)

        fig_ape = px.bar(
            model_df,
            x="Year",
            y="APE_pct",
            text=model_df["APE_pct"].round(2),
            title="APE theo năm",
        )
        fig_ape.update_traces(
            marker_color=BRAND,
            textposition="outside",
            textfont=dict(color=BRAND, size=15),
            hovertemplate="<b>Năm %{x}</b><br>APE: %{y:.2f}%<extra></extra>",
        )
        fig_ape.update_layout(xaxis_title="Năm", yaxis_title="APE (%)")
        style_base_fig(fig_ape, height=390)
        st.plotly_chart(fig_ape, use_container_width=True)

        st.success(f"MAPE = {mape:.2f}%, cho thấy mức sai số của mô hình dự báo.")

    # =====================================================
    # 1.4.3
    # =====================================================
    with tabs[2]:
        st.header("1.4.3. Phân rã tăng trưởng")

        c1, c2, c3 = st.columns(3)
        c1.metric("GDP tăng trưởng TB", f"{avg_growth:.2f}%")
        c2.metric("Yếu tố mới nổi bật", largest_new["Yếu tố"])
        c3.metric("Tỷ trọng", f"{largest_new['Tỷ trọng tăng trưởng (%)']:.1f}%")

        st.subheader("Đóng góp bình quân")
        render_clean_html_table(avg_df, decimals=6)

        fig_avg = px.bar(
            avg_df,
            x="Yếu tố",
            y="Tỷ trọng tăng trưởng (%)",
            text=avg_df["Tỷ trọng tăng trưởng (%)"].round(1).astype(str) + "%",
            title="Tỷ trọng đóng góp tăng trưởng",
        )
        fig_avg.update_traces(
            marker_color=BRAND,
            textposition="outside",
            textfont=dict(color=BRAND, size=16),
            hovertemplate="<b>%{x}</b><br>Tỷ trọng: %{y:.2f}%<extra></extra>",
        )
        fig_avg.update_layout(
            xaxis_title="Yếu tố",
            yaxis_title="Tỷ trọng (%)",
        )
        style_base_fig(fig_avg, height=430)
        st.plotly_chart(fig_avg, use_container_width=True)

        st.subheader("Theo từng giai đoạn")
        render_clean_html_table(detail_df, decimals=4)

        detail_long = detail_df.melt(
            id_vars="Giai đoạn",
            value_vars=["K (%)", "L (%)", "D (%)", "AI (%)", "H (%)", "TFP (%)"],
            var_name="Nguồn",
            value_name="Đóng góp",
        )

        fig_detail = px.bar(
            detail_long,
            x="Giai đoạn",
            y="Đóng góp",
            color="Nguồn",
            title="Phân rã theo giai đoạn",
            color_discrete_sequence=[
                BRAND,
                "#2d5d7b",
                "#4e7d99",
                "#6b95ae",
                "#88acc2",
                "#a7c1d4",
            ],
        )
        fig_detail.update_layout(
            barmode="relative",
            xaxis_title="Giai đoạn",
            yaxis_title="Đóng góp (% điểm)",
        )
        style_base_fig(fig_detail, height=430)
        st.plotly_chart(fig_detail, use_container_width=True)

        st.success(
            f"Trong nhóm D, AI, H, yếu tố đóng góp nổi bật nhất là {largest_new['Yếu tố']}."
        )

    # =====================================================
    # 1.4.4
    # =====================================================
    with tabs[3]:
        st.header("1.4.4. Dự báo GDP 2030")

        c1, c2, c3 = st.columns(3)
        c1.metric("GDP 2025", f"{y_2025:,.1f}")
        c2.metric("GDP 2030", f"{y_2030:,.1f}")
        c3.metric("Tăng 2025-2030", f"{growth_2030:.1f}%")

        table_144 = forecast_df.rename(
            columns={
                "Year": "Năm",
                "A_t": "TFP",
                "GDP_forecast": "GDP dự báo",
            }
        )

        render_clean_html_table(table_144, decimals=3)

        actual = model_df[["Year", "GDP_trillion_VND"]].rename(
            columns={"Year": "Năm", "GDP_trillion_VND": "GDP"}
        )
        actual["Loại"] = "Thực tế"

        future = forecast_df[["Year", "GDP_forecast"]].rename(
            columns={"Year": "Năm", "GDP_forecast": "GDP"}
        )
        future["Loại"] = "Dự báo"

        chart_df = pd.concat([actual, future], ignore_index=True)

        fig = px.line(
            chart_df,
            x="Năm",
            y="GDP",
            color="Loại",
            markers=True,
            title="GDP thực tế và dự báo đến 2030",
            color_discrete_map={
                "Thực tế": BRAND,
                "Dự báo": "#7a8fa3",
            },
        )
        fig.update_traces(line=dict(width=4), marker=dict(size=9))
        fig.update_layout(yaxis_title="GDP, nghìn tỷ VND")
        style_base_fig(fig, height=440)
        st.plotly_chart(fig, use_container_width=True)

        st.success(f"GDP dự báo năm 2030 đạt khoảng {y_2030:,.1f} nghìn tỷ VND.")

    # =====================================================
    # 1.5
    # =====================================================
    with tabs[4]:
        st.header("1.5. Thảo luận chính sách")

        st.markdown("### a) TFP của Việt Nam có xu hướng tăng hay giảm?")

        st.success(
            f"TFP giai đoạn 2020-2025 có xu hướng **{trend}**, "
            f"thay đổi khoảng **{tfp_change:.2f}%**."
        )

        st.markdown(
            f"""
            Kết quả này cho thấy chất lượng tăng trưởng của Việt Nam trong giai đoạn 2020-2025
            có sự thay đổi đáng chú ý. TFP phản ánh phần tăng trưởng không được giải thích trực tiếp
            bởi các yếu tố đầu vào quan sát được như vốn vật chất, lao động, số hóa, AI và vốn nhân lực số.

            Nếu TFP tăng, điều đó hàm ý nền kinh tế đang sử dụng nguồn lực hiệu quả hơn.
            Nói cách khác, cùng một lượng vốn, lao động và công nghệ, nền kinh tế tạo ra nhiều sản lượng hơn.
            Đây là dấu hiệu tích cực vì tăng trưởng không chỉ đến từ mở rộng quy mô đầu vào,
            mà còn đến từ cải thiện công nghệ, quản trị, tổ chức sản xuất, chuyển đổi số và năng lực đổi mới sáng tạo.

            Ngược lại, nếu TFP giảm hoặc tăng rất chậm, điều này cho thấy tăng trưởng vẫn có thể phụ thuộc nhiều
            vào tích lũy vốn và mở rộng lao động. Khi đó, chất lượng tăng trưởng chưa thật sự bền vững,
            vì các nguồn lực truyền thống như vốn và lao động đều có giới hạn.
            """
        )

        st.markdown("### b) Trong các yếu tố mới D, AI, H, yếu tố nào đóng góp nhiều nhất?")

        st.success(
            f"Trong ba yếu tố mới **D, AI và H**, yếu tố đóng góp nổi bật nhất theo mô hình là "
            f"**{largest_new['Yếu tố']}**, chiếm khoảng "
            f"**{largest_new['Tỷ trọng tăng trưởng (%)']:.1f}%** trong tăng trưởng GDP bình quân."
        )

        st.markdown(
            f"""
            Trong mô hình Cobb-Douglas mở rộng, mức đóng góp của một yếu tố phụ thuộc vào hai thành phần chính:
            **tốc độ tăng của yếu tố đó** và **hệ số co giãn của yếu tố đó trong hàm sản xuất**.
            Vì vậy, một biến có tốc độ tăng nhanh nhưng hệ số nhỏ có thể đóng góp vừa phải;
            ngược lại, một biến tăng không quá nhanh nhưng có hệ số lớn vẫn có thể tạo tác động đáng kể.

            Về ý nghĩa kinh tế, **D** đại diện cho mức độ số hóa của nền kinh tế, thường được đo bằng tỷ trọng
            kinh tế số trong GDP. Khi D tăng, các hoạt động kinh tế được số hóa nhiều hơn, chi phí giao dịch giảm,
            khả năng kết nối thị trường tốt hơn và năng suất tổng thể có thể được cải thiện.

            **AI** đại diện cho năng lực trí tuệ nhân tạo hoặc số lượng doanh nghiệp công nghệ số.
            Sự gia tăng của AI cho thấy khu vực doanh nghiệp có khả năng ứng dụng công nghệ mới tốt hơn,
            từ đó hỗ trợ tự động hóa, phân tích dữ liệu, tối ưu sản xuất và nâng cao chất lượng dịch vụ.

            **H** đại diện cho vốn nhân lực số, tức tỷ lệ lao động qua đào tạo hoặc có kỹ năng phù hợp.
            Đây là điều kiện nền tảng để công nghệ phát huy hiệu quả. Nếu thiếu nhân lực số,
            đầu tư vào số hóa và AI có thể không chuyển hóa đầy đủ thành tăng trưởng thực tế.

            Do đó, dù yếu tố **{largest_new['Yếu tố']}** là yếu tố nổi bật nhất trong kết quả hiện tại,
            ba yếu tố D, AI và H không nên được xem là tách rời. Chúng có quan hệ bổ trợ lẫn nhau:
            kinh tế số cần doanh nghiệp công nghệ, doanh nghiệp công nghệ cần nhân lực số,
            và nhân lực số chỉ phát huy hiệu quả khi có môi trường số đủ phát triển.
            """
        )

        st.markdown("### c) Mục tiêu kinh tế số đạt 30% GDP vào năm 2030 có khả thi không?")

        if target_D >= 30:
            st.success(
                f"Trong kịch bản hiện tại, D năm 2030 đạt **{target_D:.1f}%**, "
                "tức đã phản ánh mục tiêu kinh tế số chiếm khoảng 30% GDP."
            )
        else:
            st.warning(
                f"Trong kịch bản hiện tại, D năm 2030 mới đạt **{target_D:.1f}%**, "
                "thấp hơn mục tiêu 30% GDP."
            )

        st.markdown(
            f"""
            Dựa trên mô hình mô phỏng, mục tiêu kinh tế số đạt khoảng **30% GDP vào năm 2030**
            là có cơ sở khả thi nếu các giả định chính được duy trì. Cụ thể, nền kinh tế cần đồng thời
            mở rộng tỷ trọng kinh tế số, gia tăng số lượng và năng lực doanh nghiệp công nghệ số,
            cải thiện chất lượng nhân lực và duy trì tăng trưởng TFP.

            Tuy nhiên, mục tiêu này không nên được hiểu đơn giản là chỉ cần tăng tỷ trọng D.
            Nếu D tăng nhưng AI và H không tăng tương ứng, tác động đến GDP có thể bị giới hạn.
            Nói cách khác, kinh tế số chỉ tạo ra tăng trưởng bền vững khi đi kèm với năng lực hấp thụ công nghệ
            của doanh nghiệp và kỹ năng số của người lao động.

            Để mục tiêu 30% kinh tế số/GDP có tính khả thi cao hơn, cần một số ràng buộc chính sách:

            - **Thứ nhất, hạ tầng số phải được đầu tư đồng bộ.**
            Điều này bao gồm băng rộng, trung tâm dữ liệu, điện toán đám mây, nền tảng số,
            định danh số, thanh toán số và an toàn thông tin.

            - **Thứ hai, doanh nghiệp phải chuyển đổi số thực chất.**
            Không chỉ tăng số lượng doanh nghiệp công nghệ, mà cần tăng mức độ ứng dụng công nghệ
            trong sản xuất, logistics, tài chính, thương mại, nông nghiệp, công nghiệp và dịch vụ công.

            - **Thứ ba, vốn nhân lực số phải được nâng cao.**
            Người lao động cần kỹ năng dữ liệu, kỹ năng sử dụng AI, kỹ năng vận hành nền tảng số
            và năng lực thích ứng với công nghệ mới.

            - **Thứ tư, TFP phải tiếp tục tăng.**
            Nếu TFP không cải thiện, nền kinh tế dễ quay lại mô hình tăng trưởng dựa vào vốn và lao động,
            trong khi hiệu quả dài hạn không cao.

            - **Thứ năm, thể chế dữ liệu và đổi mới sáng tạo cần hoàn thiện.**
            Cần có chính sách về chia sẻ dữ liệu, bảo mật, quyền riêng tư, cạnh tranh nền tảng,
            hỗ trợ doanh nghiệp nhỏ và vừa chuyển đổi số, cũng như khuyến khích nghiên cứu phát triển.

            **Kết luận:** mục tiêu kinh tế số đạt 30% GDP vào năm 2030 là có thể đạt được trong mô hình,
            nhưng chỉ khả thi nếu Việt Nam không chỉ mở rộng quy mô kinh tế số, mà còn nâng đồng thời
            **năng lực AI, vốn nhân lực số và năng suất nhân tố tổng hợp TFP**.
            """
        )

        st.markdown("### Kết luận chung")

        st.info(
            f"Mô hình Cobb-Douglas mở rộng cho thấy tăng trưởng GDP Việt Nam có thể được phân tích "
            f"thông qua các yếu tố K, L, D, AI, H và TFP. "
            f"TFP có xu hướng **{trend}**, MAPE của mô hình dự báo là **{mape:.2f}%**, "
            f"và trong nhóm yếu tố mới, yếu tố nổi bật nhất là **{largest_new['Yếu tố']}**. "
            f"Hàm ý chính sách là Việt Nam cần kết hợp chuyển đổi số, phát triển AI, nâng cao nhân lực số "
            f"và cải thiện TFP để đạt tăng trưởng bền vững."
        )


def run():
    render()

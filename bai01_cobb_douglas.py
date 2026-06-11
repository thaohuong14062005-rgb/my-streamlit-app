# bai01_cobb_douglas.py

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# BÀI 1 — COBB-DOUGLAS MỞ RỘNG
# Giao diện gọn: Kết quả TFP | Phân rã | Dự báo 2030 | Chính sách
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


def load_data():
    base = assignment_base_data()
    path = find_file("vietnam_macro_2020_2025.csv")

    if path is None:
        return base

    try:
        macro = pd.read_csv(path)

        if "year" in macro.columns:
            macro = macro.rename(columns={"year": "Year"})

        cols = ["Year", "GDP_trillion_VND", "digital_economy_share_GDP_pct"]
        macro = macro[[c for c in cols if c in macro.columns]].copy()

        macro["Year"] = pd.to_numeric(macro["Year"], errors="coerce").astype("Int64")

        if "GDP_trillion_VND" in macro.columns:
            macro["GDP_trillion_VND"] = pd.to_numeric(macro["GDP_trillion_VND"], errors="coerce")

        if "digital_economy_share_GDP_pct" in macro.columns:
            macro["D_from_csv"] = pd.to_numeric(
                macro["digital_economy_share_GDP_pct"], errors="coerce"
            )
            macro = macro.drop(columns=["digital_economy_share_GDP_pct"])

        df = base.drop(columns=["GDP_trillion_VND", "D"]).merge(macro, on="Year", how="left")
        fallback = base[["Year", "GDP_trillion_VND", "D"]].copy()
        df = df.merge(fallback, on="Year", how="left", suffixes=("", "_base"))

        df["GDP_trillion_VND"] = df["GDP_trillion_VND"].fillna(df["GDP_trillion_VND_base"])

        if "D_from_csv" in df.columns:
            df["D"] = df["D_from_csv"].fillna(df["D"])
        else:
            df["D"] = df["D"]

        df = df[["Year", "GDP_trillion_VND", "K", "L", "D", "AI", "H"]]
        df = df.dropna().sort_values("Year").reset_index(drop=True)
        df["Year"] = df["Year"].astype(int)

        return df

    except Exception:
        return base


def compute_model(df, alpha, beta, gamma, delta, theta):
    out = df.copy()

    Y = out["GDP_trillion_VND"].values
    K = out["K"].values
    L = out["L"].values
    D = out["D"].values
    AI = out["AI"].values
    H = out["H"].values

    base = (K**alpha) * (L**beta) * (D**gamma) * (AI**delta) * (H**theta)

    A = Y / base
    A_mean = A.mean()
    Y_hat = A_mean * base
    mape = np.mean(np.abs((Y - Y_hat) / Y)) * 100

    out["A_t"] = A
    out["A_growth_pct"] = out["A_t"].pct_change() * 100
    out["Y_hat"] = Y_hat
    out["APE_pct"] = np.abs((Y - Y_hat) / Y) * 100

    return out, A_mean, mape


def growth_decomposition(df, alpha, beta, gamma, delta, theta):
    years = df["Year"].values

    dlnY = np.diff(np.log(df["GDP_trillion_VND"].values))
    dlnK = np.diff(np.log(df["K"].values))
    dlnL = np.diff(np.log(df["L"].values))
    dlnD = np.diff(np.log(df["D"].values))
    dlnAI = np.diff(np.log(df["AI"].values))
    dlnH = np.diff(np.log(df["H"].values))
    dlnA = np.diff(np.log(df["A_t"].values))

    detail = pd.DataFrame(
        {
            "period": [f"{years[i]}-{years[i + 1]}" for i in range(len(years) - 1)],
            "GDP_growth_pct": dlnY * 100,
            "K": alpha * dlnK * 100,
            "L": beta * dlnL * 100,
            "D": gamma * dlnD * 100,
            "AI": delta * dlnAI * 100,
            "H": theta * dlnH * 100,
            "TFP": dlnA * 100,
        }
    )

    avg_growth = detail["GDP_growth_pct"].mean()

    avg = pd.DataFrame(
        {
            "factor": ["K", "L", "D", "AI", "H", "TFP"],
            "avg_log_contribution": [
                detail["K"].mean(),
                detail["L"].mean(),
                detail["D"].mean(),
                detail["AI"].mean(),
                detail["H"].mean(),
                detail["TFP"].mean(),
            ],
        }
    )

    avg["share_of_growth_pct"] = avg["avg_log_contribution"] / avg_growth * 100

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

    Y = A * (K**alpha) * (L**beta) * (D**gamma) * (AI**delta) * (H**theta)

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


def tfp_trend_text(df):
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


def render():
    st.markdown(
        """
        <style>
        .bai1-card {
            padding: 34px 42px;
            border-radius: 0 0 22px 22px;
            border-left: 6px solid #3b82f6;
            background: white;
            box-shadow: 0 12px 35px rgba(15, 23, 42, 0.08);
            margin-bottom: 28px;
        }
        .bai1-title {
            font-size: 2.1rem;
            font-weight: 850;
            color: #172b4d;
            margin-bottom: 12px;
        }
        .bai1-sub {
            color: #64748b;
            font-size: 1.05rem;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.45rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="bai1-card">
            <div class="bai1-title">Bài 1. Hàm sản xuất Cobb-Douglas mở rộng</div>
            <div class="bai1-sub">Phân tích TFP, dự báo GDP và đóng góp tăng trưởng</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =========================
    # Sidebar
    # =========================
    st.sidebar.header("Bài 1")

    alpha = st.sidebar.slider("α - K", 0.00, 0.80, 0.33, 0.01)
    beta = st.sidebar.slider("β - L", 0.00, 0.80, 0.42, 0.01)
    gamma = st.sidebar.slider("γ - D", 0.00, 0.50, 0.10, 0.01)
    delta = st.sidebar.slider("δ - AI", 0.00, 0.50, 0.08, 0.01)
    theta = st.sidebar.slider("θ - H", 0.00, 0.50, 0.07, 0.01)

    st.sidebar.divider()

    target_D = st.sidebar.slider("D 2030 (%)", 19.5, 45.0, 30.0, 0.5)
    target_AI = st.sidebar.slider("AI 2030", 80.1, 180.0, 100.0, 1.0)
    target_H = st.sidebar.slider("H 2030 (%)", 29.2, 60.0, 35.0, 0.5)

    gK = st.sidebar.slider("Tăng K/năm (%)", 0.0, 15.0, 6.0, 0.1) / 100
    gL = st.sidebar.slider("Tăng L/năm (%)", -2.0, 10.0, 6.0, 0.1) / 100
    gTFP = st.sidebar.slider("Tăng TFP/năm (%)", -2.0, 5.0, 1.2, 0.1) / 100

    df = load_data()
    model_df, A_mean, mape = compute_model(df, alpha, beta, gamma, delta, theta)
    detail_df, avg_df, avg_growth = growth_decomposition(model_df, alpha, beta, gamma, delta, theta)
    forecast_df = forecast_2030(
        model_df, alpha, beta, gamma, delta, theta,
        target_D, target_AI, target_H, gK, gL, gTFP
    )

    trend, tfp_change = tfp_trend_text(model_df)

    largest_new = (
        avg_df[avg_df["factor"].isin(["D", "AI", "H"])]
        .assign(abs_share=lambda x: x["share_of_growth_pct"].abs())
        .sort_values("abs_share", ascending=False)
        .iloc[0]
    )

    tabs = st.tabs(
        [
            "Kết quả TFP",
            "Phân rã tăng trưởng",
            "Dự báo 2030",
            "Thảo luận chính sách",
        ]
    )

    # =====================================================
    # TAB 1 — TFP + MAPE
    # =====================================================
    with tabs[0]:
        st.header("TFP và GDP dự báo")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("A trung bình", f"{A_mean:,.2f}")
        c2.metric("MAPE", f"{mape:.2f}%")
        c3.metric("TFP 2025", f"{model_df['A_t'].iloc[-1]:,.2f}")
        c4.metric("Xu hướng", trend)

        table = model_df[
            ["Year", "GDP_trillion_VND", "D", "A_t", "Y_hat", "APE_pct"]
        ].copy()

        table = table.rename(
            columns={
                "Year": "year",
                "GDP_trillion_VND": "GDP",
                "D": "D_pct",
                "A_t": "TFP_A",
                "Y_hat": "GDP_hat",
                "APE_pct": "APE_pct",
            }
        )

        st.dataframe(table.round(4), use_container_width=True, hide_index=True)

        fig1 = go.Figure()
        fig1.add_trace(
            go.Scatter(
                x=model_df["Year"],
                y=model_df["A_t"],
                mode="lines+markers",
                name="TFP A_t",
            )
        )
        fig1.update_layout(
            title="Xu hướng TFP A_t",
            xaxis_title="year",
            yaxis_title="A_t",
            height=420,
        )
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=model_df["Year"],
                y=model_df["GDP_trillion_VND"],
                mode="lines+markers",
                name="GDP thực tế",
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=model_df["Year"],
                y=model_df["Y_hat"],
                mode="lines+markers",
                name="GDP dự báo",
            )
        )
        fig2.update_layout(
            title="GDP thực tế và GDP dự báo",
            xaxis_title="year",
            yaxis_title="GDP, nghìn tỷ VND",
            height=420,
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.success(
            f"TFP giai đoạn 2020-2025 có xu hướng {trend}; MAPE = {mape:.2f}%."
        )

    # =====================================================
    # TAB 2 — PHÂN RÃ
    # =====================================================
    with tabs[1]:
        st.header("Phân rã tăng trưởng")

        c1, c2, c3 = st.columns(3)
        c1.metric("GDP tăng trưởng TB", f"{avg_growth:.2f}%")
        c2.metric("Yếu tố mới nổi bật", largest_new["factor"])
        c3.metric("Tỷ trọng", f"{largest_new['share_of_growth_pct']:.1f}%")

        st.dataframe(avg_df.round(4), use_container_width=True, hide_index=True)

        fig = px.bar(
            avg_df,
            x="factor",
            y="share_of_growth_pct",
            text=avg_df["share_of_growth_pct"].round(1).astype(str) + "%",
            title="Tỷ trọng đóng góp tăng trưởng",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            xaxis_title="factor",
            yaxis_title="share_of_growth_pct",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Chi tiết theo giai đoạn"):
            st.dataframe(detail_df.round(4), use_container_width=True, hide_index=True)

    # =====================================================
    # TAB 3 — DỰ BÁO 2030
    # =====================================================
    with tabs[2]:
        st.header("Dự báo GDP 2030")

        row_2030 = forecast_df[forecast_df["Year"] == 2030].iloc[0]
        y_2025 = model_df["GDP_trillion_VND"].iloc[-1]
        y_2030 = row_2030["GDP_forecast"]
        growth_2030 = (y_2030 / y_2025 - 1) * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("GDP 2025", f"{y_2025:,.1f}")
        c2.metric("GDP 2030", f"{y_2030:,.1f}")
        c3.metric("Tăng 2025-2030", f"{growth_2030:.1f}%")

        summary_2030 = pd.DataFrame(
            {
                "indicator": [
                    "K_2030",
                    "L_2030",
                    "D_2030",
                    "AI_2030",
                    "H_2030",
                    "A_2030",
                    "GDP_2030_forecast",
                ],
                "value": [
                    row_2030["K"],
                    row_2030["L"],
                    row_2030["D"],
                    row_2030["AI"],
                    row_2030["H"],
                    row_2030["A_t"],
                    row_2030["GDP_forecast"],
                ],
            }
        )

        st.dataframe(summary_2030.round(4), use_container_width=True, hide_index=True)

        actual = model_df[["Year", "GDP_trillion_VND"]].rename(
            columns={"GDP_trillion_VND": "GDP"}
        )
        actual["type"] = "Thực tế"

        future = forecast_df[["Year", "GDP_forecast"]].rename(
            columns={"GDP_forecast": "GDP"}
        )
        future["type"] = "Dự báo"

        chart_df = pd.concat([actual, future], ignore_index=True)

        fig = px.line(
            chart_df,
            x="Year",
            y="GDP",
            color="type",
            markers=True,
            title="GDP thực tế và dự báo",
        )
        fig.update_layout(
            xaxis_title="year",
            yaxis_title="GDP, nghìn tỷ VND",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # TAB 4 — CHÍNH SÁCH
    # =====================================================
    with tabs[3]:
        st.header("Thảo luận chính sách")

        st.markdown(
            f"""
            **a) TFP:** TFP có xu hướng **{trend}** trong giai đoạn 2020-2025, thay đổi khoảng **{tfp_change:.2f}%**.

            **b) Yếu tố mới đóng góp nhiều nhất:** Trong nhóm D, AI, H, yếu tố nổi bật nhất là **{largest_new["factor"]}**, chiếm khoảng **{largest_new["share_of_growth_pct"]:.1f}%** tăng trưởng bình quân.

            **c) Mục tiêu kinh tế số 30% GDP năm 2030:** Có cơ sở khả thi nếu đồng thời đạt 3 điều kiện: mở rộng kinh tế số, tăng năng lực AI và cải thiện nhân lực số.
            """
        )

        st.info(
            "Hàm ý: không chỉ tăng D, mà cần nâng đồng thời AI, H và TFP để tăng trưởng bền vững."
        )


def run():
    render()

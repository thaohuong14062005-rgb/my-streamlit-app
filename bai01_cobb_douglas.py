# bai01_cobb_douglas.py

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# BÀI 1 — HÀM SẢN XUẤT COBB-DOUGLAS MỞ RỘNG
# Gọn, đúng từng mục 1.4.1 -> 1.5, không dùng 3D
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

        keep_cols = ["Year", "GDP_trillion_VND", "digital_economy_share_GDP_pct"]
        macro = macro[[c for c in keep_cols if c in macro.columns]].copy()

        macro["Year"] = pd.to_numeric(macro["Year"], errors="coerce")
        macro["GDP_trillion_VND"] = pd.to_numeric(
            macro.get("GDP_trillion_VND"), errors="coerce"
        )

        if "digital_economy_share_GDP_pct" in macro.columns:
            macro["D_csv"] = pd.to_numeric(
                macro["digital_economy_share_GDP_pct"], errors="coerce"
            )
            macro = macro.drop(columns=["digital_economy_share_GDP_pct"])
        else:
            macro["D_csv"] = np.nan

        macro = macro.dropna(subset=["Year"]).copy()
        macro["Year"] = macro["Year"].astype(int)

        df = base.drop(columns=["GDP_trillion_VND", "D"]).merge(
            macro, on="Year", how="left"
        )

        fallback = base[["Year", "GDP_trillion_VND", "D"]].copy()
        df = df.merge(fallback, on="Year", how="left", suffixes=("", "_base"))

        df["GDP_trillion_VND"] = df["GDP_trillion_VND"].fillna(
            df["GDP_trillion_VND_base"]
        )
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

    dlnY = np.diff(np.log(df["GDP_trillion_VND"].values))
    dlnK = np.diff(np.log(df["K"].values))
    dlnL = np.diff(np.log(df["L"].values))
    dlnD = np.diff(np.log(df["D"].values))
    dlnAI = np.diff(np.log(df["AI"].values))
    dlnH = np.diff(np.log(df["H"].values))
    dlnA = np.diff(np.log(df["A_t"].values))

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
            font-size: 1.42rem;
        }
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
    largest_new = new_factors.assign(
        abs_share=lambda x: x["Tỷ trọng tăng trưởng (%)"].abs()
    ).sort_values("abs_share", ascending=False).iloc[0]

    # =========================
    # Tabs đúng từng phần
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

        st.dataframe(table_141.round(3), use_container_width=True, hide_index=True)

        fig = px.line(
            model_df,
            x="Year",
            y="A_t",
            markers=True,
            title="Xu hướng TFP A_t",
        )
        fig.update_layout(xaxis_title="Năm", yaxis_title="A_t", height=430)
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

        st.dataframe(table_142.round(3), use_container_width=True, hide_index=True)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=model_df["Year"],
                y=model_df["GDP_trillion_VND"],
                mode="lines+markers",
                name="GDP thực tế",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=model_df["Year"],
                y=model_df["Y_hat"],
                mode="lines+markers",
                name="GDP dự báo",
            )
        )
        fig.update_layout(
            title="GDP thực tế và GDP dự báo",
            xaxis_title="Năm",
            yaxis_title="GDP, nghìn tỷ VND",
            height=430,
        )
        st.plotly_chart(fig, use_container_width=True)

        fig_ape = px.bar(
            model_df,
            x="Year",
            y="APE_pct",
            text=model_df["APE_pct"].round(2),
            title="APE theo năm",
        )
        fig_ape.update_traces(textposition="outside")
        fig_ape.update_layout(xaxis_title="Năm", yaxis_title="APE (%)", height=390)
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
        st.dataframe(avg_df.round(4), use_container_width=True, hide_index=True)

        fig_avg = px.bar(
            avg_df,
            x="Yếu tố",
            y="Tỷ trọng tăng trưởng (%)",
            text=avg_df["Tỷ trọng tăng trưởng (%)"].round(1).astype(str) + "%",
            title="Tỷ trọng đóng góp tăng trưởng",
        )
        fig_avg.update_traces(textposition="outside")
        fig_avg.update_layout(
            xaxis_title="Yếu tố",
            yaxis_title="Tỷ trọng (%)",
            height=430,
        )
        st.plotly_chart(fig_avg, use_container_width=True)

        st.subheader("Theo từng giai đoạn")
        st.dataframe(detail_df.round(4), use_container_width=True, hide_index=True)

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
        )
        fig_detail.update_layout(barmode="relative", height=430)
        st.plotly_chart(fig_detail, use_container_width=True)

        st.success(
            f"Trong nhóm D, AI, H, yếu tố đóng góp nổi bật nhất là {largest_new['Yếu tố']}."
        )

    # =====================================================
    # 1.4.4
    # =====================================================
    with tabs[3]:
        st.header("1.4.4. Dự báo GDP 2030")

        row_2030 = forecast_df[forecast_df["Year"] == 2030].iloc[0]
        y_2025 = model_df["GDP_trillion_VND"].iloc[-1]
        y_2030 = row_2030["GDP_forecast"]
        growth_2030 = (y_2030 / y_2025 - 1) * 100

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

        st.dataframe(table_144.round(3), use_container_width=True, hide_index=True)

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
        )
        fig.update_layout(yaxis_title="GDP, nghìn tỷ VND", height=440)
        st.plotly_chart(fig, use_container_width=True)

        st.success(f"GDP dự báo năm 2030 đạt khoảng {y_2030:,.1f} nghìn tỷ VND.")

    # =====================================================
    # 1.5
    # =====================================================
    with tabs[4]:
        st.header("1.5. Thảo luận chính sách")

        st.markdown(
            f"""
            **a) TFP:** TFP có xu hướng **{trend}** trong giai đoạn 2020-2025, thay đổi khoảng **{tfp_change:.2f}%**.

            **b) Yếu tố mới nổi bật:** Trong nhóm **D, AI, H**, yếu tố đóng góp lớn nhất là **{largest_new["Yếu tố"]}**, chiếm khoảng **{largest_new["Tỷ trọng tăng trưởng (%)"]:.1f}%** tăng trưởng bình quân.

            **c) Mục tiêu kinh tế số 30% GDP năm 2030:** Có cơ sở khả thi nếu đồng thời tăng **D**, mở rộng **AI**, nâng **H** và duy trì tăng trưởng **TFP**.
            """
        )

        st.info(
            "Hàm ý chính sách: không chỉ tăng tỷ trọng kinh tế số, mà cần nâng đồng thời năng lực AI, nhân lực số và năng suất tổng hợp."
        )


def run():
    render()

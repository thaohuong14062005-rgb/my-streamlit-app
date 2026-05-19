import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def render():
    st.title("🌱 Bài 1 — Hàm sản xuất Cobb-Douglas mở rộng với AI và số hóa")

    df = pd.DataFrame({
        "Năm": [2020, 2021, 2022, 2023, 2024, 2025],
        "Y_GDP": [8044.4, 8487.5, 9513.3, 10221.8, 11511.9, 12847.6],
        "K_von": [16500, 17800, 19600, 21300, 23500, 25900],
        "L_lao_dong": [53.6, 50.5, 51.7, 52.4, 52.9, 53.4],
        "D_kinh_te_so": [12.0, 12.7, 14.3, 16.5, 18.3, 19.5],
        "AI_dn_so": [55.6, 60.2, 65.4, 67.0, 73.8, 80.1],
        "H_nhan_luc_so": [24.1, 26.1, 26.2, 27.0, 28.4, 29.2],
    })

    st.markdown("## 1. Mô hình Cobb-Douglas mở rộng")

    st.latex(
        r"Y_t = A_t \cdot K_t^\alpha \cdot L_t^\beta \cdot D_t^\gamma \cdot AI_t^\delta \cdot H_t^\theta"
    )

    st.latex(
        r"\alpha + \beta + \gamma + \delta + \theta = 1"
    )

    st.sidebar.markdown("### Tham số Bài 1")

    alpha = st.sidebar.slider("α - Vốn K", 0.10, 0.60, 0.33, 0.01)
    beta = st.sidebar.slider("β - Lao động L", 0.10, 0.70, 0.42, 0.01)
    gamma = st.sidebar.slider("γ - Số hóa D", 0.01, 0.30, 0.10, 0.01)
    delta = st.sidebar.slider("δ - AI", 0.01, 0.30, 0.08, 0.01)

    theta = 1 - alpha - beta - gamma - delta

    st.sidebar.metric("θ - Nhân lực số H", f"{theta:.3f}")

    if theta <= 0:
        st.error("Tổng α + β + γ + δ đang lớn hơn hoặc bằng 1, làm θ ≤ 0.")
        st.stop()

    df["A_TFP"] = df["Y_GDP"] / (
        df["K_von"] ** alpha
        * df["L_lao_dong"] ** beta
        * df["D_kinh_te_so"] ** gamma
        * df["AI_dn_so"] ** delta
        * df["H_nhan_luc_so"] ** theta
    )

    A_mean = df["A_TFP"].mean()

    df["Y_du_bao"] = A_mean * (
        df["K_von"] ** alpha
        * df["L_lao_dong"] ** beta
        * df["D_kinh_te_so"] ** gamma
        * df["AI_dn_so"] ** delta
        * df["H_nhan_luc_so"] ** theta
    )

    df["Sai_so_tuyet_doi"] = abs(df["Y_GDP"] - df["Y_du_bao"])
    df["APE_%"] = df["Sai_so_tuyet_doi"] / df["Y_GDP"] * 100

    mape = df["APE_%"].mean()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Dữ liệu",
        "📈 TFP A_t",
        "📊 Dự báo & MAPE",
        "🔮 Dự báo 2030"
    ])

    with tab1:
        st.subheader("Dữ liệu Việt Nam 2020-2025")
        st.dataframe(df[[
            "Năm",
            "Y_GDP",
            "K_von",
            "L_lao_dong",
            "D_kinh_te_so",
            "AI_dn_so",
            "H_nhan_luc_so"
        ]], use_container_width=True)

    with tab2:
        st.subheader("Câu 1.4.1 — Ước lượng TFP A_t")

        st.latex(
            r"A_t = \frac{Y_t}{K_t^\alpha L_t^\beta D_t^\gamma AI_t^\delta H_t^\theta}"
        )

        st.dataframe(df[["Năm", "A_TFP"]].round(4), use_container_width=True)

        fig = px.line(
            df,
            x="Năm",
            y="A_TFP",
            markers=True,
            title="Xu hướng TFP A_t theo năm"
        )
        st.plotly_chart(fig, use_container_width=True)

        if df["A_TFP"].iloc[-1] > df["A_TFP"].iloc[0]:
            st.success(
                "TFP có xu hướng tăng trong giai đoạn 2020-2025, cho thấy chất lượng tăng trưởng được cải thiện."
            )
        else:
            st.warning(
                "TFP có xu hướng giảm, cho thấy tăng trưởng phụ thuộc nhiều hơn vào mở rộng đầu vào."
            )

    with tab3:
        st.subheader("Câu 1.4.2 — Dự báo GDP và MAPE")

        col1, col2, col3 = st.columns(3)

        col1.metric("A trung bình", f"{A_mean:.4f}")
        col2.metric("MAPE", f"{mape:.2f}%")
        col3.metric("Sai số TB", f"{df['Sai_so_tuyet_doi'].mean():,.2f}")

        st.dataframe(df[[
            "Năm",
            "Y_GDP",
            "Y_du_bao",
            "Sai_so_tuyet_doi",
            "APE_%"
        ]].round(3), use_container_width=True)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df["Năm"],
            y=df["Y_GDP"],
            mode="lines+markers",
            name="GDP thực tế"
        ))

        fig.add_trace(go.Scatter(
            x=df["Năm"],
            y=df["Y_du_bao"],
            mode="lines+markers",
            name="GDP dự báo",
            line=dict(dash="dash")
        ))

        fig.update_layout(
            title="So sánh GDP thực tế và GDP dự báo",
            xaxis_title="Năm",
            yaxis_title="GDP, nghìn tỷ VND"
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Câu 1.4.4 — Dự báo GDP Việt Nam năm 2030")

        target_D = st.number_input("D 2030 - Kinh tế số/GDP (%)", value=30.0)
        target_AI = st.number_input("AI 2030 - Nghìn DN số", value=100.0)
        target_H = st.number_input("H 2030 - Lao động qua đào tạo (%)", value=35.0)

        growth_K = st.number_input("Tăng trưởng K mỗi năm", value=0.06, step=0.005, format="%.3f")
        growth_L = st.number_input("Tăng trưởng L mỗi năm", value=0.06, step=0.005, format="%.3f")
        growth_A = st.number_input("Tăng trưởng TFP mỗi năm", value=0.012, step=0.001, format="%.3f")

        base = df.iloc[-1]

        rows = []

        for year in range(2025, 2031):
            step = year - 2025

            K = base["K_von"] * ((1 + growth_K) ** step)
            L = base["L_lao_dong"] * ((1 + growth_L) ** step)
            A = base["A_TFP"] * ((1 + growth_A) ** step)

            D = base["D_kinh_te_so"] + (target_D - base["D_kinh_te_so"]) * step / 5
            AI = base["AI_dn_so"] + (target_AI - base["AI_dn_so"]) * step / 5
            H = base["H_nhan_luc_so"] + (target_H - base["H_nhan_luc_so"]) * step / 5

            Y_hat = A * (
                K ** alpha
                * L ** beta
                * D ** gamma
                * AI ** delta
                * H ** theta
            )

            rows.append({
                "Năm": year,
                "A_TFP": A,
                "K": K,
                "L": L,
                "D": D,
                "AI": AI,
                "H": H,
                "GDP_dự_báo": Y_hat,
                "Tăng so với 2025, %": (Y_hat / base["Y_GDP"] - 1) * 100
            })

        forecast = pd.DataFrame(rows)

        gdp_2030 = forecast[forecast["Năm"] == 2030]["GDP_dự_báo"].iloc[0]
        growth_2030 = forecast[forecast["Năm"] == 2030]["Tăng so với 2025, %"].iloc[0]

        col1, col2 = st.columns(2)

        col1.metric("GDP dự báo 2030", f"{gdp_2030:,.1f} nghìn tỷ VND")
        col2.metric("Tăng so với 2025", f"{growth_2030:.2f}%")

        st.dataframe(forecast.round(3), use_container_width=True)

        fig = px.line(
            forecast,
            x="Năm",
            y="GDP_dự_báo",
            markers=True,
            title="Quỹ đạo GDP dự báo 2025-2030"
        )
        st.plotly_chart(fig, use_container_width=True)

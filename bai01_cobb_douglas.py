# bai01_cobb_douglas_ai_advanced.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Bài 1 nâng cấp", layout="wide")

def render():
    st.title("🌱 Bài 1 — Cobb-Douglas nâng cấp với AI & số hóa")

    # =========================
    # Dữ liệu Việt Nam 2020-2025
    # =========================
    df = pd.DataFrame({
        "Năm": [2020,2021,2022,2023,2024,2025],
        "Y_GDP":[8044.4,8487.5,9513.3,10221.8,11511.9,12847.6],
        "K_von":[16500,17800,19600,21300,23500,25900],
        "L_lao_dong":[53.6,50.5,51.7,52.4,52.9,53.4],
        "D_kinh_te_so":[12.0,12.7,14.3,16.5,18.3,19.5],
        "AI_dn_so":[55.6,60.2,65.4,67.0,73.8,80.1],
        "H_nhan_luc_so":[24.1,26.1,26.2,27.0,28.4,29.2],
    })

    # =========================
    # Sidebar: slider tham số
    # =========================
    st.sidebar.markdown("### Tham số Cobb-Douglas")
    alpha = st.sidebar.slider("α - Vốn K", 0.10, 0.60, 0.33, 0.01)
    beta = st.sidebar.slider("β - Lao động L", 0.10, 0.70, 0.42, 0.01)
    gamma = st.sidebar.slider("γ - Số hóa D", 0.01, 0.30, 0.10, 0.01)
    delta = st.sidebar.slider("δ - AI", 0.01, 0.30, 0.08, 0.01)
    theta = 1 - alpha - beta - gamma - delta
    st.sidebar.metric("θ - Nhân lực số H", f"{theta:.3f}")
    if theta <= 0:
        st.error("Tổng α+β+γ+δ >=1, θ <=0. Chọn lại tham số!")
        st.stop()

    # =========================
    # Tính TFP A_t
    # =========================
    df["A_TFP"] = df["Y_GDP"] / (
        df["K_von"]**alpha * df["L_lao_dong"]**beta *
        df["D_kinh_te_so"]**gamma * df["AI_dn_so"]**delta *
        df["H_nhan_luc_so"]**theta
    )
    A_mean = df["A_TFP"].mean()

    # =========================
    # Dự báo GDP
    # =========================
    df["Y_du_bao"] = A_mean * (
        df["K_von"]**alpha * df["L_lao_dong"]**beta *
        df["D_kinh_te_so"]**gamma * df["AI_dn_so"]**delta *
        df["H_nhan_luc_so"]**theta
    )
    df["Sai_so"] = abs(df["Y_GDP"] - df["Y_du_bao"])
    df["APE_%"] = df["Sai_so"] / df["Y_GDP"] * 100
    mape = df["APE_%"].mean()

    # =========================
    # Phân rã tăng trưởng 2020-2025
    # =========================
    growth_df = pd.DataFrame()
    for i in range(1,len(df)):
        delta_lnY = np.log(df["Y_GDP"].iloc[i]/df["Y_GDP"].iloc[i-1])
        contrib_K = alpha * np.log(df["K_von"].iloc[i]/df["K_von"].iloc[i-1])
        contrib_L = beta * np.log(df["L_lao_dong"].iloc[i]/df["L_lao_dong"].iloc[i-1])
        contrib_D = gamma * np.log(df["D_kinh_te_so"].iloc[i]/df["D_kinh_te_so"].iloc[i-1])
        contrib_AI = delta * np.log(df["AI_dn_so"].iloc[i]/df["AI_dn_so"].iloc[i-1])
        contrib_H = theta * np.log(df["H_nhan_luc_so"].iloc[i]/df["H_nhan_luc_so"].iloc[i-1])
        contrib_TFP = delta_lnY - (contrib_K+contrib_L+contrib_D+contrib_AI+contrib_H)
        growth_df = pd.concat([growth_df,pd.DataFrame({
            "Năm": [df["Năm"].iloc[i]],
            "K": [contrib_K],
            "L": [contrib_L],
            "D": [contrib_D],
            "AI": [contrib_AI],
            "H": [contrib_H],
            "TFP": [contrib_TFP],
        })], ignore_index=True)

    # =========================
    # Tabs Streamlit
    # =========================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Dữ liệu", "📈 TFP A_t", "📊 Dự báo & MAPE",
        "📉 Phân rã tăng trưởng", "🔮 Dự báo 2030"
    ])

    with tab1:
        st.subheader("Dữ liệu Việt Nam 2020-2025")
        st.dataframe(df.round(3), use_container_width=True)

    with tab2:
        st.subheader("Ước lượng TFP A_t")
        st.dataframe(df[["Năm","A_TFP"]].round(4), use_container_width=True)
        fig = px.line(df,x="Năm",y="A_TFP",markers=True,title="TFP A_t")
        st.plotly_chart(fig, use_container_width=True)
        if df["A_TFP"].iloc[-1]>df["A_TFP"].iloc[0]:
            st.success("TFP tăng → chất lượng tăng trưởng cải thiện")
        else:
            st.warning("TFP giảm → tăng trưởng chủ yếu dựa vào đầu vào")

    with tab3:
        st.subheader("So sánh GDP thực tế và dự báo")
        col1,col2,col3 = st.columns(3)
        col1.metric("A trung bình", f"{A_mean:.4f}")
        col2.metric("MAPE", f"{mape:.2f}%")
        col3.metric("Sai số TB", f"{df['Sai_so'].mean():,.2f}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Năm"],y=df["Y_GDP"],mode="lines+markers",name="GDP thực tế"))
        fig.add_trace(go.Scatter(x=df["Năm"],y=df["Y_du_bao"],mode="lines+markers",name="GDP dự báo",line=dict(dash="dash")))
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Phân rã tăng trưởng GDP (%)")
        growth_df_norm = growth_df.set_index("Năm")*100
        fig = go.Figure()
        fig.add_trace(go.Bar(name="K",x=growth_df_norm.index,y=growth_df_norm["K"]))
        fig.add_trace(go.Bar(name="L",x=growth_df_norm.index,y=growth_df_norm["L"]))
        fig.add_trace(go.Bar(name="D",x=growth_df_norm.index,y=growth_df_norm["D"]))
        fig.add_trace(go.Bar(name="AI",x=growth_df_norm.index,y=growth_df_norm["AI"]))
        fig.add_trace(go.Bar(name="H",x=growth_df_norm.index,y=growth_df_norm["H"]))
        fig.add_trace(go.Bar(name="TFP",x=growth_df_norm.index,y=growth_df_norm["TFP"]))
        fig.update_layout(barmode='stack',title="Đóng góp từng yếu tố vào tăng trưởng")
        st.plotly_chart(fig, use_container_width=True)

        # AI phân tích
        top_factor = growth_df_norm[["K","L","D","AI","H","TFP"]].sum().idxmax()
        st.info(f"Yếu tố đóng góp lớn nhất 2020-2025: {top_factor}")

    with tab5:
        st.subheader("Dự báo GDP 2030")
        target_D = st.number_input("D 2030 (%)", value=30.0)
        target_AI = st.number_input("AI 2030 (nghìn DN)", value=100.0)
        target_H = st.number_input("H 2030 (%)", value=35.0)
        growth_K = st.number_input("Tăng trưởng K", value=0.06)
        growth_L = st.number_input("Tăng trưởng L", value=0.06)
        growth_A = st.number_input("Tăng trưởng TFP", value=0.012)
        base = df.iloc[-1]
        rows=[]
        for year in range(2025,2031):
            step = year-2025
            K=base["K_von"]*(1+growth_K)**step
            L=base["L_lao_dong"]*(1+growth_L)**step
            A=base["A_TFP"]*(1+growth_A)**step
            D=base["D_kinh_te_so"] + (target_D-base["D_kinh_te_so"])*step/5
            AI=base["AI_dn_so"] + (target_AI-base["AI_dn_so"])*step/5
            H=base["H_nhan_luc_so"] + (target_H-base["H_nhan_luc_so"])*step/5
            Y_hat = A*(K**alpha*L**beta*D**gamma*AI**delta*H**theta)
            rows.append({"Năm":year,"K":K,"L":L,"D":D,"AI":AI,"H":H,"GDP_dự_báo":Y_hat})
        forecast = pd.DataFrame(rows)
        st.dataframe(forecast.round(2))
        fig = px.line(forecast,x="Năm",y="GDP_dự_báo",markers=True,title="GDP dự báo 2025-2030")
        st.plotly_chart(fig, use_container_width=True)

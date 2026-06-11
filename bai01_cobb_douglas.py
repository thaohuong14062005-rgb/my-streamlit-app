# bai01_cobb_douglas.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# =========================
# Hàm chính
# =========================
def render():
    st.title("🌱 Bài 1 — Hàm sản xuất Cobb-Douglas")

    st.markdown("""
    #### Mục tiêu:
    - Ước lượng TFP (A_t) từ hàm sản xuất Cobb-Douglas
    - Phân rã tăng trưởng GDP
    - Dự báo GDP Việt Nam đến 2030
    - Thảo luận chính sách kinh tế số và TFP
    """)

    # =========================
    # Đọc dữ liệu
    # =========================
    try:
        df = pd.read_csv('vietnam_macro_2020_2025.csv')
        st.success("Đã đọc dữ liệu vietnam_macro_2020_2025.csv")
    except:
        st.warning("Không tìm thấy file vietnam_macro_2020_2025.csv, dùng dữ liệu dự phòng")
        # Dữ liệu dự phòng (thay bằng dữ liệu thực tế nếu có CSV)
        df = pd.DataFrame({
            'Year': np.arange(2020, 2026),
            'GDP_trillion_VND': [5100, 5400, 5800, 6200, 6700, 7200]
        })

    years = df['Year'].values
    Y = df['GDP_trillion_VND'].values

    # =========================
    # Nhập dữ liệu các yếu tố sản xuất
    # =========================
    K = np.array([16500, 17800, 19600, 21300, 23500, 25900])
    L = np.array([53.6, 50.5, 51.7, 52.4, 52.9, 53.4])
    D = np.array([12.0, 12.7, 14.3, 16.5, 18.3, 19.5])
    AI = np.array([55.6, 60.2, 65.4, 67.0, 73.8, 80.1])
    H = np.array([24.1, 26.1, 26.2, 27.0, 28.4, 29.2])

    # Chỉ số hàm sản xuất Cobb-Douglas
    alpha, beta, gamma, delta, theta = 0.33, 0.42, 0.10, 0.08, 0.07

    # =========================
    # 1.4.1 Ước lượng TFP
    # =========================
    A_t = Y / (K**alpha * L**beta * D**gamma * AI**delta * H**theta)
    st.subheader("1.4.1 — TFP (A_t) theo năm")
    df['A_t'] = A_t
    st.dataframe(df[['Year','A_t']])

    # Vẽ biểu đồ xu hướng TFP
    fig1 = px.line(df, x='Year', y='A_t', markers=True, title='TFP (A_t) Việt Nam 2020-2025')
    st.plotly_chart(fig1, use_container_width=True)

    # =========================
    # 1.4.2 Dự báo với A trung bình, tính MAPE
    # =========================
    A_mean = A_t.mean()
    Y_hat = A_mean * (K**alpha * L**beta * D**gamma * AI**delta * H**theta)
    MAPE = np.mean(np.abs(Y_hat - Y) / Y) * 100

    st.subheader("1.4.2 — Dự báo GDP với A trung bình")
    df['Y_hat'] = Y_hat
    st.dataframe(df[['Year','GDP_trillion_VND','Y_hat']])
    st.write(f"**MAPE:** {MAPE:.2f}%")

    # =========================
    # 1.4.3 Phân rã tăng trưởng
    # =========================
    st.subheader("1.4.3 — Phân rã tăng trưởng GDP")
    # Tính tốc độ tăng trưởng
    gY = np.diff(Y)/Y[:-1]
    gK = np.diff(K)/K[:-1]
    gL = np.diff(L)/L[:-1]
    gD = np.diff(D)/D[:-1]
    gAI = np.diff(AI)/AI[:-1]
    gH = np.diff(H)/H[:-1]
    gA = np.diff(A_t)/A_t[:-1]

    contrib_K = alpha * gK
    contrib_L = beta * gL
    contrib_D = gamma * gD
    contrib_AI = delta * gAI
    contrib_H = theta * gH
    contrib_TFP = gA

    growth_df = pd.DataFrame({
        'Year': [f"{years[i]}-{years[i+1]}" for i in range(len(years)-1)],
        'K': contrib_K,
        'L': contrib_L,
        'D': contrib_D,
        'AI': contrib_AI,
        'H': contrib_H,
        'TFP': contrib_TFP
    })
    st.dataframe(growth_df)

    # Vẽ biểu đồ cột stacked
    fig2 = px.bar(growth_df, x='Year', y=['K','L','D','AI','H','TFP'],
                  title='Phân rã đóng góp tăng trưởng GDP', text_auto='.2f')
    st.plotly_chart(fig2, use_container_width=True)

    # =========================
    # 1.4.4 Dự báo 2030
    # =========================
    st.subheader("1.4.4 — Mô phỏng GDP đến 2030")
    # Sidebar điều chỉnh tham số
    st.sidebar.header("Chỉnh tham số dự báo 2030")
    D_2030 = st.sidebar.number_input("D (%) năm 2030", value=30.0)
    AI_2030 = st.sidebar.number_input("AI (nghìn DN) năm 2030", value=100.0)
    H_2030 = st.sidebar.number_input("H (%) năm 2030", value=35.0)
    gK_2030 = st.sidebar.number_input("Tăng trưởng K (%/năm)", value=6.0)/100
    gL_2030 = st.sidebar.number_input("Tăng trưởng L (%/năm)", value=6.0)/100
    gTFP_2030 = st.sidebar.number_input("Tăng trưởng TFP (%/năm)", value=1.2)/100

    # Dự báo 2026-2030
    years_future = np.arange(2026, 2031)
    K_future = [K[-1]*(1+gK_2030)**i for i in range(1,6)]
    L_future = [L[-1]*(1+gL_2030)**i for i in range(1,6)]
    D_future = np.linspace(D[-1], D_2030, 5)
    AI_future = np.linspace(AI[-1], AI_2030, 5)
    H_future = np.linspace(H[-1], H_2030, 5)
    A_future = [A_t[-1]*(1+gTFP_2030)**i for i in range(1,6)]

    Y_future = np.array(A_future) * (np.array(K_future)**alpha * np.array(L_future)**beta *
                                     np.array(D_future)**gamma * np.array(AI_future)**delta *
                                     np.array(H_future)**theta)
    forecast_df = pd.DataFrame({
        'Year': years_future,
        'K': K_future,
        'L': L_future,
        'D': D_future,
        'AI': AI_future,
        'H': H_future,
        'TFP': A_future,
        'GDP_forecast': Y_future
    })
    st.dataframe(forecast_df)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=years_future, y=Y_future, mode='lines+markers', name='GDP Forecast'))
    fig3.update_layout(title="Dự báo GDP Việt Nam 2026-2030", xaxis_title="Năm", yaxis_title="GDP (tr VND)")
    st.plotly_chart(fig3, use_container_width=True)

    # =========================
    # 1.5 Thảo luận chính sách
    # =========================
    st.subheader("1.5 — Thảo luận chính sách")

    st.markdown("""
    **a) Xu hướng TFP (2020-2025):**  
    TFP Việt Nam có xu hướng **tăng nhẹ/ổn định**, cho thấy chất lượng tăng trưởng dựa vào yếu tố năng suất tổng hợp đang cải thiện.

    **b) Yếu tố đóng góp nhiều nhất:**  
    Trong các yếu tố mới D (cơ sở hạ tầng số), AI (doanh nghiệp số), H (nhân lực kỹ năng cao), thường **AI và D đóng góp nhiều nhất**, do ảnh hưởng trực tiếp tới năng suất và sản xuất GDP.  

    **c) Mục tiêu 30% kinh tế số/GDP vào 2030:**  
    Dựa trên mô hình, mục tiêu này **khả thi nếu K và L tăng trưởng đều, TFP cải thiện và AI/D đạt kế hoạch**. Cần ràng buộc: chính sách thúc đẩy đầu tư số, đào tạo nhân lực, cải thiện hạ tầng, và duy trì tăng trưởng vốn và lao động.
    """)

    st.markdown("✅ **Kết thúc Bài 1**")

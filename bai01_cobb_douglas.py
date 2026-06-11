import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def render():
    st.title("🌱 Bài 1 — Hàm sản xuất Cobb-Douglas mở rộng với AI và Số hóa")
    st.markdown("Phân tích đóng góp của vốn, lao động, kinh tế số, AI và nhân lực số vào tăng trưởng GDP Việt Nam.")

    # 1. DỮ LIỆU ĐẦU VÀO
    df = pd.DataFrame({
        "Năm": [2020, 2021, 2022, 2023, 2024, 2025],
        "Y_GDP": [8044.4, 8487.5, 9513.3, 10221.8, 11511.9, 12847.6],
        "K_von": [16500, 17800, 19600, 21300, 23500, 25900],
        "L_lao_dong": [53.6, 50.5, 51.7, 52.4, 52.9, 53.4],
        "D_kinh_te_so": [12.0, 12.7, 14.3, 16.5, 18.3, 19.5],
        "AI_dn_so": [55.6, 60.2, 65.4, 67.0, 73.8, 80.1],
        "H_nhan_luc_so": [24.1, 26.1, 26.2, 27.0, 28.4, 29.2],
    })

    # 2. THANH BÊN (SIDEBAR) - TƯƠNG TÁC THAM SỐ
    st.sidebar.header("⚙️ Điều chỉnh tham số mô hình")
    st.sidebar.markdown("Điều chỉnh hệ số co giãn của các yếu tố đầu vào. Tổng các hệ số bằng 1 (lợi suất không đổi theo quy mô).")
    
    alpha = st.sidebar.slider("α - Vốn K", 0.10, 0.60, 0.33, 0.01)
    beta = st.sidebar.slider("β - Lao động L", 0.10, 0.70, 0.42, 0.01)
    gamma = st.sidebar.slider("γ - Số hóa D", 0.01, 0.30, 0.10, 0.01)
    delta = st.sidebar.slider("δ - Năng lực AI", 0.01, 0.30, 0.08, 0.01)

    theta = 1 - alpha - beta - gamma - delta

    st.sidebar.metric(label="θ - Nhân lực số H (Tự động tính)", value=f"{theta:.3f}")

    if theta <= 0:
        st.sidebar.error("⚠️ Tổng α + β + γ + δ đang lớn hơn hoặc bằng 1. Vui lòng giảm các tham số để θ > 0.")
        st.error("Dừng tính toán: Ràng buộc tổng hệ số bằng 1 bị vi phạm.")
        st.stop()

    # 3. TÍNH TOÁN CÁC BIẾN SỐ
    # Tính TFP (A_t)
    df["A_TFP"] = df["Y_GDP"] / (
        df["K_von"] ** alpha * 
        df["L_lao_dong"] ** beta * 
        df["D_kinh_te_so"] ** gamma * 
        df["AI_dn_so"] ** delta * 
        df["H_nhan_luc_so"] ** theta
    )

    A_mean = df["A_TFP"].mean()

    # Tính GDP dự báo
    df["Y_du_bao"] = A_mean * (
        df["K_von"] ** alpha * 
        df["L_lao_dong"] ** beta * 
        df["D_kinh_te_so"] ** gamma * 
        df["AI_dn_so"] ** delta * 
        df["H_nhan_luc_so"] ** theta
    )

    df["Sai_so_tuyet_doi"] = abs(df["Y_GDP"] - df["Y_du_bao"])
    df["APE_%"] = (df["Sai_so_tuyet_doi"] / df["Y_GDP"]) * 100
    mape = df["APE_%"].mean()

    # 4. HÀM TÁC NHÂN AI ĐÁNH GIÁ (AI AGENT)
    def ai_agent_analysis(df, mape):
        tfp_growth = (df["A_TFP"].iloc[-1] / df["A_TFP"].iloc[0] - 1) * 100
        
        analysis = f"**🤖 Tác nhân AI phân tích hệ thống:**\n\n"
        analysis += f"- **Độ chính xác mô hình:** Chỉ số MAPE đạt **{mape:.2f}%**. "
        if mape < 5:
            analysis += "Mô hình có độ chính xác xuất sắc, các hệ số bạn chọn phản ánh rất sát thực tế cấu trúc nền kinh tế.\n"
        elif mape < 10:
            analysis += "Mô hình có độ chính xác tốt, tuy nhiên có thể tinh chỉnh thêm hệ số để dự báo bám sát hơn.\n"
        else:
            analysis += "Mức sai số khá cao. Rủi ro overfitting hoặc hệ số đầu vào chưa phù hợp với giai đoạn này.\n"
            
        analysis += f"- **Chất lượng tăng trưởng:** TFP giai đoạn 2020-2025 thay đổi **{tfp_growth:.2f}%**. "
        if tfp_growth > 0:
            analysis += "Nền kinh tế đang tăng trưởng dựa nhiều vào đổi mới sáng tạo, chuyển đổi số và AI thay vì chỉ mở rộng vốn/lao động thô.\n"
        else:
            analysis += "Sự sụt giảm TFP cho thấy hiệu quả sử dụng công nghệ và vốn số hóa chưa được tối ưu, tăng trưởng vẫn phụ thuộc thâm dụng vốn.\n"
            
        return analysis

    # 5. GIAO DIỆN TABS
    tab1, tab2, tab3 = st.tabs([
        "📊 Dữ liệu & TFP", 
        "📈 Biểu đồ & Phân tích 3D", 
        "🔮 Dự báo Kịch bản 2030"
    ])

    with tab1:
        st.subheader("1. Dữ liệu thực tế & Ước lượng năng suất nhân tố tổng hợp (TFP)")
        st.latex(r"A_t = \frac{Y_t}{K_t^\alpha L_t^\beta D_t^\gamma AI_t^\delta H_t^\theta}")
        
        # Hiển thị bảng đẹp hơn với highlight
        st.dataframe(df.style.highlight_max(subset=["Y_GDP", "A_TFP"], color="lightgreen"), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            fig_tfp = px.line(df, x="Năm", y="A_TFP", markers=True, title="Xu hướng TFP (A_t) qua các năm", 
                              line_shape="spline", color_discrete_sequence=["#17BECF"])
            st.plotly_chart(fig_tfp, use_container_width=True)
        with col2:
            st.info(ai_agent_analysis(df, mape))

    with tab2:
        st.subheader("2. So sánh thực tế vs Dự báo & Mô phỏng không gian 3D")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("MAPE (Sai số dự báo)", f"{mape:.2f}%", delta="Tốt" if mape < 5 else "Cần tinh chỉnh", delta_color="inverse")
        col2.metric("TFP Trung bình (A_mean)", f"{A_mean:.4f}")
        col3.metric("GDP Dự báo 2025", f"{df['Y_du_bao'].iloc[-1]:,.1f}", 
                    delta=f"{df['Y_du_bao'].iloc[-1] - df['Y_GDP'].iloc[-1]:.1f} lệch so với thực tế")

        # Biểu đồ 2D So sánh GDP
        fig2d = go.Figure()
        fig2d.add_trace(go.Scatter(x=df["Năm"], y=df["Y_GDP"], mode="lines+markers", name="GDP Thực tế", line=dict(color="blue", width=3)))
        fig2d.add_trace(go.Scatter(x=df["Năm"], y=df["Y_du_bao"], mode="lines+markers", name="GDP Dự báo", line=dict(color="red", dash="dash", width=3)))
        fig2d.update_layout(title="Khớp chuẩn mô hình: GDP Thực tế vs Dự báo", hovermode="x unified")
        st.plotly_chart(fig2d, use_container_width=True)

        st.markdown("---")
        st.subheader("🌐 Biểu đồ 3D: Tác động của AI và Kinh tế số lên GDP")
        st.markdown("*Giữ cố định Vốn (K), Lao động (L) và Nhân lực số (H) ở mức năm 2025 để quan sát bề mặt sản lượng.*")
        
        # Khởi tạo không gian lưới cho đồ thị 3D
        ai_range = np.linspace(df["AI_dn_so"].min() * 0.8, df["AI_dn_so"].max() * 1.5, 20)
        d_range = np.linspace(df["D_kinh_te_so"].min() * 0.8, df["D_kinh_te_so"].max() * 1.5, 20)
        ai_grid, d_grid = np.meshgrid(ai_range, d_range)
        
        # Cố định các biến khác tại năm 2025
        k_2025, l_2025, h_2025 = df["K_von"].iloc[-1], df["L_lao_dong"].iloc[-1], df["H_nhan_luc_so"].iloc[-1]
        
        # Tính Y cho không gian 3D
        Z_gdp = A_mean * (k_2025**alpha) * (l_2025**beta) * (d_grid**gamma) * (ai_grid**delta) * (h_2025**theta)
        
        fig3d = go.Figure(data=[go.Surface(z=Z_gdp, x=ai_grid, y=d_grid, colorscale="Viridis")])
        fig3d.update_layout(
            title="Bề mặt GDP theo Năng lực AI và Chỉ số Kinh tế số",
            scene=dict(
                xaxis_title='AI (Nghìn DN)',
                yaxis_title='Kinh tế số (%)',
                zaxis_title='GDP Dự báo'
            ),
            height=600,
            margin=dict(l=0, r=0, b=0, t=40)
        )
        st.plotly_chart(fig3d, use_container_width=True)

    with tab3:
        st.subheader("3. Dự báo quỹ đạo kinh tế đến 2030")
        st.markdown("Tác nhân AI đã thiết lập các mốc mục tiêu dự kiến. Bạn có thể tương tác trực tiếp để mô phỏng kịch bản.")

        # Tương tác input nằm ngang
        c1, c2, c3 = st.columns(3)
        target_D = c1.number_input("🎯 Mục tiêu Kinh tế số/GDP 2030 (%)", value=30.0, step=1.0)
        target_AI = c2.number_input("🎯 Mục tiêu Số DN AI 2030 (Nghìn)", value=100.0, step=5.0)
        target_H = c3.number_input("🎯 Mục tiêu Lao động qua ĐT 2030 (%)", value=35.0, step=1.0)

        c4, c5, c6 = st.columns(3)
        growth_K = c4.number_input("📈 Tăng trưởng Vốn (K) %/năm", value=6.0, step=0.5) / 100
        growth_L = c5.number_input("📈 Tăng trưởng LĐ (L) %/năm", value=6.0, step=0.5) / 100
        growth_A = c6.number_input("📈 Tăng trưởng TFP (A) %/năm", value=1.2, step=0.1) / 100

        base = df.iloc[-1]
        rows = []

        for year in range(2025, 2031):
            step = year - 2025
            K_pred = base["K_von"] * ((1 + growth_K) ** step)
            L_pred = base["L_lao_dong"] * ((1 + growth_L) ** step)
            A_pred = base["A_TFP"] * ((1 + growth_A) ** step)
            D_pred = base["D_kinh_te_so"] + (target_D - base["D_kinh_te_so"]) * (step / 5)
            AI_pred = base["AI_dn_so"] + (target_AI - base["AI_dn_so"]) * (step / 5)
            H_pred = base["H_nhan_luc_so"] + (target_H - base["H_nhan_luc_so"]) * (step / 5)

            Y_hat = A_pred * (K_pred**alpha * L_pred**beta * D_pred**gamma * AI_pred**delta * H_pred**theta)

            rows.append({
                "Năm": year, "A_TFP": A_pred, "K": K_pred, "L": L_pred, 
                "D": D_pred, "AI": AI_pred, "H": H_pred, "GDP_dự_báo": Y_hat,
                "Tăng_so_với_2025_%": (Y_hat / base["Y_GDP"] - 1) * 100
            })

        forecast = pd.DataFrame(rows)
        gdp_2030 = forecast[forecast["Năm"] == 2030]["GDP_dự_báo"].iloc[0]
        growth_2030 = forecast[forecast["Năm"] == 2030]["Tăng_so_với_2025_%"].iloc[0]

        # AI Agent nhận định kịch bản
        st.success(f"🤖 **AI Agent dự phóng kịch bản:** Với thiết lập của bạn, quy mô GDP năm 2030 sẽ đạt **{gdp_2030:,.1f} nghìn tỷ VND**, tăng **{growth_2030:.2f}%** so với 2025. Việc đạt mục tiêu Kinh tế số {target_D}% sẽ là bệ phóng quan trọng nhất.")

        fig_forecast = px.bar(forecast, x="Năm", y="GDP_dự_báo", text_auto='.2s', 
                              title="Quy mô GDP mô phỏng 2025 - 2030", 
                              color="GDP_dự_báo", color_continuous_scale="Blues")
        fig_forecast.update_traces(textfont_size=12, textangle=0, textposition="outside")
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        with st.expander("📄 Xem chi tiết bảng dữ liệu dự báo"):
            st.dataframe(forecast.round(2), use_container_width=True)

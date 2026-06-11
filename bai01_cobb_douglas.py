import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# =========================== AI ANALYST ===========================
def ai_analysis(df, alpha, beta, gamma, delta, theta, mape, growth_decomposition, gdp_2030, A_forecast):
    """Trả về phân tích tự động dựa trên kết quả tính toán"""
    insights = []
    
    # 1. Đánh giá TFP
    tfp_trend = df["A_TFP"].iloc[-1] - df["A_TFP"].iloc[0]
    if tfp_trend > 0:
        insights.append(f"📈 **Chất lượng tăng trưởng:** TFP có xu hướng TĂNG ({df['A_TFP'].iloc[0]:.4f} → {df['A_TFP'].iloc[-1]:.4f}). Tăng trưởng đang dần bớt phụ thuộc vào việc bơm vốn/lao động thô, bắt đầu có dấu hiệu chuyển dịch nhờ đổi mới công nghệ.")
    else:
        insights.append(f"📉 **Chất lượng tăng trưởng:** TFP có xu hướng GIẢM. Mô hình cảnh báo tăng trưởng GDP hiện tại chủ yếu do thâm dụng vốn và lao động – tiềm ẩn rủi ro thiếu bền vững trong dài hạn.")
    
    # 2. Đánh giá Mô hình (MAPE)
    if mape < 5:
        insights.append(f"✅ **Độ tin cậy mô hình:** Rất xuất sắc (MAPE = {mape:.2f}%). Hàm Cobb-Douglas mở rộng mô phỏng gần như hoàn hảo quỹ đạo thực tế của kinh tế Việt Nam.")
    elif mape < 10:
        insights.append(f"⚠️ **Độ tin cậy mô hình:** Chấp nhận được (MAPE = {mape:.2f}%). Tuy nhiên, có thể tinh chỉnh thêm hệ số hoặc xét đến độ trễ (lag) của các chính sách đầu tư số.")
    else:
        insights.append(f"❌ **Độ tin cậy mô hình:** Sai số khá cao (MAPE = {mape:.2f}%). Cần đánh giá lại hệ số co giãn do vi phạm giả định về lợi suất không đổi.")
    
    # 3. Phân rã tăng trưởng
    contrib = growth_decomposition.set_index("Thành phần")["Đóng góp (pp/ năm)"]
    top = contrib.nlargest(2)
    insights.append(f"🔑 **Động lực cốt lõi (2020-2025):** Yếu tố đóng góp lớn nhất là **{top.index[0]}** ({top.values[0]:.2f} điểm %/năm), tiếp đến là **{top.index[1]}** ({top.values[1]:.2f} điểm %/năm).")
    
    if "AI" in top.index[:2] or "Số hóa D" in top.index[:2]:
        insights.append("🚀 **Khuyến nghị:** Trụ cột Kinh tế số và AI đang phát huy tác dụng rõ rệt. Kết quả này củng cố tính đúng đắn của Nghị quyết 57-NQ/TW về đột phá công nghệ.")
    else:
        insights.append("🏗️ **Khuyến nghị:** Tăng trưởng vẫn dậm chân ở động lực truyền thống. Để đạt mục tiêu Quyết định 749/QĐ-TTg, Việt Nam cần đẩy nhanh tốc độ lan tỏa của AI và số hóa vào các ngành sản xuất thực.")
    
    return "\n\n".join(insights)

# =========================== HÀM CHÍNH ===========================
def render():
    st.title("🌱 Bài 1 — Hàm sản xuất Cobb-Douglas mở rộng với AI và Số hóa")
    st.markdown("Phân tích đóng góp của vốn, lao động, kinh tế số, AI và nhân lực số vào tăng trưởng GDP Việt Nam dựa trên dữ liệu 2020-2025.")

    # ---------- KHỞI TẠO STATE CHO NÚT RESET ----------
    default_params = {"alpha": 0.33, "beta": 0.42, "gamma": 0.10, "delta": 0.08}
    for key, val in default_params.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # ---------- DỮ LIỆU GỐC ----------
    df_raw = pd.DataFrame({
        "Năm": [2020, 2021, 2022, 2023, 2024, 2025],
        "Y_GDP": [8044.4, 8487.5, 9513.3, 10221.8, 11511.9, 12847.6],
        "K_von": [16500, 17800, 19600, 21300, 23500, 25900],
        "L_lao_dong": [53.6, 50.5, 51.7, 52.4, 52.9, 53.4],
        "D_kinh_te_so": [12.0, 12.7, 14.3, 16.5, 18.3, 19.5],
        "AI_dn_so": [55.6, 60.2, 65.4, 67.0, 73.8, 80.1],
        "H_nhan_luc_so": [24.1, 26.1, 26.2, 27.0, 28.4, 29.2],
    })

    # ---------- SIDEBAR: ĐIỀU CHỈNH HỆ SỐ ----------
    st.sidebar.markdown("## ⚙️ Hệ số Co giãn (Elasticity)")
    
    alpha = st.sidebar.slider("α (Vốn K)", 0.10, 0.60, st.session_state["alpha"], 0.01, key="slider_alpha", help="Độ co giãn của GDP theo vốn vật chất")
    beta  = st.sidebar.slider("β (Lao động L)", 0.10, 0.70, st.session_state["beta"], 0.01, key="slider_beta", help="Độ co giãn theo lao động")
    gamma = st.sidebar.slider("γ (Số hóa D)", 0.01, 0.30, st.session_state["gamma"], 0.01, key="slider_gamma", help="Độ co giãn theo tỷ trọng kinh tế số")
    delta = st.sidebar.slider("δ (Năng lực AI)", 0.01, 0.30, st.session_state["delta"], 0.01, key="slider_delta", help="Độ co giãn theo năng lực AI")
    
    theta = 1 - alpha - beta - gamma - delta
    st.sidebar.metric("θ (Nhân lực số H)", f"{theta:.3f}", help="Tự động tính để bảo đảm lợi suất không đổi theo quy mô (tổng = 1)")
    
    if theta <= 0:
        st.sidebar.error("❌ Tổng α+β+γ+δ ≥1 khiến θ ≤ 0. Vui lòng giảm các hệ số.")
        st.error("Dừng tính toán: Ràng buộc quy mô bị vi phạm.")
        return
    
    if st.sidebar.button("🔄 Reset Tham số"):
        st.session_state["alpha"] = 0.33
        st.session_state["beta"] = 0.42
        st.session_state["gamma"] = 0.10
        st.session_state["delta"] = 0.08
        st.rerun()

    # ---------- TÍNH TOÁN TFP, DỰ BÁO ----------
    df = df_raw.copy()
    df["A_TFP"] = df["Y_GDP"] / (
        df["K_von"]**alpha * df["L_lao_dong"]**beta * df["D_kinh_te_so"]**gamma *
        df["AI_dn_so"]**delta * df["H_nhan_luc_so"]**theta
    )
    
    use_avg_A = st.sidebar.checkbox("Dùng TFP trung bình để dự báo", value=True)
    A_forecast = df["A_TFP"].mean() if use_avg_A else df["A_TFP"].iloc[-1]
    st.sidebar.info(f"TFP cơ sở (Baseline A) = {A_forecast:.4f}")
    
    df["Y_du_bao"] = A_forecast * (
        df["K_von"]**alpha * df["L_lao_dong"]**beta * df["D_kinh_te_so"]**gamma *
        df["AI_dn_so"]**delta * df["H_nhan_luc_so"]**theta
    )
    df["Sai_so_%"] = abs(df["Y_GDP"] - df["Y_du_bao"]) / df["Y_GDP"] * 100
    mape = df["Sai_so_%"].mean()

    # ---------- PHÂN RÃ TĂNG TRƯỞNG (Log-difference) ----------
    log_vars = {
        "Y": np.log(df["Y_GDP"].iloc[-1] / df["Y_GDP"].iloc[0]),
        "K": np.log(df["K_von"].iloc[-1] / df["K_von"].iloc[0]),
        "L": np.log(df["L_lao_dong"].iloc[-1] / df["L_lao_dong"].iloc[0]),
        "D": np.log(df["D_kinh_te_so"].iloc[-1] / df["D_kinh_te_so"].iloc[0]),
        "AI": np.log(df["AI_dn_so"].iloc[-1] / df["AI_dn_so"].iloc[0]),
        "H": np.log(df["H_nhan_luc_so"].iloc[-1] / df["H_nhan_luc_so"].iloc[0]),
    }
    
    contrib_TFP = log_vars["Y"] - (alpha*log_vars["K"] + beta*log_vars["L"] + gamma*log_vars["D"] + delta*log_vars["AI"] + theta*log_vars["H"])
    n_years = 5  # 2020-2025
    
    growth_decomp = pd.DataFrame({
        "Thành phần": ["Vốn (K)", "Lao động (L)", "Số hóa (D)", "AI", "Nhân lực số (H)", "TFP (Năng suất)"],
        "Đóng góp (pp/ năm)": [
            (alpha*log_vars["K"])/n_years*100, 
            (beta*log_vars["L"])/n_years*100,
            (gamma*log_vars["D"])/n_years*100, 
            (delta*log_vars["AI"])/n_years*100,
            (theta*log_vars["H"])/n_years*100, 
            contrib_TFP/n_years*100
        ]
    })

    # ---------- DỰ BÁO KỊCH BẢN 2030 ----------
    st.sidebar.markdown("## 🔮 Giả định Kịch bản 2030")
    target_D = st.sidebar.number_input("Mục tiêu Kinh tế số (%)", 20.0, 50.0, 30.0)
    target_AI = st.sidebar.number_input("Mục tiêu AI (Nghìn DN)", 80.0, 200.0, 100.0)
    target_H = st.sidebar.number_input("Mục tiêu Nhân lực số (%)", 25.0, 50.0, 35.0)
    growth_K = st.sidebar.number_input("Tăng trưởng Vốn (%/năm)", 2.0, 12.0, 6.0, 0.5) / 100
    growth_L = st.sidebar.number_input("Tăng trưởng LĐ (%/năm)", 0.0, 8.0, 0.6, 0.1) / 100
    growth_A = st.sidebar.number_input("Tăng trưởng TFP (%/năm)", 0.0, 3.0, 1.2, 0.1) / 100

    base = df.iloc[-1]
    rows = []
    for step, year in enumerate(range(2025, 2031)):
        K = base["K_von"] * (1 + growth_K)**step
        L = base["L_lao_dong"] * (1 + growth_L)**step
        A_t = base["A_TFP"] * (1 + growth_A)**step
        D = base["D_kinh_te_so"] + (target_D - base["D_kinh_te_so"]) * step/5
        AI = base["AI_dn_so"] + (target_AI - base["AI_dn_so"]) * step/5
        H = base["H_nhan_luc_so"] + (target_H - base["H_nhan_luc_so"]) * step/5
        
        Y_hat = A_t * (K**alpha * L**beta * D**gamma * AI**delta * H**theta)
        rows.append({"Năm": year, "GDP_dự_báo": Y_hat, "A_TFP": A_t, "K": K, "L": L, "D": D, "AI": AI, "H": H})
        
    forecast = pd.DataFrame(rows)
    gdp_2030 = forecast[forecast["Năm"]==2030]["GDP_dự_báo"].values[0]

    # =========================== GIAO DIỆN TABS ===========================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dữ liệu & Đánh giá", 
        "📐 Phân rã Tăng trưởng", 
        "🌐 Mô phỏng Không gian 3D",
        "🔮 Quỹ đạo 2030"
    ])
    
    with tab1:
        st.subheader("1. Kiểm định Mô hình & TFP")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("MAPE (Độ lệch)", f"{mape:.2f}%", delta="Tốt" if mape < 5 else "Cần rà soát", delta_color="inverse")
        c2.metric("TFP Trung bình", f"{df['A_TFP'].mean():.4f}")
        c3.metric("GDP Dự báo 2025", f"{df['Y_du_bao'].iloc[-1]:,.1f}", delta=f"Thực tế: {df['Y_GDP'].iloc[-1]:,.1f}")

        # AI Agent Box
        with st.expander("🤖 Đọc phân tích từ AI Agent", expanded=True):
            st.markdown(ai_analysis(df, alpha, beta, gamma, delta, theta, mape, growth_decomp, gdp_2030, A_forecast))

        col_a, col_b = st.columns(2)
        with col_a:
            fig_tfp = px.line(df, x="Năm", y="A_TFP", markers=True, title="Xu hướng TFP (A_t)", line_shape="spline")
            st.plotly_chart(fig_tfp, use_container_width=True)
        with col_b:
            fig_compare = go.Figure()
            fig_compare.add_trace(go.Scatter(x=df["Năm"], y=df["Y_GDP"], mode="lines+markers", name="Thực tế", line=dict(width=3)))
            fig_compare.add_trace(go.Scatter(x=df["Năm"], y=df["Y_du_bao"], mode="lines+markers", name="Dự báo", line=dict(dash="dash", width=3)))
            fig_compare.update_layout(title="Mức độ khớp chuẩn (Thực tế vs Dự báo)", hovermode="x unified")
            st.plotly_chart(fig_compare, use_container_width=True)
            
    with tab2:
        st.subheader("2. Phân rã Động lực Tăng trưởng (Growth Accounting)")
        st.markdown("Sử dụng phương pháp **sai phân Logarit** để bóc tách phần đóng góp của từng yếu tố vào mức tăng trưởng trung bình hàng năm giai đoạn 2020-2025.")
        
        col_c, col_d = st.columns([1, 2])
        with col_c:
            st.dataframe(growth_decomp.style.background_gradient(cmap="Blues", subset=["Đóng góp (pp/ năm)"]).format({"Đóng góp (pp/ năm)": "{:.2f}"}), use_container_width=True)
        with col_d:
            fig_bar = px.bar(growth_decomp, x="Đóng góp (pp/ năm)", y="Thành phần", orientation='h',
                             title="Cấu trúc đóng góp tăng trưởng (Điểm % / năm)", color="Thành phần", text_auto=".2f")
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

    with tab3:
        st.subheader("3. Bề mặt Sản lượng 3D (Production Surface)")
        st.markdown("*Giữ cố định Vốn (K), Lao động (L) và Nhân lực số (H) ở mức năm 2025. Bề mặt mô phỏng sự gia tăng GDP nếu ta đẩy mạnh AI và chuyển đổi số.*")
        
        # Khởi tạo không gian lưới cho đồ thị 3D Surface
        ai_range = np.linspace(df["AI_dn_so"].min() * 0.8, df["AI_dn_so"].max() * 1.5, 30)
        d_range = np.linspace(df["D_kinh_te_so"].min() * 0.8, df["D_kinh_te_so"].max() * 1.5, 30)
        ai_grid, d_grid = np.meshgrid(ai_range, d_range)
        
        k_2025, l_2025, h_2025 = base["K_von"], base["L_lao_dong"], base["H_nhan_luc_so"]
        Z_gdp = A_forecast * (k_2025**alpha) * (l_2025**beta) * (d_grid**gamma) * (ai_grid**delta) * (h_2025**theta)
        
        fig3d = go.Figure(data=[go.Surface(z=Z_gdp, x=ai_grid, y=d_grid, colorscale="Viridis")])
        fig3d.update_layout(
            title="Bề mặt GDP theo Năng lực AI và Kinh tế số",
            scene=dict(xaxis_title='AI (Nghìn DN)', yaxis_title='Kinh tế số (%)', zaxis_title='GDP Mô phỏng'),
            height=600, margin=dict(l=0, r=0, b=0, t=40)
        )
        st.plotly_chart(fig3d, use_container_width=True)

    with tab4:
        st.subheader("4. Quỹ đạo Kinh tế Kỷ nguyên Số (2025 - 2030)")
        
        growth_25_30 = (gdp_2030 / base['Y_GDP'] - 1) * 100
        st.success(f"🎯 **Tổng kết Kịch bản:** Theo thiết lập của bạn tại thanh bên, GDP Việt Nam năm 2030 dự kiến đạt **{gdp_2030:,.1f} nghìn tỷ VND** (tăng **{growth_25_30:.1f}%** so với 2025).")
        
        fig_fore = px.bar(forecast, x="Năm", y="GDP_dự_báo", text_auto='.2s', 
                          title="Quy mô GDP dự báo", color="GDP_dự_báo", color_continuous_scale="Teal")
        fig_fore.update_traces(textposition="outside")
        st.plotly_chart(fig_fore, use_container_width=True)
        
        with st.expander("📄 Xem bảng chi tiết dữ liệu 2025-2030"):
            st.dataframe(forecast.round(2), use_container_width=True)

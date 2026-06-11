# bai01_cobb_douglas.py
# Bài 1 - Hàm sản xuất Cobb-Douglas mở rộng với AI và số hóa
# Phiên bản sửa lỗi cú pháp f-string

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================== AI ANALYST ===========================
def ai_analysis(df, alpha, beta, gamma, delta, theta, mape, growth_decomposition, gdp_2030, A_mean):
    """Trả về phân tích bằng tiếng Việt dựa trên kết quả tính toán"""
    insights = []
    
    # Xu hướng TFP
    tfp_trend = df["A_TFP"].iloc[-1] - df["A_TFP"].iloc[0]
    if tfp_trend > 0:
        insights.append(f"📈 **TFP có xu hướng tăng** ({df['A_TFP'].iloc[0]:.4f} → {df['A_TFP'].iloc[-1]:.4f}), cho thấy chất lượng tăng trưởng được cải thiện nhờ đổi mới công nghệ và hiệu quả đầu tư.")
    else:
        insights.append(f"📉 **TFP có xu hướng giảm**, tăng trưởng GDP phụ thuộc nhiều vào mở rộng đầu vào (vốn, lao động) – tiềm ẩn rủi ro thiếu bền vững.")
    
    # MAPE
    if mape < 5:
        insights.append(f"✅ **Mô hình dự báo rất tốt** (MAPE = {mape:.2f}%) – hàm Cobb-Douglas mở rộng giải thích được gần như biến động GDP thực tế.")
    elif mape < 10:
        insights.append(f"⚠️ **Mô hình dự báo chấp nhận được** (MAPE = {mape:.2f}%) – cần bổ sung thêm biến như vốn xã hội hoặc tác động lan tỏa.")
    else:
        insights.append(f"❌ **Mô hình chưa phù hợp** (MAPE = {mape:.2f}%) – xem xét lại hệ số co giãn hoặc cấu trúc hàm sản xuất.")
    
    # Đóng góp tăng trưởng
    contrib = growth_decomposition.set_index("Thành phần")["Đóng góp (pp/ năm)"]
    top = contrib.nlargest(2)
    insights.append(f"🔑 **Yếu tố đóng góp nhiều nhất** là **{top.index[0]}** ({top.values[0]:.2f} điểm phần trăm/năm), tiếp đến là **{top.index[1]}** ({top.values[1]:.2f} pp/năm).")
    
    if "AI" in top.index[:2] or "Số hóa D" in top.index[:2]:
        insights.append("🚀 Kinh tế số và AI đang trở thành động lực tăng trưởng chính – phù hợp với Nghị quyết 57-NQ/TW.")
    else:
        insights.append("🏗️ Tăng trưởng vẫn dựa vào vốn vật chất và lao động – cần đẩy nhanh chuyển đổi số.")
    
    # Dự báo 2030
    growth_2025_2030 = (gdp_2030 / df[df["Năm"]==2025]["Y_GDP"].values[0] - 1) * 100
    # Dòng bị lỗi đã được sửa: không có dấu ngoặc đơn thừa
    insights.append(f"📅 **Dự báo 2030**: GDP đạt {gdp_2030:,.0f} nghìn tỷ, tăng {growth_2025_2030:.1f}% so với 2025. {'Khả thi nếu D đạt 30% và AI 100 nghìn DN.' if growth_2025_2030 > 50 else 'Cần kịch bản đầu tư cao hơn để đạt mục tiêu tăng trưởng kép.'}")
    
    # So sánh với mục tiêu chính sách
    if df["D_kinh_te_so"].iloc[-1] < 20:
        insights.append("📡 **Kinh tế số/GDP** hiện dưới 20% – theo Quyết định 749, cần tăng tốc đầu tư hạ tầng số và hỗ trợ doanh nghiệp chuyển đổi.")
    
    return "\n\n".join(insights)

# =========================== HÀM CHÍNH ===========================
def render():
    st.set_page_config(page_title="Bài 1 – Cobb-Douglas mở rộng", layout="wide")
    st.title("🌱 Bài 1 — Hàm sản xuất Cobb-Douglas mở rộng với AI và số hóa")
    st.markdown("*Dữ liệu Việt Nam 2020–2025 | Tương tác thay đổi hệ số → cập nhật ngay kết quả*")

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
    st.sidebar.markdown("## ⚙️ Tham số co giãn")
    alpha = st.sidebar.slider("α (Vốn K)", 0.10, 0.60, 0.33, 0.01, help="Độ co giãn của GDP theo vốn vật chất")
    beta  = st.sidebar.slider("β (Lao động L)", 0.10, 0.70, 0.42, 0.01, help="Độ co giãn theo lao động")
    gamma = st.sidebar.slider("γ (Số hóa D)", 0.01, 0.30, 0.10, 0.01, help="Độ co giãn theo tỷ trọng kinh tế số")
    delta = st.sidebar.slider("δ (AI)", 0.01, 0.30, 0.08, 0.01, help="Độ co giãn theo năng lực AI")
    
    theta = 1 - alpha - beta - gamma - delta
    st.sidebar.metric("θ (Nhân lực số H)", f"{theta:.3f}", help="Được tính tự động để tổng = 1")
    
    if theta <= 0:
        st.error("❌ Tổng α+β+γ+δ ≥1, θ ≤ 0. Giảm các hệ số khác lại.")
        return
    
    # Nút reset (dùng session state để lưu giá trị mặc định)
    if st.sidebar.button("🔄 Reset về hệ số mặc định"):
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
    
    # Lựa chọn cách tính A cho dự báo
    use_avg_A = st.sidebar.checkbox("Dùng A trung bình (2020-2025) để dự báo", value=True)
    if use_avg_A:
        A_forecast = df["A_TFP"].mean()
        st.sidebar.info("Dự báo dùng A trung bình = {:.4f}".format(A_forecast))
    else:
        A_forecast = df["A_TFP"].iloc[-1]  # năm cuối
        st.sidebar.info("Dự báo dùng A năm 2025 = {:.4f}".format(A_forecast))
    
    df["Y_du_bao"] = A_forecast * (
        df["K_von"]**alpha * df["L_lao_dong"]**beta * df["D_kinh_te_so"]**gamma *
        df["AI_dn_so"]**delta * df["H_nhan_luc_so"]**theta
    )
    df["Sai_so_%"] = abs(df["Y_GDP"] - df["Y_du_bao"]) / df["Y_GDP"] * 100
    mape = df["Sai_so_%"].mean()
    
    # ---------- PHÂN RÃ TĂNG TRƯỞNG (câu 1.4.3) ----------
    # Lấy log và tính sai phân năm đầu - năm cuối (từ 2020 đến 2025)
    log_vars = {
        "Y": np.log(df["Y_GDP"].iloc[-1] / df["Y_GDP"].iloc[0]),
        "K": np.log(df["K_von"].iloc[-1] / df["K_von"].iloc[0]),
        "L": np.log(df["L_lao_dong"].iloc[-1] / df["L_lao_dong"].iloc[0]),
        "D": np.log(df["D_kinh_te_so"].iloc[-1] / df["D_kinh_te_so"].iloc[0]),
        "AI": np.log(df["AI_dn_so"].iloc[-1] / df["AI_dn_so"].iloc[0]),
        "H": np.log(df["H_nhan_luc_so"].iloc[-1] / df["H_nhan_luc_so"].iloc[0]),
    }
    # TFP đóng góp = dlnY - (α dlnK + β dlnL + ...)
    contrib_TFP = log_vars["Y"] - (alpha*log_vars["K"] + beta*log_vars["L"] + gamma*log_vars["D"] + 
                                    delta*log_vars["AI"] + theta*log_vars["H"])
    # Đóng góp từng yếu tố (tốc độ tăng bình quân năm = tổng đóng góp / số năm)
    n_years = 5  # 2020->2025 là 5 bước
    growth_decomp = pd.DataFrame({
        "Thành phần": ["Vốn K", "Lao động L", "Số hóa D", "AI", "Nhân lực số H", "TFP (phần dư)"],
        "Công thức": [f"{alpha}×ΔlnK", f"{beta}×ΔlnL", f"{gamma}×ΔlnD", f"{delta}×ΔlnAI", f"{theta}×ΔlnH", "ΔlnY - tổng"],
        "Tổng đóng góp (ln)": [alpha*log_vars["K"], beta*log_vars["L"], gamma*log_vars["D"], 
                                delta*log_vars["AI"], theta*log_vars["H"], contrib_TFP],
        "Đóng góp (pp/ năm)": [(alpha*log_vars["K"])/n_years*100, (beta*log_vars["L"])/n_years*100,
                                (gamma*log_vars["D"])/n_years*100, (delta*log_vars["AI"])/n_years*100,
                                (theta*log_vars["H"])/n_years*100, (contrib_TFP)/n_years*100]
    })
    
    # ---------- DỰ BÁO 2030 (câu 1.4.4) ----------
    st.sidebar.markdown("## 🔮 Kịch bản 2030")
    target_D = st.sidebar.number_input("D (KTS/GDP, %)", 20.0, 50.0, 30.0)
    target_AI = st.sidebar.number_input("AI (nghìn DN số)", 80.0, 200.0, 100.0)
    target_H = st.sidebar.number_input("H (LĐ qua đào tạo, %)", 25.0, 50.0, 35.0)
    growth_K = st.sidebar.number_input("Tăng trưởng K hàng năm", 0.02, 0.12, 0.06, 0.005)
    growth_L = st.sidebar.number_input("Tăng trưởng L hàng năm", 0.00, 0.08, 0.006, 0.001)
    growth_A = st.sidebar.number_input("Tăng trưởng TFP hàng năm", 0.0, 0.03, 0.012, 0.001)
    
    base = df.iloc[-1]  # năm 2025
    forecast_years = list(range(2025, 2031))
    rows = []
    for step, year in enumerate(forecast_years):
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
    
    # =========================== GIAO DIỆN CHÍNH ===========================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Dữ liệu & TFP", "📊 Dự báo & MAPE", "📐 Phân rã tăng trưởng", "🔮 Dự báo 2030", "✨ 3D & AI Analyst"
    ])
    
    with tab1:
        st.subheader("Dữ liệu Việt Nam 2020-2025 và TFP ước lượng")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(df[["Năm","Y_GDP","K_von","L_lao_dong","D_kinh_te_so","AI_dn_so","H_nhan_luc_so","A_TFP"]].round(3), use_container_width=True)
        with col2:
            fig_tfp = px.line(df, x="Năm", y="A_TFP", markers=True, title="Xu hướng TFP A_t")
            st.plotly_chart(fig_tfp, use_container_width=True)
        
    with tab2:
        st.subheader("So sánh GDP thực tế và dự báo")
        metric_cols = st.columns(3)
        metric_cols[0].metric("A trung bình", f"{A_forecast:.4f}")
        metric_cols[1].metric("MAPE", f"{mape:.2f}%", help="Mean Absolute Percentage Error – càng nhỏ càng tốt")
        metric_cols[2].metric("Sai số tuyệt đối TB", f"{df['Sai_so_%'].mean():.2f}%")
        
        fig_compare = go.Figure()
        fig_compare.add_trace(go.Scatter(x=df["Năm"], y=df["Y_GDP"], mode="lines+markers", name="GDP thực tế"))
        fig_compare.add_trace(go.Scatter(x=df["Năm"], y=df["Y_du_bao"], mode="lines+markers", name="GDP dự báo", line=dict(dash="dash")))
        fig_compare.update_layout(title="GDP thực tế vs dự báo (A trung bình)", xaxis_title="Năm", yaxis_title="nghìn tỷ VND")
        st.plotly_chart(fig_compare, use_container_width=True)
        
    with tab3:
        st.subheader("Đóng góp của từng yếu tố vào tăng trưởng GDP bình quân năm (2020–2025)")
        st.dataframe(growth_decomp[["Thành phần","Đóng góp (pp/ năm)"]].round(3), use_container_width=True)
        fig_bar = px.bar(growth_decomp, x="Thành phần", y="Đóng góp (pp/ năm)", 
                         title="Phân rã tăng trưởng (điểm phần trăm/năm)",
                         color="Thành phần", text_auto=".2f")
        fig_bar.update_layout(yaxis_title="Đóng góp (pp/ năm)", xaxis_title="")
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with tab4:
        st.subheader("Dự báo GDP Việt Nam đến 2030 theo kịch bản tự chọn")
        col1, col2 = st.columns(2)
        col1.metric("GDP 2030", f"{gdp_2030:,.0f} nghìn tỷ VND")
        col2.metric("Tăng so với 2025", f"{(gdp_2030/base['Y_GDP']-1)*100:.1f}%")
        st.dataframe(forecast.round(1), use_container_width=True)
        fig_fore = px.line(forecast, x="Năm", y="GDP_dự_báo", markers=True, title="Quỹ đạo GDP 2025-2030")
        st.plotly_chart(fig_fore, use_container_width=True)
        
    with tab5:
        st.subheader("🎯 Biểu đồ 3D tương tác & Phân tích AI")
        # 3D scatter plot: GDP vs D vs AI vs H (chọn 2 trong 3)
        col3d, col_selector = st.columns([3, 1])
        with col_selector:
            x_3d = st.selectbox("Trục X", ["D_kinh_te_so", "AI_dn_so", "H_nhan_luc_so"], index=0)
            y_3d = st.selectbox("Trục Y", ["AI_dn_so", "H_nhan_luc_so", "K_von"], index=1)
            color_dim = st.selectbox("Màu sắc theo", ["Y_GDP", "A_TFP", "Năm"], index=0)
        fig3d = px.scatter_3d(df, x=x_3d, y=y_3d, z="Y_GDP", color=color_dim, 
                              size_max=10, opacity=0.7, hover_name="Năm",
                              title=f"Tương quan 3D: {x_3d} – {y_3d} – GDP")
        st.plotly_chart(fig3d, use_container_width=True)
        
        # AI phân tích
        st.markdown("---")
        st.subheader("🤖 Nhận định từ AI phân tích chính sách")
        with st.spinner("AI đang đọc kết quả..."):
            analysis_text = ai_analysis(df, alpha, beta, gamma, delta, theta, mape, growth_decomp, gdp_2030, A_forecast)
        st.info(analysis_text)
        
        # Thêm khung thảo luận
        st.markdown("**💬 Thảo luận chính sách gợi ý:**")
        st.markdown("""
        - TFP có xu hướng tăng/giảm? Chất lượng tăng trưởng thế nào?
        - Yếu tố D, AI, H đóng góp nhiều nhất? Tại sao?
        - Mục tiêu 30% kinh tế số vào 2030 – theo mô hình có khả thi không? Cần ràng buộc gì?
        """)

# =========================== MAIN ===========================
def run():
    render()

if __name__ == "__main__":
    render()

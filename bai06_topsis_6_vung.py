# bai06_topsis_6_vung.py
# Bài 6 — TOPSIS 6 vùng
# Module dùng được với streamlit_app.py có cơ chế gọi module.render()

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# =====================================================
# 1. DỮ LIỆU MẶC ĐỊNH THEO ĐỀ
# =====================================================
def default_region_data():
    data = pd.DataFrame({
        "Vùng": [
            "Trung du miền núi phía Bắc",
            "Đồng bằng sông Hồng",
            "Bắc Trung Bộ + DH Trung Bộ",
            "Tây Nguyên",
            "Đông Nam Bộ",
            "Đồng bằng sông Cửu Long"
        ],
        "GDP2024": [3800, 12500, 5500, 3200, 22000, 4800],
        "Kinh tế số": [0.22, 0.35, 0.25, 0.18, 0.42, 0.20],
        "FDI": [5.8, 12.2, 6.5, 4.0, 15.0, 5.0],
        "Lao động số": [1.2, 4.5, 2.0, 0.8, 5.0, 1.5],
        "AI readiness": [0.35, 0.50, 0.40, 0.30, 0.55, 0.38]
    })
    return data

# =====================================================
# 2. CHUẨN HÓA DỮ LIỆU
# =====================================================
def normalize_df(df, method="minmax"):
    norm_df = df.copy()
    for col in df.columns[1:]:
        if method == "minmax":
            norm_df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        elif method == "zscore":
            norm_df[col] = (df[col] - df[col].mean()) / df[col].std()
    return norm_df

# =====================================================
# 3. TÍNH TOPSIS
# =====================================================
def compute_topsis(df, weights=None):
    n_criteria = df.shape[1] - 1
    if weights is None:
        weights = np.ones(n_criteria) / n_criteria

    norm_df = normalize_df(df)
    norm_values = norm_df.iloc[:, 1:].values
    weighted = norm_values * weights

    # Giải pháp lý tưởng
    ideal_best = weighted.max(axis=0)
    ideal_worst = weighted.min(axis=0)

    d_plus = np.linalg.norm(weighted - ideal_best, axis=1)
    d_minus = np.linalg.norm(weighted - ideal_worst, axis=1)

    c_index = d_minus / (d_plus + d_minus)
    df_result = df.copy()
    df_result["C_i"] = c_index
    df_result["Rank"] = df_result["C_i"].rank(ascending=False, method="min").astype(int)
    return df_result, weighted, ideal_best, ideal_worst, d_plus, d_minus

# =====================================================
# 4. GIAO DIỆN STREAMLIT
# =====================================================
def render():
    st.markdown(
        """
        <div style="
            background: rgba(15, 23, 42, 0.86);
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 22px;
            padding: 24px;
            margin-bottom: 18px;
            box-shadow: 0 10px 35px rgba(0,0,0,0.28);
        ">
            <h1>🏆 Bài 6 — TOPSIS 6 vùng</h1>
            <p>
            Module tính xếp hạng 6 vùng Việt Nam theo TOPSIS dựa trên GDP, kinh tế số, FDI, lao động số, AI readiness.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    df = default_region_data()

    st.info("Dữ liệu mặc định 6 vùng với 5 tiêu chí. Có thể upload CSV nếu muốn.")
    uploaded_file = st.file_uploader("Upload CSV 6 vùng", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.warning(f"Không đọc được file. Dữ liệu mặc định được dùng. Lỗi: {e}")

    st.markdown("### Dữ liệu vùng")
    st.dataframe(df, use_container_width=True)

    st.markdown("### Chọn trọng số TOPSIS (mặc định đều nhau)")
    weights = []
    for col in df.columns[1:]:
        w = st.number_input(f"Trọng số {col}", value=1.0)
        weights.append(w)
    weights = np.array(weights) / sum(weights)

    df_result, weighted, ideal_best, ideal_worst, d_plus, d_minus = compute_topsis(df, weights)

    st.markdown("### TOPSIS — Kết quả xếp hạng")
    st.dataframe(df_result.round(4).sort_values("Rank"), use_container_width=True)

    st.markdown("### Biểu đồ C_i theo vùng")
    fig = px.bar(df_result, x="Vùng", y="C_i", text="Rank", title="Chỉ số TOPSIS C_i theo vùng")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Heatmap khoảng cách D+, D-")
    heat_df = pd.DataFrame({
        "Vùng": df_result["Vùng"],
        "D+": d_plus,
        "D-": d_minus
    })
    fig_heat = px.imshow(
        heat_df.set_index("Vùng").T,
        text_auto=".2f",
        aspect="auto",
        title="Khoảng cách D+, D- theo vùng",
        labels=dict(x="Vùng", y="Khoảng cách", color="Giá trị")
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# Alias để streamlit_app.py gọi module.run()
def run():
    render()

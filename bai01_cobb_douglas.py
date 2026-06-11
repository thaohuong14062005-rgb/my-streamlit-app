# bai01_cobb_douglas.py


import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# BÀI 1 — HÀM SẢN XUẤT COBB-DOUGLAS MỞ RỘNG
# Hiển thị từ 1.4.1 đến 1.5, không dùng mô hình 3D
# =========================================================


def get_default_data():
    return pd.DataFrame(
        {
            "Year": [2020, 2021, 2022, 2023, 2024, 2025],
            "GDP_trillion_VND": [8044.4, 8487.5, 9513.3, 10221.8, 11511.9, 12847.6],
            "K": [16500, 17800, 19600, 21300, 23500, 25900],
            "L": [53.6, 50.5, 51.7, 52.4, 52.9, 53.4],
            "D": [12.0, 12.7, 14.3, 16.5, 18.3, 19.5],
            "AI": [55.6, 60.2, 65.4, 67.0, 73.8, 80.1],
            "H": [24.1, 26.1, 26.2, 27.0, 28.4, 29.2],
        }
    )


def parse_vietnamese_number(x):
    """
    Chuyển số kiểu Việt Nam:
    '8.044,4' -> 8044.4
    '16.500' -> 16500
    '12,7' -> 12.7
    """
    if pd.isna(x):
        return np.nan

    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    text = str(x).strip()

    if text == "":
        return np.nan

    if "." in text and "," in text:
        text = text.replace(".", "").replace(",", ".")
    elif "," in text:
        text = text.replace(",", ".")
    elif "." in text:
        parts = text.split(".")
        if len(parts) > 1 and len(parts[-1]) == 3:
            text = text.replace(".", "")

    try:
        return float(text)
    except ValueError:
        return np.nan


def load_data():
    """
    Sử dụng trực tiếp bảng dữ liệu tổng hợp Việt Nam 2020-2025 theo đề bài.
    Không đọc file CSV bên ngoài để tránh lỗi thiếu file trên Streamlit/GitHub.
    """
    return get_default_data()

    possible_paths = [
        "vietnam_macro_2020_2025.csv",
        "./vietnam_macro_2020_2025.csv",
        "data/vietnam_macro_2020_2025.csv",
        "./data/vietnam_macro_2020_2025.csv",
    ]

    csv_path = None
    for path in possible_paths:
        if os.path.exists(path):
            csv_path = path
            break

    if csv_path is None:
        return default_df, "default", None

    try:
        df = pd.read_csv(csv_path)

        rename_map = {
            "year": "Year",
            "Year": "Year",
            "GDP": "GDP_trillion_VND",
            "Y": "GDP_trillion_VND",
            "GDP_trillion_VND": "GDP_trillion_VND",
            "GDP_nghin_ty_VND": "GDP_trillion_VND",
            "K": "K",
            "Capital": "K",
            "L": "L",
            "Labor": "L",
            "D": "D",
            "Digital": "D",
            "AI": "AI",
            "H": "H",
            "HumanCapital": "H",
        }

        df = df.rename(columns={col: rename_map.get(col, col) for col in df.columns})

        required_cols = ["Year", "GDP_trillion_VND", "K", "L", "D", "AI", "H"]

        for col in required_cols:
            if col not in df.columns:
                if len(df) == len(default_df):
                    df[col] = default_df[col].values
                else:
                    df[col] = np.nan

        df = df[required_cols].copy()

        for col in required_cols:
            df[col] = df[col].apply(parse_vietnamese_number)

        df = df.dropna().sort_values("Year").reset_index(drop=True)

        if len(df) < 2:
            return default_df, "default_error", csv_path

        df["Year"] = df["Year"].astype(int)

        return df, "csv", csv_path

    except Exception:
        return default_df, "default_error", csv_path


def compute_model(df, alpha, beta, gamma, delta, theta):
    model_df = df.copy()

    Y = model_df["GDP_trillion_VND"].astype(float).values
    K = model_df["K"].astype(float).values
    L = model_df["L"].astype(float).values
    D = model_df["D"].astype(float).values
    AI = model_df["AI"].astype(float).values
    H = model_df["H"].astype(float).values

    denominator = (
        (K ** alpha)
        * (L ** beta)
        * (D ** gamma)
        * (AI ** delta)
        * (H ** theta)
    )

    A_t = Y / denominator
    A_mean = np.mean(A_t)
    Y_hat = A_mean * denominator

    mape = np.mean(np.abs((Y - Y_hat) / Y)) * 100

    model_df["TFP_A"] = A_t
    model_df["A_mean"] = A_mean
    model_df["GDP_hat"] = Y_hat
    model_df["Error"] = Y - Y_hat
    model_df["APE_percent"] = np.abs((Y - Y_hat) / Y) * 100
    model_df["TFP_growth_percent"] = model_df["TFP_A"].pct_change() * 100

    return model_df, A_t, A_mean, Y_hat, mape


def growth_decomposition(model_df, alpha, beta, gamma, delta, theta):
    years = model_df["Year"].values

    dlnY = np.diff(np.log(model_df["GDP_trillion_VND"].astype(float).values))
    dlnK = np.diff(np.log(model_df["K"].astype(float).values))
    dlnL = np.diff(np.log(model_df["L"].astype(float).values))
    dlnD = np.diff(np.log(model_df["D"].astype(float).values))
    dlnAI = np.diff(np.log(model_df["AI"].astype(float).values))
    dlnH = np.diff(np.log(model_df["H"].astype(float).values))
    dlnA = np.diff(np.log(model_df["TFP_A"].astype(float).values))

    decomp_df = pd.DataFrame(
        {
            "Giai đoạn": [f"{years[i]}-{years[i + 1]}" for i in range(len(years) - 1)],
            "Tăng trưởng GDP thực tế (%)": dlnY * 100,
            "K - Vốn vật chất (%)": alpha * dlnK * 100,
            "L - Lao động (%)": beta * dlnL * 100,
            "D - Số hóa (%)": gamma * dlnD * 100,
            "AI - Năng lực AI (%)": delta * dlnAI * 100,
            "H - Nhân lực số (%)": theta * dlnH * 100,
            "TFP (%)": dlnA * 100,
        }
    )

    contribution_cols = [
        "K - Vốn vật chất (%)",
        "L - Lao động (%)",
        "D - Số hóa (%)",
        "AI - Năng lực AI (%)",
        "H - Nhân lực số (%)",
        "TFP (%)",
    ]

    decomp_df["Tổng đóng góp mô hình (%)"] = decomp_df[contribution_cols].sum(axis=1)

    avg_growth = decomp_df["Tăng trưởng GDP thực tế (%)"].mean()

    avg_contrib_df = pd.DataFrame(
        {
            "Yếu tố": ["K", "L", "D", "AI", "H", "TFP"],
            "Diễn giải": [
                "Vốn vật chất",
                "Lao động",
                "Số hóa / kinh tế số",
                "Năng lực AI / doanh nghiệp số",
                "Vốn nhân lực số",
                "Năng suất nhân tố tổng hợp",
            ],
            "Đóng góp bình quân vào tăng trưởng GDP (% điểm)": [
                decomp_df["K - Vốn vật chất (%)"].mean(),
                decomp_df["L - Lao động (%)"].mean(),
                decomp_df["D - Số hóa (%)"].mean(),
                decomp_df["AI - Năng lực AI (%)"].mean(),
                decomp_df["H - Nhân lực số (%)"].mean(),
                decomp_df["TFP (%)"].mean(),
            ],
        }
    )

    if abs(avg_growth) > 1e-12:
        avg_contrib_df["Tỷ trọng trong tăng trưởng GDP bình quân (%)"] = (
            avg_contrib_df["Đóng góp bình quân vào tăng trưởng GDP (% điểm)"]
            / avg_growth
            * 100
        )
    else:
        avg_contrib_df["Tỷ trọng trong tăng trưởng GDP bình quân (%)"] = np.nan

    return decomp_df, avg_contrib_df, avg_growth


def forecast_2030(
    model_df,
    alpha,
    beta,
    gamma,
    delta,
    theta,
    target_D_2030,
    target_AI_2030,
    target_H_2030,
    growth_K,
    growth_L,
    growth_TFP,
):
    last = model_df.iloc[-1]

    start_year = int(last["Year"])
    future_years = np.arange(start_year + 1, 2031)
    n = len(future_years)

    if n <= 0:
        return pd.DataFrame()

    K_future = np.array([last["K"] * ((1 + growth_K) ** i) for i in range(1, n + 1)])
    L_future = np.array([last["L"] * ((1 + growth_L) ** i) for i in range(1, n + 1)])
    A_future = np.array([last["TFP_A"] * ((1 + growth_TFP) ** i) for i in range(1, n + 1)])

    D_future = np.linspace(last["D"], target_D_2030, n + 1)[1:]
    AI_future = np.linspace(last["AI"], target_AI_2030, n + 1)[1:]
    H_future = np.linspace(last["H"], target_H_2030, n + 1)[1:]

    GDP_future = (
        A_future
        * (K_future ** alpha)
        * (L_future ** beta)
        * (D_future ** gamma)
        * (AI_future ** delta)
        * (H_future ** theta)
    )

    forecast_df = pd.DataFrame(
        {
            "Year": future_years,
            "K": K_future,
            "L": L_future,
            "D": D_future,
            "AI": AI_future,
            "H": H_future,
            "TFP_A": A_future,
            "GDP_forecast": GDP_future,
        }
    )

    return forecast_df


def get_tfp_comment(model_df):
    first_a = model_df["TFP_A"].iloc[0]
    last_a = model_df["TFP_A"].iloc[-1]
    change = (last_a / first_a - 1) * 100

    if change > 3:
        trend = "tăng rõ rệt"
        meaning = "chất lượng tăng trưởng có cải thiện, vì GDP tăng không chỉ nhờ mở rộng vốn, lao động và các yếu tố đầu vào quan sát được."
    elif change > 0:
        trend = "tăng nhẹ"
        meaning = "chất lượng tăng trưởng có dấu hiệu cải thiện, nhưng mức cải thiện chưa quá mạnh."
    elif change < -3:
        trend = "giảm rõ rệt"
        meaning = "tăng trưởng có thể đang phụ thuộc nhiều hơn vào mở rộng đầu vào thay vì cải thiện hiệu quả sử dụng nguồn lực."
    elif change < 0:
        trend = "giảm nhẹ"
        meaning = "hiệu quả tổng hợp của nền kinh tế chưa thật sự ổn định."
    else:
        trend = "gần như đi ngang"
        meaning = "TFP chưa tạo ra thay đổi đáng kể trong chất lượng tăng trưởng."

    return trend, change, meaning


def get_largest_new_factor(avg_contrib_df):
    sub = avg_contrib_df[avg_contrib_df["Yếu tố"].isin(["D", "AI", "H"])].copy()
    sub["abs_value"] = sub["Đóng góp bình quân vào tăng trưởng GDP (% điểm)"].abs()
    row = sub.sort_values("abs_value", ascending=False).iloc[0]

    return (
        row["Yếu tố"],
        row["Diễn giải"],
        row["Đóng góp bình quân vào tăng trưởng GDP (% điểm)"],
        row["Tỷ trọng trong tăng trưởng GDP bình quân (%)"],
    )


def render():
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2.05rem;
            font-weight: 850;
            line-height: 1.25;
            margin-bottom: 0.3rem;
        }
        .sub-title {
            color: #cbd5e1;
            font-size: 1rem;
            margin-bottom: 1rem;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.45rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="main-title">🌱 Bài 1 — Hàm sản xuất Cobb-Douglas mở rộng</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sub-title">Ước lượng TFP, dự báo GDP, phân rã tăng trưởng và thảo luận chính sách cho Việt Nam giai đoạn 2020-2025, mô phỏng đến 2030.</div>',
        unsafe_allow_html=True,
    )

    raw_df = load_data()

    # =====================================================
    # SIDEBAR — THAM SỐ TƯƠNG TÁC
    # =====================================================
    st.sidebar.header("⚙️ Tham số Bài 1")

    st.sidebar.markdown("### Hệ số Cobb-Douglas")

    alpha = st.sidebar.slider("α - Vốn vật chất K", 0.00, 0.80, 0.33, 0.01)
    beta = st.sidebar.slider("β - Lao động L", 0.00, 0.80, 0.42, 0.01)
    gamma = st.sidebar.slider("γ - Số hóa D", 0.00, 0.50, 0.10, 0.01)
    delta = st.sidebar.slider("δ - Năng lực AI", 0.00, 0.50, 0.08, 0.01)
    theta = st.sidebar.slider("θ - Nhân lực số H", 0.00, 0.50, 0.07, 0.01)

    coef_sum = alpha + beta + gamma + delta + theta

    normalize = st.sidebar.checkbox(
        "Tự chuẩn hóa tổng hệ số = 1",
        value=False,
        help="Bật lựa chọn này nếu muốn bảo đảm điều kiện lợi suất không đổi theo quy mô.",
    )

    if normalize and coef_sum > 0:
        alpha = alpha / coef_sum
        beta = beta / coef_sum
        gamma = gamma / coef_sum
        delta = delta / coef_sum
        theta = theta / coef_sum
        coef_sum = 1.0

    st.sidebar.info(f"Tổng hệ số hiện tại: {coef_sum:.3f}")

    st.sidebar.markdown("### Kịch bản đến năm 2030")

    target_D_2030 = st.sidebar.slider(
        "D năm 2030 - Kinh tế số/GDP (%)",
        19.5,
        45.0,
        30.0,
        0.5,
    )

    target_AI_2030 = st.sidebar.slider(
        "AI năm 2030 - Nghìn DN số",
        80.1,
        180.0,
        100.0,
        1.0,
    )

    target_H_2030 = st.sidebar.slider(
        "H năm 2030 - LĐ qua đào tạo (%)",
        29.2,
        60.0,
        35.0,
        0.5,
    )

    growth_K = st.sidebar.slider("Tăng trưởng K mỗi năm (%)", 0.0, 15.0, 6.0, 0.1) / 100
    growth_L = st.sidebar.slider("Tăng trưởng L mỗi năm (%)", -2.0, 10.0, 6.0, 0.1) / 100
    growth_TFP = st.sidebar.slider("Tăng trưởng TFP mỗi năm (%)", -2.0, 5.0, 1.2, 0.1) / 100

   
    model_df, A_t, A_mean, Y_hat, mape = compute_model(
        raw_df,
        alpha,
        beta,
        gamma,
        delta,
        theta,
    )

    decomp_df, avg_contrib_df, avg_growth = growth_decomposition(
        model_df,
        alpha,
        beta,
        gamma,
        delta,
        theta,
    )

    forecast_df = forecast_2030(
        model_df,
        alpha,
        beta,
        gamma,
        delta,
        theta,
        target_D_2030,
        target_AI_2030,
        target_H_2030,
        growth_K,
        growth_L,
        growth_TFP,
    )

    tfp_trend, tfp_change, tfp_meaning = get_tfp_comment(model_df)

    (
        largest_factor,
        largest_factor_name,
        largest_factor_value,
        largest_factor_share,
    ) = get_largest_new_factor(avg_contrib_df)
with st.expander("📊 Bảng dữ liệu tổng hợp Việt Nam 2020-2025", expanded=False):
    display_df = raw_df.rename(
        columns={
            "Year": "Năm",
            "GDP_trillion_VND": "Y (GDP, nghìn tỷ VND)",
            "K": "K (vốn tích lũy, nghìn tỷ)",
            "L": "L (triệu lao động)",
            "D": "D (KTS/GDP, %)",
            "AI": "AI (nghìn DN số)",
            "H": "H (LĐ qua đào tạo, %)",
        }
    )

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.caption(
        "Hệ số đề xuất có thể tinh chỉnh: α = 0,33; β = 0,42; γ = 0,10; δ = 0,08; θ = 0,07."
    )
    tabs = st.tabs(
        [
            "Ước lượng TFP",
            "Kiểm định dự báo GDP",
            "Phân rã nguồn tăng trưởng",
            "Mô phỏng kịch bản 2030",
            "Hàm ý chính sách",
        ]
    )

    # =====================================================
    # TAB 1.4.1 — TFP
    # =====================================================
    with tabs[0]:
        st.subheader("Ước lượng năng suất nhân tố tổng hợp Aₜ")

        st.markdown(
            """
           **Yêu cầu:** Sử dụng bảng dữ liệu tổng hợp Việt Nam giai đoạn 2020-2025.
Dùng `numpy` và `pandas`, ước lượng năng suất nhân tố tổng hợp `A_t`
cho mỗi năm bằng cách giải ngược từ hàm sản xuất. Vẽ đồ thị `A_t`
theo năm và bình luận xu hướng.
            """
        )

        st.latex(
            r"A_t = \frac{Y_t}{K_t^\alpha \cdot L_t^\beta \cdot D_t^\gamma \cdot AI_t^\delta \cdot H_t^\theta}"
        )

        tfp_table = model_df[
            [
                "Year",
                "GDP_trillion_VND",
                "K",
                "L",
                "D",
                "AI",
                "H",
                "TFP_A",
                "TFP_growth_percent",
            ]
        ].copy()

        tfp_table = tfp_table.rename(
            columns={
                "Year": "Năm",
                "GDP_trillion_VND": "GDP thực tế",
                "K": "K",
                "L": "L",
                "D": "D",
                "AI": "AI",
                "H": "H",
                "TFP_A": "A_t - TFP",
                "TFP_growth_percent": "Tăng trưởng TFP (%)",
            }
        )

        st.dataframe(tfp_table, use_container_width=True, hide_index=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("TFP năm 2020", f"{model_df['TFP_A'].iloc[0]:,.3f}")
        col2.metric("TFP năm 2025", f"{model_df['TFP_A'].iloc[-1]:,.3f}")
        col3.metric("Thay đổi TFP 2020-2025", f"{tfp_change:.2f}%")

        fig_tfp = px.line(
            model_df,
            x="Year",
            y="TFP_A",
            markers=True,
            title="Xu hướng năng suất nhân tố tổng hợp A_t giai đoạn 2020-2025",
        )
        fig_tfp.update_traces(line=dict(width=4), marker=dict(size=10))
        fig_tfp.update_layout(
            xaxis_title="Năm",
            yaxis_title="TFP A_t",
            height=500,
        )
        st.plotly_chart(fig_tfp, use_container_width=True)

   

        st.success(
            f"TFP A_t được tính bằng cách giải ngược từ hàm sản xuất Cobb-Douglas mở rộng. "
            f"Kết quả cho thấy TFP giai đoạn 2020-2025 có xu hướng **{tfp_trend}**, "
            f"thay đổi khoảng **{tfp_change:.2f}%** từ năm 2020 đến năm 2025. "
            f"Điều này cho thấy **{tfp_meaning}**"
        )

    # =====================================================
    # TAB 1.4.2 — MAPE
    # =====================================================
    with tabs[1]:
        st.subheader("Dự báo GDP với A trung bình và báo cáo MAPE")

        st.markdown(
            """
            **Yêu cầu:** Lấy `A_t = trung bình A` của giai đoạn 2020-2025,
            tính sản lượng dự báo `Ŷ_t` và so sánh với `Y` thực tế.
            Báo cáo chỉ tiêu sai số MAPE.
            """
        )

        st.latex(r"\bar{A} = \frac{1}{n}\sum_{t=1}^{n} A_t")
        st.latex(
            r"\hat{Y}_t = \bar{A} \cdot K_t^\alpha \cdot L_t^\beta \cdot D_t^\gamma \cdot AI_t^\delta \cdot H_t^\theta"
        )
        st.latex(
            r"MAPE = \frac{1}{n}\sum \left|\frac{Y_t - \hat{Y}_t}{Y_t}\right| \times 100"
        )

        pred_table = model_df[
            ["Year", "GDP_trillion_VND", "GDP_hat", "Error", "APE_percent"]
        ].copy()

        pred_table = pred_table.rename(
            columns={
                "Year": "Năm",
                "GDP_trillion_VND": "GDP thực tế",
                "GDP_hat": "GDP dự báo Ŷ_t",
                "Error": "Sai số Y - Ŷ",
                "APE_percent": "APE (%)",
            }
        )

        st.dataframe(pred_table, use_container_width=True, hide_index=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("A trung bình", f"{A_mean:,.3f}")
        col2.metric("MAPE", f"{mape:.2f}%")
        col3.metric("Sai số tuyệt đối TB", f"{model_df['Error'].abs().mean():,.1f}")

        fig_compare = go.Figure()

        fig_compare.add_trace(
            go.Scatter(
                x=model_df["Year"],
                y=model_df["GDP_trillion_VND"],
                mode="lines+markers",
                name="GDP thực tế",
                line=dict(width=4),
            )
        )

        fig_compare.add_trace(
            go.Scatter(
                x=model_df["Year"],
                y=model_df["GDP_hat"],
                mode="lines+markers",
                name="GDP dự báo Ŷ_t",
                line=dict(width=4, dash="dash"),
            )
        )

        fig_compare.update_layout(
            title="So sánh GDP thực tế và GDP dự báo với A trung bình",
            xaxis_title="Năm",
            yaxis_title="GDP, nghìn tỷ VND",
            height=520,
        )

        st.plotly_chart(fig_compare, use_container_width=True)

        fig_ape = px.bar(
            model_df,
            x="Year",
            y="APE_percent",
            text=model_df["APE_percent"].round(2),
            title="Sai số tuyệt đối phần trăm APE theo năm",
        )
        fig_ape.update_traces(textposition="outside")
        fig_ape.update_layout(
            xaxis_title="Năm",
            yaxis_title="APE (%)",
            height=450,
        )
        st.plotly_chart(fig_ape, use_container_width=True)

       

        if mape < 5:
            quality = "mức sai số thấp, mô hình khớp khá tốt với dữ liệu thực tế."
        elif mape < 10:
            quality = "mức sai số chấp nhận được, mô hình mô phỏng tương đối hợp lý."
        else:
            quality = "mức sai số còn cao, mô hình cần được hiệu chỉnh thêm về hệ số hoặc biến đầu vào."

        st.success(
            f"A trung bình giai đoạn 2020-2025 là **{A_mean:.3f}**. "
            f"Khi sử dụng A trung bình để dự báo GDP, chỉ tiêu MAPE đạt **{mape:.2f}%**. "
            f"Điều này cho thấy {quality}"
        )

    # =====================================================
    # TAB 1.4.3 — PHÂN RÃ
    # =====================================================
    with tabs[2]:
        st.subheader("Phân rã tăng trưởng GDP giai đoạn 2020-2025")

        st.markdown(
            """
            **Yêu cầu:** Thực hiện phân rã tăng trưởng giai đoạn 2020-2025:
            trong tổng tăng trưởng GDP bình quân hằng năm, bao nhiêu phần trăm do
            `K`, `L`, `D`, `AI`, `H` và `TFP` đóng góp?
            Trình bày kết quả dưới dạng bảng và biểu đồ cột.
            """
        )

        st.latex(
            r"\Delta \ln Y_t = \Delta \ln A_t + \alpha \Delta \ln K_t + \beta \Delta \ln L_t + \gamma \Delta \ln D_t + \delta \Delta \ln AI_t + \theta \Delta \ln H_t"
        )

        st.markdown("### Bảng phân rã theo từng giai đoạn")
        st.dataframe(decomp_df, use_container_width=True, hide_index=True)

        st.markdown("### Bảng đóng góp bình quân")
        st.dataframe(avg_contrib_df, use_container_width=True, hide_index=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Tăng trưởng GDP bình quân", f"{avg_growth:.2f}%")
        col2.metric("Yếu tố mới đóng góp nổi bật", largest_factor)
        col3.metric("Đóng góp của yếu tố này", f"{largest_factor_value:.2f} điểm %")

        contribution_cols = [
            "K - Vốn vật chất (%)",
            "L - Lao động (%)",
            "D - Số hóa (%)",
            "AI - Năng lực AI (%)",
            "H - Nhân lực số (%)",
            "TFP (%)",
        ]

        decomp_long = decomp_df.melt(
            id_vars="Giai đoạn",
            value_vars=contribution_cols,
            var_name="Nguồn đóng góp",
            value_name="Đóng góp (% điểm)",
        )

        fig_decomp = px.bar(
            decomp_long,
            x="Giai đoạn",
            y="Đóng góp (% điểm)",
            color="Nguồn đóng góp",
            title="Phân rã tăng trưởng GDP theo từng giai đoạn",
        )
        fig_decomp.update_layout(
            barmode="relative",
            height=540,
        )
        st.plotly_chart(fig_decomp, use_container_width=True)

        fig_avg = px.bar(
            avg_contrib_df,
            x="Yếu tố",
            y="Tỷ trọng trong tăng trưởng GDP bình quân (%)",
            text=avg_contrib_df["Tỷ trọng trong tăng trưởng GDP bình quân (%)"].round(2),
            hover_data=[
                "Diễn giải",
                "Đóng góp bình quân vào tăng trưởng GDP (% điểm)",
            ],
            title="Tỷ trọng đóng góp vào tăng trưởng GDP bình quân 2020-2025",
        )
        fig_avg.update_traces(textposition="outside")
        fig_avg.update_layout(
            yaxis_title="Tỷ trọng trong tăng trưởng GDP bình quân (%)",
            height=500,
        )
        st.plotly_chart(fig_avg, use_container_width=True)

   

        st.success(
            f"Tăng trưởng GDP bình quân giai đoạn 2020-2025 theo dạng log đạt khoảng **{avg_growth:.2f}%/năm**. "
            f"Trong nhóm yếu tố mới gồm D, AI và H, yếu tố đóng góp nổi bật nhất là "
            f"**{largest_factor} - {largest_factor_name}**, với mức đóng góp bình quân khoảng "
            f"**{largest_factor_value:.2f} điểm phần trăm**, tương đương khoảng "
            f"**{largest_factor_share:.2f}%** trong tăng trưởng GDP bình quân. "
            f"Kết quả này cho thấy các yếu tố số hóa, AI và nhân lực số có vai trò quan trọng trong mô hình tăng trưởng mới."
        )

    # =====================================================
    # TAB 1.4.4 — DỰ BÁO 2030
    # =====================================================
    with tabs[3]:
        st.subheader("Mô phỏng và dự báo GDP Việt Nam đến năm 2030")

        st.markdown(
            """
            **Yêu cầu:** Giả sử kịch bản đến 2030:
            `D` tăng lên 30%, `AI = 100` nghìn DN, `H = 35%`,
            `K` và `L` tăng trưởng đều 6%/năm, `TFP` tăng 1,2%/năm.
            Hãy mô phỏng và dự báo GDP Việt Nam năm 2030.
            """
        )

        st.info("Bạn có thể chỉnh trực tiếp các giả định của kịch bản 2030 trong sidebar bên trái.")

        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

        if not forecast_df.empty and 2030 in forecast_df["Year"].values:
            y_2025 = model_df["GDP_trillion_VND"].iloc[-1]
            y_2030 = forecast_df.loc[forecast_df["Year"] == 2030, "GDP_forecast"].iloc[0]
            increase_2030 = (y_2030 / y_2025 - 1) * 100

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("GDP thực tế 2025", f"{y_2025:,.1f}")
            col2.metric("GDP dự báo 2030", f"{y_2030:,.1f}")
            col3.metric("Tăng so với 2025", f"{increase_2030:.2f}%")
            col4.metric("D mục tiêu 2030", f"{target_D_2030:.1f}%")

            actual_df = model_df[["Year", "GDP_trillion_VND"]].rename(
                columns={"GDP_trillion_VND": "GDP"}
            )
            actual_df["Loại"] = "Thực tế"

            future_df = forecast_df[["Year", "GDP_forecast"]].rename(
                columns={"GDP_forecast": "GDP"}
            )
            future_df["Loại"] = "Dự báo"

            combined_df = pd.concat([actual_df, future_df], ignore_index=True)

            fig_forecast = px.line(
                combined_df,
                x="Year",
                y="GDP",
                color="Loại",
                markers=True,
                title="GDP thực tế 2020-2025 và GDP dự báo 2026-2030",
            )
            fig_forecast.update_traces(line=dict(width=4), marker=dict(size=9))
            fig_forecast.update_layout(
                xaxis_title="Năm",
                yaxis_title="GDP, nghìn tỷ VND",
                height=540,
            )
            st.plotly_chart(fig_forecast, use_container_width=True)

            future_long = forecast_df.melt(
                id_vars="Year",
                value_vars=["K", "L", "D", "AI", "H", "TFP_A"],
                var_name="Biến",
                value_name="Giá trị",
            )

            fig_inputs_future = px.line(
                future_long,
                x="Year",
                y="Giá trị",
                color="Biến",
                markers=True,
                title="Quỹ đạo các biến đầu vào trong kịch bản 2026-2030",
            )
            fig_inputs_future.update_layout(height=520)
            st.plotly_chart(fig_inputs_future, use_container_width=True)

        

            st.success(
                f"Với kịch bản hiện tại, GDP Việt Nam năm 2030 được dự báo đạt khoảng "
                f"**{y_2030:,.1f} nghìn tỷ VND**. "
                f"So với GDP năm 2025 là **{y_2025:,.1f} nghìn tỷ VND**, mức tăng tương ứng khoảng "
                f"**{increase_2030:.2f}%**. "
                f"Kết quả này phụ thuộc vào giả định tăng trưởng đều của K, L, mức tăng TFP và khả năng đạt mục tiêu "
                f"D = {target_D_2030:.1f}%, AI = {target_AI_2030:.1f} nghìn DN số, H = {target_H_2030:.1f}% vào năm 2030."
            )
        else:
            st.warning("Không đủ dữ liệu để dự báo đến năm 2030.")

    # =====================================================
    # TAB 1.5 — CHÍNH SÁCH
    # =====================================================
    with tabs[4]:
        st.subheader("1.5. Câu hỏi thảo luận chính sách")

        st.markdown("## a) TFP của Việt Nam có xu hướng tăng hay giảm trong giai đoạn 2020-2025?")

        st.success(
            f"TFP của Việt Nam trong mô hình có xu hướng **{tfp_trend}**, "
            f"với mức thay đổi khoảng **{tfp_change:.2f}%** trong giai đoạn 2020-2025."
        )

        st.markdown(
            f"""
            Điều này cho thấy **{tfp_meaning}**

            Về mặt chính sách, TFP là chỉ báo quan trọng phản ánh chất lượng tăng trưởng.
            Nếu GDP tăng chủ yếu nhờ tăng vốn và lao động, tăng trưởng có thể mở rộng về quy mô nhưng chưa chắc bền vững.
            Ngược lại, nếu TFP tăng, nền kinh tế đang tạo ra nhiều sản lượng hơn từ cùng một lượng đầu vào,
            phản ánh cải thiện về công nghệ, quản trị, tổ chức sản xuất, năng lực số và hiệu quả phân bổ nguồn lực.
            """
        )

        st.markdown("## b) Trong các yếu tố mới D, AI, H, yếu tố nào đóng góp nhiều nhất cho tăng trưởng? Vì sao?")

        st.success(
            f"Trong ba yếu tố mới D, AI và H, yếu tố đóng góp nhiều nhất theo mô hình hiện tại là "
            f"**{largest_factor} - {largest_factor_name}**."
        )

        st.markdown(
            f"""
            Yếu tố này có mức đóng góp bình quân khoảng **{largest_factor_value:.2f} điểm phần trăm**,
            tương đương khoảng **{largest_factor_share:.2f}%** trong tăng trưởng GDP bình quân.

            Nguyên nhân là đóng góp của một yếu tố trong mô hình Cobb-Douglas phụ thuộc vào hai thành phần:
            **tốc độ tăng của yếu tố đó** và **hệ số co giãn của nó trong hàm sản xuất**.
            Nếu một biến như D, AI hoặc H tăng nhanh và có hệ số co giãn đủ lớn,
            nó sẽ tạo ra đóng góp đáng kể vào tăng trưởng GDP.

            Về ý nghĩa kinh tế, D phản ánh mức độ số hóa của nền kinh tế,
            AI phản ánh năng lực công nghệ và số lượng doanh nghiệp số,
            còn H phản ánh chất lượng nguồn nhân lực.
            Ba yếu tố này có quan hệ bổ trợ: hạ tầng số và doanh nghiệp công nghệ chỉ phát huy hiệu quả khi có nhân lực số đủ tốt.
            """
        )

        st.markdown("## c) Mục tiêu Việt Nam đạt 30% kinh tế số/GDP vào 2030 có khả thi không? Cần ràng buộc gì?")

        if target_D_2030 >= 30:
            st.success(
                "Trong kịch bản hiện tại, mục tiêu kinh tế số đạt 30% GDP vào năm 2030 được đưa vào mô hình và có thể đạt được về mặt mô phỏng."
            )
        else:
            st.warning(
                "Trong kịch bản hiện tại, D mục tiêu đang thấp hơn 30%, vì vậy chưa phản ánh đầy đủ mục tiêu kinh tế số đạt 30% GDP vào năm 2030."
            )

        if not forecast_df.empty and 2030 in forecast_df["Year"].values:
            y_2030_policy = forecast_df.loc[forecast_df["Year"] == 2030, "GDP_forecast"].iloc[0]
        else:
            y_2030_policy = np.nan

        st.markdown(
            f"""
            Theo kết quả mô phỏng, khi D đạt **{target_D_2030:.1f}%**, AI đạt **{target_AI_2030:.1f} nghìn DN số**,
            H đạt **{target_H_2030:.1f}%**, K và L duy trì tăng trưởng, đồng thời TFP tăng đều,
            GDP năm 2030 có thể đạt khoảng **{y_2030_policy:,.1f} nghìn tỷ VND**.

            Tuy nhiên, mục tiêu 30% kinh tế số/GDP **không tự động khả thi** nếu chỉ tăng tỷ trọng kinh tế số trên giấy tờ.
            Cần các ràng buộc chính sách sau:

            - Thứ nhất, hạ tầng số phải được đầu tư đồng bộ, bao gồm dữ liệu, nền tảng số, điện toán đám mây, băng rộng và an toàn thông tin.
            - Thứ hai, số lượng doanh nghiệp công nghệ số và doanh nghiệp ứng dụng AI phải tăng thực chất, không chỉ tăng về số lượng đăng ký.
            - Thứ ba, vốn nhân lực số phải tăng nhanh, vì công nghệ chỉ tạo năng suất khi người lao động có kỹ năng sử dụng.
            - Thứ tư, TFP cần duy trì xu hướng tăng, nếu không tăng trưởng sẽ vẫn phụ thuộc chủ yếu vào vốn và lao động.
            - Thứ năm, cần hoàn thiện thể chế dữ liệu, chính sách cạnh tranh, bảo vệ quyền riêng tư và khuyến khích đổi mới sáng tạo.

            **Kết luận:** mục tiêu kinh tế số đạt 30% GDP vào năm 2030 là có cơ sở khả thi trong mô hình,
            nhưng điều kiện then chốt là phải đồng thời nâng cấp **D, AI, H và TFP**.
            Nếu chỉ tăng D mà không cải thiện AI, nhân lực số và năng suất tổng hợp,
            tác động đến tăng trưởng sẽ bị giới hạn.
            """
        )

      

        st.info(
            f"Mô hình Cobb-Douglas mở rộng cho thấy GDP Việt Nam giai đoạn 2020-2025 có thể được phân tích "
            f"thông qua sáu nguồn đóng góp: K, L, D, AI, H và TFP. "
            f"MAPE của mô hình dùng A trung bình là **{mape:.2f}%**. "
            f"TFP có xu hướng **{tfp_trend}**. "
            f"Trong nhóm yếu tố mới, yếu tố nổi bật nhất là **{largest_factor} - {largest_factor_name}**. "
            f"Kịch bản 2030 cho thấy mục tiêu kinh tế số 30% GDP có thể khả thi nếu đi kèm tăng trưởng TFP, AI và nhân lực số."
        )


def run():
    render()

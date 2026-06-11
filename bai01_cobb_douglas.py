# bai01_cobb_douglas.py
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# BÀI 1 — COBB-DOUGLAS MỞ RỘNG CHO KINH TẾ VIỆT NAM
# =========================================================


# ---------------------------------------------------------
# 1. DỮ LIỆU MẶC ĐỊNH
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# 2. HÀM CHUYỂN ĐỔI SỐ LIỆU KIỂU VIỆT NAM
# ---------------------------------------------------------
def parse_vietnamese_number(x):
    """
    Chuyển chuỗi kiểu Việt Nam như '8.044,4' hoặc '16.500'
    thành số float: 8044.4 hoặc 16500.
    """
    if pd.isna(x):
        return np.nan

    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    text = str(x).strip()

    if text == "":
        return np.nan

    # Nếu có cả dấu . và dấu , thì hiểu là định dạng Việt Nam: 8.044,4
    if "." in text and "," in text:
        text = text.replace(".", "").replace(",", ".")
    # Nếu chỉ có dấu phẩy thì đổi thành dấu chấm
    elif "," in text:
        text = text.replace(",", ".")
    # Nếu chỉ có dấu chấm và phần sau có đúng 3 chữ số, hiểu là phân tách nghìn
    elif "." in text:
        parts = text.split(".")
        if len(parts[-1]) == 3 and len(parts) > 1:
            text = text.replace(".", "")

    try:
        return float(text)
    except ValueError:
        return np.nan


# ---------------------------------------------------------
# 3. ĐỌC FILE CSV HOẶC DÙNG DỮ LIỆU MẶC ĐỊNH
# ---------------------------------------------------------
def load_data():
    default_df = get_default_data()

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

        # Chuẩn hóa tên cột nếu file CSV có tên cột khác nhẹ
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

        df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

        # Nếu CSV chỉ có GDP thì tự bổ sung K, L, D, AI, H từ đề bài
        for col in ["Year", "GDP_trillion_VND", "K", "L", "D", "AI", "H"]:
            if col not in df.columns:
                df[col] = default_df[col].values[: len(df)]

        df = df[["Year", "GDP_trillion_VND", "K", "L", "D", "AI", "H"]].copy()

        for col in ["Year", "GDP_trillion_VND", "K", "L", "D", "AI", "H"]:
            df[col] = df[col].apply(parse_vietnamese_number)

        df = df.dropna().sort_values("Year").reset_index(drop=True)
        df["Year"] = df["Year"].astype(int)

        if len(df) < 2:
            return default_df, "default_error", csv_path

        return df, "csv", csv_path

    except Exception:
        return default_df, "default_error", csv_path


# ---------------------------------------------------------
# 4. TÍNH TOÁN COBB-DOUGLAS
# ---------------------------------------------------------
def compute_model(df, alpha, beta, gamma, delta, theta):
    out = df.copy()

    Y = out["GDP_trillion_VND"].astype(float).values
    K = out["K"].astype(float).values
    L = out["L"].astype(float).values
    D = out["D"].astype(float).values
    AI = out["AI"].astype(float).values
    H = out["H"].astype(float).values

    denominator = (K**alpha) * (L**beta) * (D**gamma) * (AI**delta) * (H**theta)

    A_t = Y / denominator
    A_mean = np.mean(A_t)
    Y_hat = A_mean * denominator

    mape = np.mean(np.abs((Y - Y_hat) / Y)) * 100

    out["TFP_A"] = A_t
    out["A_mean"] = A_mean
    out["Y_hat_A_mean"] = Y_hat
    out["Forecast_Error"] = Y - Y_hat
    out["APE_percent"] = np.abs((Y - Y_hat) / Y) * 100

    return out, A_t, A_mean, Y_hat, mape


# ---------------------------------------------------------
# 5. PHÂN RÃ TĂNG TRƯỞNG DẠNG LOG
# ---------------------------------------------------------
def growth_decomposition(model_df, alpha, beta, gamma, delta, theta):
    years = model_df["Year"].values

    log_cols = ["GDP_trillion_VND", "K", "L", "D", "AI", "H", "TFP_A"]

    log_data = {}
    for col in log_cols:
        log_data[col] = np.log(model_df[col].astype(float).values)

    dlnY = np.diff(log_data["GDP_trillion_VND"])
    dlnK = np.diff(log_data["K"])
    dlnL = np.diff(log_data["L"])
    dlnD = np.diff(log_data["D"])
    dlnAI = np.diff(log_data["AI"])
    dlnH = np.diff(log_data["H"])
    dlnA = np.diff(log_data["TFP_A"])

    decomp = pd.DataFrame(
        {
            "Giai đoạn": [f"{years[i]}-{years[i + 1]}" for i in range(len(years) - 1)],
            "Tăng trưởng GDP thực tế (%)": dlnY * 100,
            "Đóng góp K (%)": alpha * dlnK * 100,
            "Đóng góp L (%)": beta * dlnL * 100,
            "Đóng góp D (%)": gamma * dlnD * 100,
            "Đóng góp AI (%)": delta * dlnAI * 100,
            "Đóng góp H (%)": theta * dlnH * 100,
            "Đóng góp TFP (%)": dlnA * 100,
        }
    )

    contribution_cols = [
        "Đóng góp K (%)",
        "Đóng góp L (%)",
        "Đóng góp D (%)",
        "Đóng góp AI (%)",
        "Đóng góp H (%)",
        "Đóng góp TFP (%)",
    ]

    decomp["Tổng đóng góp mô hình (%)"] = decomp[contribution_cols].sum(axis=1)

    avg_growth = decomp["Tăng trưởng GDP thực tế (%)"].mean()

    avg_contrib = pd.DataFrame(
        {
            "Yếu tố": ["K", "L", "D", "AI", "H", "TFP"],
            "Diễn giải": [
                "Vốn vật chất",
                "Lao động",
                "Kinh tế số / số hóa",
                "Năng lực AI / DN công nghệ số",
                "Vốn nhân lực số",
                "Năng suất nhân tố tổng hợp",
            ],
            "Đóng góp bình quân vào tăng trưởng GDP (% điểm)": [
                decomp["Đóng góp K (%)"].mean(),
                decomp["Đóng góp L (%)"].mean(),
                decomp["Đóng góp D (%)"].mean(),
                decomp["Đóng góp AI (%)"].mean(),
                decomp["Đóng góp H (%)"].mean(),
                decomp["Đóng góp TFP (%)"].mean(),
            ],
        }
    )

    if abs(avg_growth) > 1e-12:
        avg_contrib["Tỷ trọng trong tăng trưởng GDP bình quân (%)"] = (
            avg_contrib["Đóng góp bình quân vào tăng trưởng GDP (% điểm)"] / avg_growth * 100
        )
    else:
        avg_contrib["Tỷ trọng trong tăng trưởng GDP bình quân (%)"] = np.nan

    return decomp, avg_contrib, avg_growth


# ---------------------------------------------------------
# 6. DỰ BÁO ĐẾN 2030
# ---------------------------------------------------------
def forecast_to_2030(
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

    Y_future = (
        A_future
        * (K_future**alpha)
        * (L_future**beta)
        * (D_future**gamma)
        * (AI_future**delta)
        * (H_future**theta)
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
            "GDP_forecast": Y_future,
        }
    )

    return forecast_df


# ---------------------------------------------------------
# 7. HÀM SINH NHẬN XÉT TỰ ĐỘNG
# ---------------------------------------------------------
def tfp_comment(model_df):
    first_a = model_df["TFP_A"].iloc[0]
    last_a = model_df["TFP_A"].iloc[-1]
    change = (last_a / first_a - 1) * 100

    if change > 3:
        trend = "tăng rõ rệt"
        meaning = "chất lượng tăng trưởng có cải thiện, nền kinh tế không chỉ dựa vào mở rộng đầu vào."
    elif change > 0:
        trend = "tăng nhẹ"
        meaning = "chất lượng tăng trưởng có dấu hiệu cải thiện nhưng mức cải thiện còn thận trọng."
    elif change < -3:
        trend = "giảm rõ rệt"
        meaning = "tăng trưởng có thể đang phụ thuộc nhiều hơn vào vốn, lao động và các đầu vào quan sát được."
    elif change < 0:
        trend = "giảm nhẹ"
        meaning = "hiệu quả sử dụng các nguồn lực chưa thật sự ổn định."
    else:
        trend = "gần như đi ngang"
        meaning = "năng suất tổng hợp chưa tạo ra thay đổi đáng kể."

    return trend, change, meaning


def find_largest_new_factor(avg_contrib):
    sub = avg_contrib[avg_contrib["Yếu tố"].isin(["D", "AI", "H"])].copy()
    sub["abs_value"] = sub["Đóng góp bình quân vào tăng trưởng GDP (% điểm)"].abs()
    row = sub.sort_values("abs_value", ascending=False).iloc[0]
    return row["Yếu tố"], row["Diễn giải"], row["Đóng góp bình quân vào tăng trưởng GDP (% điểm)"]


# ---------------------------------------------------------
# 8. GIAO DIỆN CHÍNH
# ---------------------------------------------------------
def render():
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2.1rem;
            font-weight: 850;
            line-height: 1.2;
            margin-bottom: 0.2rem;
        }
        .sub-note {
            color: #cbd5e1;
            font-size: 1.02rem;
            margin-bottom: 1.1rem;
        }
        .metric-card {
            padding: 1rem;
            border-radius: 1rem;
            background: rgba(148, 163, 184, 0.10);
            border: 1px solid rgba(148, 163, 184, 0.20);
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.55rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-title">🌱 Bài 1 — Cobb-Douglas mở rộng cho kinh tế Việt Nam</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-note">Mô hình hóa GDP Việt Nam với vốn vật chất, lao động, số hóa, AI, vốn nhân lực số và TFP.</div>',
        unsafe_allow_html=True,
    )

    raw_df, data_status, csv_path = load_data()

    # =========================
    # SIDEBAR
    # =========================
    st.sidebar.header("⚙️ Tham số mô hình")

    st.sidebar.markdown("### Hệ số co giãn")
    alpha = st.sidebar.slider("α — Vốn vật chất K", 0.00, 0.80, 0.33, 0.01)
    beta = st.sidebar.slider("β — Lao động L", 0.00, 0.80, 0.42, 0.01)
    gamma = st.sidebar.slider("γ — Số hóa D", 0.00, 0.50, 0.10, 0.01)
    delta = st.sidebar.slider("δ — Năng lực AI", 0.00, 0.50, 0.08, 0.01)
    theta = st.sidebar.slider("θ — Nhân lực số H", 0.00, 0.50, 0.07, 0.01)

    coef_sum = alpha + beta + gamma + delta + theta
    normalize = st.sidebar.checkbox(
        "Tự chuẩn hóa để α + β + γ + δ + θ = 1",
        value=False,
        help="Nếu bật, các hệ số được chia cho tổng hiện tại để bảo đảm lợi suất không đổi theo quy mô.",
    )

    if normalize and coef_sum > 0:
        alpha, beta, gamma, delta, theta = [
            x / coef_sum for x in [alpha, beta, gamma, delta, theta]
        ]
        coef_sum = 1.0

    st.sidebar.info(f"Tổng hệ số hiện tại: {coef_sum:.3f}")

    st.sidebar.markdown("### Kịch bản 2030")
    target_D_2030 = st.sidebar.slider("D năm 2030 — Kinh tế số/GDP (%)", 19.5, 45.0, 30.0, 0.5)
    target_AI_2030 = st.sidebar.slider("AI năm 2030 — Nghìn DN số", 80.1, 180.0, 100.0, 1.0)
    target_H_2030 = st.sidebar.slider("H năm 2030 — LĐ qua đào tạo (%)", 29.2, 60.0, 35.0, 0.5)

    growth_K = st.sidebar.slider("Tăng trưởng K mỗi năm (%)", 0.0, 15.0, 6.0, 0.1) / 100
    growth_L = st.sidebar.slider("Tăng trưởng L mỗi năm (%)", -2.0, 10.0, 6.0, 0.1) / 100
    growth_TFP = st.sidebar.slider("Tăng trưởng TFP mỗi năm (%)", -2.0, 5.0, 1.2, 0.1) / 100

    # =========================
    # CHO PHÉP CHỈNH DỮ LIỆU
    # =========================
    with st.expander("🧾 Xem/chỉnh nhanh dữ liệu đầu vào 2020-2025", expanded=False):
        st.caption("Bạn có thể sửa trực tiếp bảng này để kiểm thử mô hình. Khi chạy trên GitHub/Streamlit, app vẫn ưu tiên đọc file vietnam_macro_2020_2025.csv nếu có.")
        edited_df = st.data_editor(
            raw_df,
            use_container_width=True,
            num_rows="fixed",
            column_config={
                "Year": st.column_config.NumberColumn("Năm", format="%d"),
                "GDP_trillion_VND": st.column_config.NumberColumn("GDP, nghìn tỷ VND", format="%.1f"),
                "K": st.column_config.NumberColumn("K, nghìn tỷ VND", format="%.1f"),
                "L": st.column_config.NumberColumn("L, triệu lao động", format="%.1f"),
                "D": st.column_config.NumberColumn("D, KTS/GDP (%)", format="%.1f"),
                "AI": st.column_config.NumberColumn("AI, nghìn DN số", format="%.1f"),
                "H": st.column_config.NumberColumn("H, LĐ qua đào tạo (%)", format="%.1f"),
            },
        )

    edited_df = edited_df.copy()
    for col in ["Year", "GDP_trillion_VND", "K", "L", "D", "AI", "H"]:
        edited_df[col] = pd.to_numeric(edited_df[col], errors="coerce")

    edited_df = edited_df.dropna().sort_values("Year").reset_index(drop=True)
    edited_df["Year"] = edited_df["Year"].astype(int)

    model_df, A_t, A_mean, Y_hat, mape = compute_model(
        edited_df, alpha, beta, gamma, delta, theta
    )

    decomp_df, avg_contrib_df, avg_growth = growth_decomposition(
        model_df, alpha, beta, gamma, delta, theta
    )

    forecast_df = forecast_to_2030(
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

    tfp_trend, tfp_change, tfp_meaning = tfp_comment(model_df)
    largest_factor, largest_factor_name, largest_factor_value = find_largest_new_factor(avg_contrib_df)

    # =========================
    # THÔNG BÁO DỮ LIỆU
    # =========================
    if data_status == "csv":
        st.success(f"✅ Đã đọc dữ liệu từ file: {csv_path}")
    elif data_status == "default_error":
        st.warning("⚠️ Có tìm thấy CSV nhưng không đọc được đúng định dạng. App đang dùng dữ liệu mặc định theo đề bài.")
    else:
        st.info("ℹ️ Chưa tìm thấy vietnam_macro_2020_2025.csv. App đang dùng dữ liệu mặc định theo đề bài.")

    # =========================
    # TABS
    # =========================
    tabs = st.tabs(
        [
            "📌 1.1 Bối cảnh",
            "🧮 1.2 Mô hình",
            "📊 1.3 Dữ liệu",
            "📈 1.4.1 TFP",
            "🎯 1.4.2 MAPE",
            "🧩 1.4.3 Phân rã",
            "🚀 1.4.4 Dự báo 2030",
            "🏛️ 1.5 Chính sách",
        ]
    )

    # =====================================================
    # TAB 1.1 — BỐI CẢNH
    # =====================================================
    with tabs[0]:
        st.subheader("1.1. Bối cảnh Việt Nam")

        st.markdown(
            """
            Theo bối cảnh đề bài, GDP Việt Nam năm 2024 đạt **11.511,9 nghìn tỷ VND**, tăng **7,09%**
            so với năm 2023; năng suất lao động năm 2024 đạt **221,9 triệu VND/người** và đến năm 2025
            đạt khoảng **245,0 triệu VND/người**.

            Đồng thời, đóng góp của khoa học - công nghệ vào GDP năm 2025 ước khoảng **2,49%**
            gồm **1,68% trực tiếp** và **0,81% lan tỏa**. Kinh tế số chiếm khoảng **18,3-19,5% GDP**.

            Câu hỏi phân tích chính của bài là: nếu mô hình hóa nền kinh tế Việt Nam bằng hàm sản xuất
            Cobb-Douglas mở rộng, có thêm các yếu tố **số hóa D**, **năng lực AI** và **vốn nhân lực số H**,
            thì sản lượng dự báo có khớp với số liệu thực tế không, và yếu tố nào đóng góp lớn nhất vào tăng trưởng?
            """
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("GDP 2024", "11.511,9", "nghìn tỷ VND")
        c2.metric("GDP 2025", "12.847,6", "nghìn tỷ VND")
        c3.metric("Kinh tế số 2025", "19,5%", "GDP")
        c4.metric("LĐ qua đào tạo 2025", "29,2%", "H")

        fig_context = go.Figure()
        fig_context.add_trace(
            go.Bar(
                x=model_df["Year"],
                y=model_df["GDP_trillion_VND"],
                name="GDP",
                text=np.round(model_df["GDP_trillion_VND"], 1),
                textposition="outside",
            )
        )
        fig_context.update_layout(
            title="GDP Việt Nam giai đoạn 2020-2025",
            xaxis_title="Năm",
            yaxis_title="GDP, nghìn tỷ VND",
            height=430,
        )
        st.plotly_chart(fig_context, use_container_width=True)

    # =====================================================
    # TAB 1.2 — MÔ HÌNH
    # =====================================================
    with tabs[1]:
        st.subheader("1.2. Mô hình toán học")

        st.markdown(
            r"""
            Hàm sản xuất tổng hợp cấp quốc gia được giả định có dạng Cobb-Douglas mở rộng:

            $$
            Y_t = A_t \cdot K_t^\alpha \cdot L_t^\beta \cdot D_t^\gamma \cdot AI_t^\delta \cdot H_t^\theta
            $$

            Với điều kiện lợi suất không đổi theo quy mô:

            $$
            \alpha + \beta + \gamma + \delta + \theta = 1
            $$

            Trong đó:

            - $Y_t$: GDP, đơn vị nghìn tỷ VND.
            - $A_t$: năng suất nhân tố tổng hợp, hay TFP.
            - $K_t$: vốn vật chất.
            - $L_t$: lao động.
            - $D_t$: chỉ số số hóa, đại diện bằng tỷ trọng kinh tế số trong GDP.
            - $AI_t$: năng lực AI, đại diện bằng số lượng doanh nghiệp công nghệ số.
            - $H_t$: vốn nhân lực số, đại diện bằng tỷ lệ lao động qua đào tạo có bằng cấp/chứng chỉ.
            """
        )

        st.markdown(
            r"""
            Dạng logarit của mô hình:

            $$
            \ln Y_t = \ln A_t + \alpha \ln K_t + \beta \ln L_t
            + \gamma \ln D_t + \delta \ln AI_t + \theta \ln H_t
            $$

            Phương trình phân rã tăng trưởng:

            $$
            \Delta \ln Y_t = \Delta \ln A_t
            + \alpha \Delta \ln K_t
            + \beta \Delta \ln L_t
            + \gamma \Delta \ln D_t
            + \delta \Delta \ln AI_t
            + \theta \Delta \ln H_t
            $$
            """
        )

        coef_df = pd.DataFrame(
            {
                "Hệ số": ["α", "β", "γ", "δ", "θ", "Tổng"],
                "Yếu tố": ["K", "L", "D", "AI", "H", "α+β+γ+δ+θ"],
                "Giá trị hiện tại": [alpha, beta, gamma, delta, theta, coef_sum],
                "Diễn giải": [
                    "Độ co giãn GDP theo vốn vật chất",
                    "Độ co giãn GDP theo lao động",
                    "Độ co giãn GDP theo số hóa",
                    "Độ co giãn GDP theo năng lực AI",
                    "Độ co giãn GDP theo vốn nhân lực số",
                    "Kiểm tra điều kiện lợi suất không đổi theo quy mô",
                ],
            }
        )

        st.dataframe(coef_df, use_container_width=True, hide_index=True)

        if abs(coef_sum - 1) <= 0.01:
            st.success("✅ Tổng hệ số xấp xỉ bằng 1. Mô hình phù hợp giả định lợi suất không đổi theo quy mô.")
        else:
            st.warning("⚠️ Tổng hệ số chưa bằng 1. Bạn có thể bật tùy chọn tự chuẩn hóa ở sidebar.")

    # =====================================================
    # TAB 1.3 — DỮ LIỆU
    # =====================================================
    with tabs[2]:
        st.subheader("1.3. Dữ liệu Việt Nam 2020-2025")

        st.markdown(
            """
            Bảng dữ liệu gồm GDP, vốn vật chất, lao động, tỷ trọng kinh tế số, năng lực AI
            và vốn nhân lực số trong giai đoạn 2020-2025.
            """
        )

        display_data = model_df[
            ["Year", "GDP_trillion_VND", "K", "L", "D", "AI", "H"]
        ].copy()

        display_data = display_data.rename(
            columns={
                "Year": "Năm",
                "GDP_trillion_VND": "Y - GDP, nghìn tỷ VND",
                "K": "K - Vốn tích lũy, nghìn tỷ",
                "L": "L - Triệu lao động",
                "D": "D - KTS/GDP (%)",
                "AI": "AI - Nghìn DN số",
                "H": "H - LĐ qua đào tạo (%)",
            }
        )

        st.dataframe(display_data, use_container_width=True, hide_index=True)

        long_df = model_df.melt(
            id_vars="Year",
            value_vars=["K", "L", "D", "AI", "H"],
            var_name="Biến",
            value_name="Giá trị",
        )

        fig_inputs = px.line(
            long_df,
            x="Year",
            y="Giá trị",
            color="Biến",
            markers=True,
            title="Xu hướng các yếu tố đầu vào K, L, D, AI, H",
        )
        fig_inputs.update_layout(height=500)
        st.plotly_chart(fig_inputs, use_container_width=True)

        fig_3d_data = px.scatter_3d(
            model_df,
            x="D",
            y="AI",
            z="GDP_trillion_VND",
            size="H",
            color="Year",
            hover_data=["K", "L", "TFP_A"],
            title="Không gian 3D: Số hóa D - AI - GDP, kích thước điểm biểu thị H",
        )
        fig_3d_data.update_layout(height=620)
        st.plotly_chart(fig_3d_data, use_container_width=True)

    # =====================================================
    # TAB 1.4.1 — TFP
    # =====================================================
    with tabs[3]:
        st.subheader("1.4.1. Ước lượng TFP Aₜ theo năm")

        st.markdown(
            r"""
            TFP được giải ngược từ hàm sản xuất:

            $$
            A_t = \frac{Y_t}{K_t^\alpha L_t^\beta D_t^\gamma AI_t^\delta H_t^\theta}
            $$
            """
        )

        tfp_table = model_df[["Year", "GDP_trillion_VND", "TFP_A"]].copy()
        tfp_table["TFP_growth_percent"] = tfp_table["TFP_A"].pct_change() * 100

        tfp_table = tfp_table.rename(
            columns={
                "Year": "Năm",
                "GDP_trillion_VND": "GDP thực tế",
                "TFP_A": "TFP A_t",
                "TFP_growth_percent": "Tăng trưởng TFP (%)",
            }
        )

        st.dataframe(tfp_table, use_container_width=True, hide_index=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("TFP 2020", f"{model_df['TFP_A'].iloc[0]:,.3f}")
        col2.metric("TFP 2025", f"{model_df['TFP_A'].iloc[-1]:,.3f}")
        col3.metric("Thay đổi TFP", f"{tfp_change:,.2f}%")

        fig_tfp = go.Figure()
        fig_tfp.add_trace(
            go.Scatter(
                x=model_df["Year"],
                y=model_df["TFP_A"],
                mode="lines+markers",
                name="TFP A_t",
                line=dict(width=4),
                marker=dict(size=10),
            )
        )
        fig_tfp.update_layout(
            title="Xu hướng TFP A_t giai đoạn 2020-2025",
            xaxis_title="Năm",
            yaxis_title="TFP A_t",
            height=480,
        )
        st.plotly_chart(fig_tfp, use_container_width=True)

        st.markdown("### 🤖 Tác nhân AI phân tích nhanh")
        st.info(
            f"TFP giai đoạn 2020-2025 có xu hướng **{tfp_trend}**, thay đổi khoảng **{tfp_change:.2f}%**. "
            f"Điều này hàm ý rằng **{tfp_meaning}**"
        )

    # =====================================================
    # TAB 1.4.2 — MAPE
    # =====================================================
    with tabs[4]:
        st.subheader("1.4.2. Dự báo GDP với A trung bình và tính MAPE")

        st.markdown(
            r"""
            Lấy:

            $$
            \bar{A} = \frac{1}{n}\sum_{t=1}^{n} A_t
            $$

            Sản lượng dự báo:

            $$
            \hat{Y}_t = \bar{A} K_t^\alpha L_t^\beta D_t^\gamma AI_t^\delta H_t^\theta
            $$

            Sai số MAPE:

            $$
            MAPE = \frac{1}{n}\sum \left|\frac{Y_t - \hat{Y}_t}{Y_t}\right| \times 100
            $$
            """
        )

        pred_table = model_df[
            ["Year", "GDP_trillion_VND", "Y_hat_A_mean", "Forecast_Error", "APE_percent"]
        ].copy()

        pred_table = pred_table.rename(
            columns={
                "Year": "Năm",
                "GDP_trillion_VND": "GDP thực tế",
                "Y_hat_A_mean": "GDP dự báo với A trung bình",
                "Forecast_Error": "Sai số",
                "APE_percent": "APE (%)",
            }
        )

        st.dataframe(pred_table, use_container_width=True, hide_index=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("A trung bình", f"{A_mean:,.3f}")
        col2.metric("MAPE", f"{mape:.2f}%")
        col3.metric("Sai số TB tuyệt đối", f"{model_df['Forecast_Error'].abs().mean():,.1f}")

        fig_pred = go.Figure()
        fig_pred.add_trace(
            go.Scatter(
                x=model_df["Year"],
                y=model_df["GDP_trillion_VND"],
                mode="lines+markers",
                name="GDP thực tế",
                line=dict(width=4),
            )
        )
        fig_pred.add_trace(
            go.Scatter(
                x=model_df["Year"],
                y=model_df["Y_hat_A_mean"],
                mode="lines+markers",
                name="GDP dự báo",
                line=dict(width=4, dash="dash"),
            )
        )
        fig_pred.update_layout(
            title="So sánh GDP thực tế và GDP dự báo bằng A trung bình",
            xaxis_title="Năm",
            yaxis_title="GDP, nghìn tỷ VND",
            height=500,
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        fig_error = px.bar(
            model_df,
            x="Year",
            y="APE_percent",
            title="Sai số tuyệt đối phần trăm APE theo năm",
            text=model_df["APE_percent"].round(2),
        )
        fig_error.update_traces(textposition="outside")
        fig_error.update_layout(yaxis_title="APE (%)", height=430)
        st.plotly_chart(fig_error, use_container_width=True)

        if mape < 5:
            st.success("✅ MAPE thấp. Mô hình với A trung bình khớp khá tốt với dữ liệu thực tế.")
        elif mape < 10:
            st.info("ℹ️ MAPE ở mức chấp nhận được. Mô hình có khả năng mô phỏng tương đối tốt.")
        else:
            st.warning("⚠️ MAPE còn cao. Có thể cần hiệu chỉnh hệ số, biến đầu vào hoặc cách đo lường TFP.")

    # =====================================================
    # TAB 1.4.3 — PHÂN RÃ TĂNG TRƯỞNG
    # =====================================================
    with tabs[5]:
        st.subheader("1.4.3. Phân rã tăng trưởng GDP giai đoạn 2020-2025")

        st.markdown(
            """
            Phần này cho biết trong tổng tăng trưởng GDP bình quân hằng năm,
            bao nhiêu phần đến từ vốn vật chất, lao động, số hóa, AI, vốn nhân lực số và TFP.
            """
        )

        st.markdown("### Bảng phân rã theo từng giai đoạn")
        st.dataframe(decomp_df, use_container_width=True, hide_index=True)

        st.markdown("### Bảng đóng góp bình quân 2020-2025")
        st.dataframe(avg_contrib_df, use_container_width=True, hide_index=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Tăng trưởng GDP bình quân", f"{avg_growth:.2f}%")
        col2.metric("Yếu tố mới nổi bật nhất", largest_factor)
        col3.metric("Đóng góp bình quân của yếu tố này", f"{largest_factor_value:.2f} điểm %")

        contribution_cols = [
            "Đóng góp K (%)",
            "Đóng góp L (%)",
            "Đóng góp D (%)",
            "Đóng góp AI (%)",
            "Đóng góp H (%)",
            "Đóng góp TFP (%)",
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
            title="Biểu đồ cột phân rã tăng trưởng GDP theo từng giai đoạn",
        )
        fig_decomp.update_layout(height=520, barmode="relative")
        st.plotly_chart(fig_decomp, use_container_width=True)

        fig_avg = px.bar(
            avg_contrib_df,
            x="Yếu tố",
            y="Tỷ trọng trong tăng trưởng GDP bình quân (%)",
            hover_data=["Diễn giải", "Đóng góp bình quân vào tăng trưởng GDP (% điểm)"],
            title="Tỷ trọng đóng góp vào tăng trưởng GDP bình quân 2020-2025",
            text=avg_contrib_df["Tỷ trọng trong tăng trưởng GDP bình quân (%)"].round(2),
        )
        fig_avg.update_traces(textposition="outside")
        fig_avg.update_layout(height=500, yaxis_title="Tỷ trọng (%)")
        st.plotly_chart(fig_avg, use_container_width=True)

        fig_3d_decomp = px.scatter_3d(
            decomp_df,
            x="Đóng góp D (%)",
            y="Đóng góp AI (%)",
            z="Đóng góp TFP (%)",
            size="Tăng trưởng GDP thực tế (%)",
            color="Giai đoạn",
            title="3D: Đóng góp của D, AI và TFP trong các giai đoạn tăng trưởng",
        )
        fig_3d_decomp.update_layout(height=620)
        st.plotly_chart(fig_3d_decomp, use_container_width=True)

        st.markdown("### 🤖 Tác nhân AI phân tích nhanh")
        st.info(
            f"Trong nhóm yếu tố mới gồm D, AI và H, yếu tố có đóng góp bình quân nổi bật nhất là "
            f"**{largest_factor} — {largest_factor_name}**, với mức đóng góp khoảng "
            f"**{largest_factor_value:.2f} điểm phần trăm** vào tăng trưởng GDP bình quân năm. "
            f"Kết quả này cho thấy chuyển đổi số và năng lực công nghệ có thể là trụ cột quan trọng "
            f"trong mô hình tăng trưởng mới."
        )

    # =====================================================
    # TAB 1.4.4 — DỰ BÁO 2030
    # =====================================================
    with tabs[6]:
        st.subheader("1.4.4. Mô phỏng và dự báo GDP Việt Nam năm 2030")

        st.markdown(
            """
            Kịch bản mặc định theo đề bài:

            - D tăng lên **30%** vào năm 2030.
            - AI đạt **100 nghìn doanh nghiệp số**.
            - H đạt **35%** lao động qua đào tạo.
            - K và L tăng trưởng đều **6%/năm**.
            - TFP tăng **1,2%/năm**.

            Bạn có thể thay đổi toàn bộ tham số kịch bản ở sidebar.
            """
        )

        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

        combined_gdp = pd.concat(
            [
                model_df[["Year", "GDP_trillion_VND"]].rename(
                    columns={"GDP_trillion_VND": "GDP"}
                ).assign(Loại="Thực tế"),
                forecast_df[["Year", "GDP_forecast"]].rename(
                    columns={"GDP_forecast": "GDP"}
                ).assign(Loại="Dự báo"),
            ],
            ignore_index=True,
        )

        fig_forecast = px.line(
            combined_gdp,
            x="Year",
            y="GDP",
            color="Loại",
            markers=True,
            title="GDP thực tế 2020-2025 và dự báo 2026-2030",
        )
        fig_forecast.update_layout(height=520, yaxis_title="GDP, nghìn tỷ VND")
        st.plotly_chart(fig_forecast, use_container_width=True)

        if not forecast_df.empty:
            y_2025 = model_df["GDP_trillion_VND"].iloc[-1]
            y_2030 = forecast_df.loc[forecast_df["Year"] == 2030, "GDP_forecast"].iloc[0]
            increase_2030 = (y_2030 / y_2025 - 1) * 100

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("GDP 2025", f"{y_2025:,.1f}")
            c2.metric("GDP dự báo 2030", f"{y_2030:,.1f}")
            c3.metric("Tăng so với 2025", f"{increase_2030:.2f}%")
            c4.metric("D mục tiêu 2030", f"{target_D_2030:.1f}%")

        future_long = forecast_df.melt(
            id_vars="Year",
            value_vars=["K", "L", "D", "AI", "H", "TFP_A"],
            var_name="Biến",
            value_name="Giá trị",
        )

        fig_future_inputs = px.line(
            future_long,
            x="Year",
            y="Giá trị",
            color="Biến",
            markers=True,
            title="Quỹ đạo các biến đầu vào trong kịch bản 2026-2030",
        )
        fig_future_inputs.update_layout(height=520)
        st.plotly_chart(fig_future_inputs, use_container_width=True)

        fig_3d_forecast = px.scatter_3d(
            forecast_df,
            x="D",
            y="AI",
            z="GDP_forecast",
            size="H",
            color="Year",
            hover_data=["K", "L", "TFP_A"],
            title="3D dự báo: D - AI - GDP 2030, kích thước điểm biểu thị H",
        )
        fig_3d_forecast.update_layout(height=650)
        st.plotly_chart(fig_3d_forecast, use_container_width=True)

        st.markdown("### 🤖 Tác nhân AI phân tích nhanh")
        if not forecast_df.empty:
            st.success(
                f"Với kịch bản hiện tại, GDP dự báo năm 2030 đạt khoảng **{y_2030:,.1f} nghìn tỷ VND**, "
                f"tăng khoảng **{increase_2030:.2f}%** so với năm 2025. "
                f"Kết quả phụ thuộc mạnh vào giả định tăng trưởng K, L, TFP và tốc độ mở rộng của D, AI, H."
            )

    # =====================================================
    # TAB 1.5 — CHÍNH SÁCH
    # =====================================================
    with tabs[7]:
        st.subheader("1.5. Câu hỏi thảo luận chính sách")

        st.markdown("### a) TFP của Việt Nam có xu hướng tăng hay giảm trong giai đoạn 2020-2025?")

        st.info(
            f"Trong mô hình hiện tại, TFP có xu hướng **{tfp_trend}** trong giai đoạn 2020-2025, "
            f"với mức thay đổi khoảng **{tfp_change:.2f}%**. "
            f"Điều này cho thấy **{tfp_meaning}**"
        )

        st.markdown(
            """
            Về chính sách, nếu TFP tăng, điều đó hàm ý chất lượng tăng trưởng được cải thiện:
            cùng một lượng vốn, lao động và đầu vào số, nền kinh tế tạo ra nhiều sản lượng hơn.
            Ngược lại, nếu TFP giảm hoặc đi ngang, tăng trưởng có thể vẫn phụ thuộc nhiều vào mở rộng quy mô
            vốn và lao động hơn là cải thiện hiệu quả.
            """
        )

        st.markdown("### b) Trong các yếu tố mới D, AI, H, yếu tố nào đóng góp nhiều nhất? Vì sao?")

        st.success(
            f"Trong ba yếu tố mới D, AI và H, yếu tố đóng góp nổi bật nhất theo kết quả hiện tại là "
            f"**{largest_factor} — {largest_factor_name}**."
        )

        st.markdown(
            """
            Có ba cách giải thích chính:

            1. **Tốc độ tăng của biến đầu vào**: yếu tố nào tăng nhanh hơn sẽ tạo ra đóng góp lớn hơn nếu hệ số co giãn đủ lớn.
            2. **Độ co giãn trong mô hình**: hệ số γ, δ, θ càng lớn thì cùng một mức tăng đầu vào sẽ tạo ra đóng góp GDP lớn hơn.
            3. **Hiệu ứng lan tỏa**: số hóa, AI và nhân lực số không chỉ tác động trực tiếp đến sản lượng, mà còn hỗ trợ đổi mới quy trình, giảm chi phí giao dịch và tăng hiệu quả phân bổ nguồn lực.
            """
        )

        st.markdown("### c) Mục tiêu kinh tế số đạt 30% GDP vào 2030 có khả thi không? Cần ràng buộc gì?")

        if target_D_2030 >= 30:
            st.success(
                "Theo kịch bản hiện tại, mục tiêu D đạt 30% GDP vào năm 2030 được đưa trực tiếp vào mô hình "
                "và có thể tạo ra mức GDP dự báo cao hơn so với năm 2025."
            )
        else:
            st.warning(
                "Trong kịch bản hiện tại, D mục tiêu đang thấp hơn 30%, vì vậy chưa phản ánh đầy đủ mục tiêu "
                "kinh tế số chiếm 30% GDP vào năm 2030."
            )

        st.markdown(
            """
            Tuy nhiên, tính khả thi của mục tiêu **30% kinh tế số/GDP vào 2030** cần các ràng buộc chính sách sau:

            - **Hạ tầng số** phải mở rộng đồng đều, bao gồm dữ liệu, điện toán đám mây, kết nối băng rộng và nền tảng số.
            - **Doanh nghiệp số và năng lực AI** phải tăng thực chất, không chỉ tăng về số lượng mà còn về năng lực ứng dụng trong sản xuất, logistics, tài chính, thương mại và dịch vụ công.
            - **Vốn nhân lực số H** phải cải thiện đủ nhanh, vì công nghệ chỉ tạo năng suất khi lao động có kỹ năng sử dụng.
            - **TFP phải tăng bền vững**, nếu TFP không tăng thì tăng trưởng dễ quay lại mô hình mở rộng đầu vào truyền thống.
            - **Thể chế dữ liệu, an toàn số và cạnh tranh thị trường** cần được hoàn thiện để giảm rủi ro độc quyền nền tảng, phân mảnh dữ liệu và đầu tư kém hiệu quả.

            Kết luận chính sách: mục tiêu 30% là **có khả năng đạt được trong mô hình**, nhưng không tự động xảy ra.
            Điều kiện then chốt là phải đồng thời nâng cấp **D, AI, H và TFP**, thay vì chỉ tăng riêng tỷ trọng kinh tế số.
            """
        )

        st.markdown("### ✅ Kết luận tổng hợp của Bài 1")

        st.markdown(
            f"""
            - Mô hình Cobb-Douglas mở rộng cho phép lượng hóa vai trò của **vốn, lao động, số hóa, AI, nhân lực số và TFP**.
            - MAPE hiện tại là **{mape:.2f}%**, phản ánh mức độ khớp giữa GDP thực tế và GDP dự báo với TFP trung bình.
            - TFP giai đoạn 2020-2025 có xu hướng **{tfp_trend}**.
            - Trong nhóm yếu tố mới, yếu tố nổi bật nhất là **{largest_factor} — {largest_factor_name}**.
            - Kịch bản 2030 cho thấy GDP phụ thuộc đáng kể vào khả năng duy trì tăng trưởng **K, L, TFP** và mở rộng **D, AI, H**.
            """
        )


# Cho phép app gọi bằng render() hoặc run()
def run():
    render()

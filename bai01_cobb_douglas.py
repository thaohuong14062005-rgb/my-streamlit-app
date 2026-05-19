# streamlit_app.py
# Bài 1 - Hàm sản xuất Cobb-Douglas mở rộng với AI và số hóa
# Chạy độc lập trên Streamlit Cloud

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# =========================
# 1. CẤU HÌNH TRANG
# =========================

st.set_page_config(
    page_title="Bài 1 - Cobb-Douglas mở rộng",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

px.defaults.template = "plotly_dark"


# =========================
# 2. CSS GIAO DIỆN
# =========================

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #020617 100%);
            color: #f8fafc;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #111827 0%, #1e293b 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        h1, h2, h3 {
            color: #f8fafc !important;
            font-weight: 800 !important;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        div[data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.95);
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 18px;
            padding: 18px;
            box-shadow: 0 8px 28px rgba(0,0,0,0.25);
        }

        .aideom-card {
            background: rgba(15, 23, 42, 0.82);
            border: 1px solid rgba(148, 163, 184, 0.22);
            padding: 22px;
            border-radius: 22px;
            margin-bottom: 18px;
            box-shadow: 0 10px 35px rgba(0,0,0,0.28);
        }

        .pill {
            display: inline-block;
            padding: 6px 13px;
            border-radius: 999px;
            background: linear-gradient(90deg, #10b981, #06b6d4);
            color: white;
            font-weight: 700;
            font-size: 13px;
            margin-right: 8px;
            margin-bottom: 6px;
        }

        .small-note {
            color: #cbd5e1;
            font-size: 15px;
            line-height: 1.65;
        }

        .formula-box {
            background: rgba(30, 41, 59, 0.85);
            border-left: 5px solid #10b981;
            padding: 16px 18px;
            border-radius: 14px;
            margin-bottom: 18px;
            color: #e2e8f0;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# 3. DỮ LIỆU MẶC ĐỊNH TỪ ĐỀ BÀI
# =========================

def build_default_macro_data() -> pd.DataFrame:
    """
    Dữ liệu Bài 1 trong đề:
    Y: GDP, nghìn tỷ VND
    K: vốn tích lũy, nghìn tỷ VND
    L: lao động, triệu người
    D: tỷ trọng kinh tế số/GDP, %
    AI: nghìn doanh nghiệp công nghệ số
    H: lao động qua đào tạo, %
    """
    return pd.DataFrame({
        "Năm": [2020, 2021, 2022, 2023, 2024, 2025],
        "Y": [8044.4, 8487.5, 9513.3, 10221.8, 11511.9, 12847.6],
        "K": [16500, 17800, 19600, 21300, 23500, 25900],
        "L": [53.6, 50.5, 51.7, 52.4, 52.9, 53.4],
        "D": [12.0, 12.7, 14.3, 16.5, 18.3, 19.5],
        "AI": [55.6, 60.2, 65.4, 67.0, 73.8, 80.1],
        "H": [24.1, 26.1, 26.2, 27.0, 28.4, 29.2],
    })


def clean_number(x):
    """
    Làm sạch số nếu file CSV dùng dấu phẩy kiểu Việt Nam.
    Ví dụ: '8.044,4' -> 8044.4
    """
    if pd.isna(x):
        return np.nan

    if isinstance(x, (int, float, np.number)):
        return float(x)

    text = str(x).strip().replace(" ", "")

    if "," in text and "." in text:
        # Ưu tiên định dạng Việt Nam: 8.044,4
        text = text.replace(".", "").replace(",", ".")
    elif "," in text:
        text = text.replace(",", ".")

    try:
        return float(text)
    except ValueError:
        return np.nan


def standardize_macro_columns(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa tên cột nếu người dùng upload file vietnam_macro_2020_2025.csv.
    Nếu không nhận diện được đủ cột, sẽ báo lỗi để dùng dữ liệu mặc định.
    """
    df = raw_df.copy()
    lower_cols = {str(c).strip().lower(): c for c in df.columns}

    candidates = {
        "Năm": ["year", "nam", "năm"],
        "Y": ["gdp_trillion_vnd", "gdp", "y", "y_thuc_te", "y_thực_tế"],
        "K": ["k", "capital", "capital_stock", "von", "vốn"],
        "L": ["l", "labor", "labour", "lao_dong", "lao động"],
        "D": ["d", "digital_economy", "digital", "kts", "kinh_te_so"],
        "AI": ["ai", "digital_firms", "technology_firms", "dn_so", "doanh_nghiep_so"],
        "H": ["h", "trained_labor", "human_capital", "lao_dong_qua_dao_tao"],
    }

    selected = {}

    for target_col, cand_list in candidates.items():
        found = None

        for cand in cand_list:
            if cand.lower() in lower_cols:
                found = lower_cols[cand.lower()]
                break

        if found is None:
            for original_col in df.columns:
                original_lower = str(original_col).strip().lower()
                if any(cand.lower() in original_lower for cand in cand_list):
                    found = original_col
                    break

        selected[target_col] = found

    if any(v is None for v in selected.values()):
        missing = [k for k, v in selected.items() if v is None]
        raise ValueError(f"Không nhận diện được các cột: {missing}")

    out = pd.DataFrame({
        "Năm": df[selected["Năm"]],
        "Y": df[selected["Y"]],
        "K": df[selected["K"]],
        "L": df[selected["L"]],
        "D": df[selected["D"]],
        "AI": df[selected["AI"]],
        "H": df[selected["H"]],
    })

    for col in ["Năm", "Y", "K", "L", "D", "AI", "H"]:
        out[col] = out[col].apply(clean_number)

    out = out.dropna().sort_values("Năm").reset_index(drop=True)
    out["Năm"] = out["Năm"].astype(int)

    return out


def load_macro_data(uploaded_file=None) -> tuple[pd.DataFrame, str]:
    """
    Thứ tự nạp dữ liệu:
    1. File người dùng upload trên giao diện.
    2. File vietnam_macro_2020_2025.csv ở cùng thư mục với streamlit_app.py.
    3. File data/vietnam_macro_2020_2025.csv.
    4. Dữ liệu mặc định từ đề bài.
    """
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            return standardize_macro_columns(raw_df), "CSV upload từ giao diện"
        except Exception as e:
            st.warning(f"Không đọc được file upload. App dùng dữ liệu mặc định. Chi tiết lỗi: {e}")
            return build_default_macro_data(), "Dữ liệu mặc định từ đề bài"

    candidate_paths = [
        Path("vietnam_macro_2020_2025.csv"),
        Path("data") / "vietnam_macro_2020_2025.csv",
    ]

    for p in candidate_paths:
        if p.exists():
            try:
                raw_df = pd.read_csv(p)
                return standardize_macro_columns(raw_df), f"File CSV: {p}"
            except Exception as e:
                st.warning(f"Tìm thấy {p}, nhưng không chuẩn hóa được cột. App dùng dữ liệu mặc định. Lỗi: {e}")
                return build_default_macro_data(), "Dữ liệu mặc định từ đề bài"

    return build_default_macro_data(), "Dữ liệu mặc định từ đề bài"


# =========================
# 4. HÀM TÍNH TOÁN MÔ HÌNH
# =========================

def compute_cobb_douglas(df, alpha, beta, gamma, delta, theta):
    """
    Tính:
    - TFP A_t
    - A trung bình
    - GDP dự báo Y_hat
    - APE và MAPE
    """
    out = df.copy()

    out["TFP_A_t"] = out["Y"] / (
        (out["K"] ** alpha)
        * (out["L"] ** beta)
        * (out["D"] ** gamma)
        * (out["AI"] ** delta)
        * (out["H"] ** theta)
    )

    A_mean = out["TFP_A_t"].mean()

    out["Y_dự_báo"] = A_mean * (
        (out["K"] ** alpha)
        * (out["L"] ** beta)
        * (out["D"] ** gamma)
        * (out["AI"] ** delta)
        * (out["H"] ** theta)
    )

    out["Sai_số_tuyệt_đối"] = (out["Y"] - out["Y_dự_báo"]).abs()
    out["APE_%"] = out["Sai_số_tuyệt_đối"] / out["Y"] * 100
    mape = out["APE_%"].mean()

    return out, A_mean, mape


def compute_growth_accounting(model_df, alpha, beta, gamma, delta, theta):
    """
    Phân rã tăng trưởng theo log:
    ΔlnY = ΔlnA + αΔlnK + βΔlnL + γΔlnD + δΔlnAI + θΔlnH
    """
    df = model_df.copy().sort_values("Năm").reset_index(drop=True)

    variables = ["Y", "TFP_A_t", "K", "L", "D", "AI", "H"]

    for col in variables:
        df[f"dln_{col}"] = np.log(df[col]).diff()

    acc = pd.DataFrame({
        "Giai đoạn": df["Năm"].astype(str).shift(1) + "-" + df["Năm"].astype(str),
        "Tăng trưởng GDP log": df["dln_Y"],
        "TFP": df["dln_TFP_A_t"],
        "K": alpha * df["dln_K"],
        "L": beta * df["dln_L"],
        "D": gamma * df["dln_D"],
        "AI": delta * df["dln_AI"],
        "H": theta * df["dln_H"],
    }).dropna().reset_index(drop=True)

    contribution_cols = ["TFP", "K", "L", "D", "AI", "H"]

    acc["Tổng đóng góp kiểm tra"] = acc[contribution_cols].sum(axis=1)
    acc["Sai lệch kiểm tra"] = acc["Tăng trưởng GDP log"] - acc["Tổng đóng góp kiểm tra"]

    avg_growth = acc["Tăng trưởng GDP log"].mean()
    avg_contrib = acc[contribution_cols].mean()

    summary = pd.DataFrame({
        "Yếu tố": contribution_cols,
        "Đóng góp bình quân log": avg_contrib.values,
        "Đóng góp bình quân, điểm % log": avg_contrib.values * 100,
        "Tỷ trọng trong tăng trưởng GDP, %": avg_contrib.values / avg_growth * 100,
    })

    summary = summary.sort_values("Tỷ trọng trong tăng trưởng GDP, %", ascending=False).reset_index(drop=True)

    return acc, summary, avg_growth


def forecast_to_2030(
    model_df,
    alpha,
    beta,
    gamma,
    delta,
    theta,
    target_D=30.0,
    target_AI=100.0,
    target_H=35.0,
    growth_K=0.06,
    growth_L=0.06,
    growth_A=0.012
):
    """
    Mô phỏng 2026-2030:
    - K tăng growth_K mỗi năm.
    - L tăng growth_L mỗi năm.
    - TFP tăng growth_A mỗi năm.
    - D, AI, H đi tuyến tính từ mức 2025 đến mục tiêu 2030.
    """
    base = model_df.sort_values("Năm").iloc[-1]

    base_year = int(base["Năm"])
    base_Y = float(base["Y"])
    base_K = float(base["K"])
    base_L = float(base["L"])
    base_D = float(base["D"])
    base_AI = float(base["AI"])
    base_H = float(base["H"])
    base_A = float(base["TFP_A_t"])

    rows = []

    for year in range(base_year, 2031):
        step = year - base_year

        if step == 0:
            K = base_K
            L = base_L
            D = base_D
            AI = base_AI
            H = base_H
            A = base_A
        else:
            K = base_K * ((1 + growth_K) ** step)
            L = base_L * ((1 + growth_L) ** step)
            A = base_A * ((1 + growth_A) ** step)

            # Nội suy tuyến tính từ 2025 đến 2030
            D = base_D + (target_D - base_D) * step / 5
            AI = base_AI + (target_AI - base_AI) * step / 5
            H = base_H + (target_H - base_H) * step / 5

        Y_hat = A * (K ** alpha) * (L ** beta) * (D ** gamma) * (AI ** delta) * (H ** theta)

        rows.append({
            "Năm": year,
            "A_TFP": A,
            "K": K,
            "L": L,
            "D": D,
            "AI": AI,
            "H": H,
            "GDP_dự_báo": Y_hat,
            "Tăng so với 2025, %": (Y_hat / base_Y - 1) * 100,
        })

    return pd.DataFrame(rows)


def format_df(df, decimals=3):
    return df.round(decimals)


# =========================
# 5. SIDEBAR THAM SỐ
# =========================

with st.sidebar:
    st.markdown("## 🌱 Bài 1")
    st.caption("Hàm sản xuất Cobb-Douglas mở rộng với AI và số hóa")

    uploaded_file = st.file_uploader(
        "Upload vietnam_macro_2020_2025.csv nếu có",
        type=["csv"]
    )

    st.divider()
    st.markdown("### Tham số độ co giãn")

    alpha = st.slider("α - Vốn vật chất K", 0.10, 0.60, 0.33, 0.01)
    beta = st.slider("β - Lao động L", 0.10, 0.70, 0.42, 0.01)
    gamma = st.slider("γ - Số hóa D", 0.01, 0.30, 0.10, 0.01)
    delta = st.slider("δ - Năng lực AI", 0.01, 0.30, 0.08, 0.01)

    theta = 1 - alpha - beta - gamma - delta
    st.metric("θ - Vốn nhân lực số H", f"{theta:.3f}")

    st.caption("θ được tự động tính để bảo đảm α + β + γ + δ + θ = 1.")

    st.divider()
    st.markdown("### Kịch bản 2030")

    target_D = st.number_input("D mục tiêu 2030, % GDP", value=30.0, step=0.5)
    target_AI = st.number_input("AI mục tiêu 2030, nghìn DN", value=100.0, step=1.0)
    target_H = st.number_input("H mục tiêu 2030, %", value=35.0, step=0.5)

    growth_K = st.number_input("Tăng trưởng K mỗi năm", value=0.06, step=0.005, format="%.3f")
    growth_L = st.number_input("Tăng trưởng L mỗi năm", value=0.06, step=0.005, format="%.3f")
    growth_A = st.number_input("Tăng trưởng TFP mỗi năm", value=0.012, step=0.001, format="%.3f")


# =========================
# 6. MAIN APP
# =========================

st.markdown(
    """
    <div class="aideom-card">
        <h1>🌱 Bài 1 — Hàm sản xuất Cobb-Douglas mở rộng với AI và số hóa</h1>
        <p class="small-note">
        Mục tiêu: ước lượng năng suất nhân tố tổng hợp TFP A<sub>t</sub>, dự báo GDP,
        tính MAPE, phân rã tăng trưởng 2020-2025 và mô phỏng GDP Việt Nam đến năm 2030.
        </p>
        <span class="pill">Cobb-Douglas</span>
        <span class="pill">Growth Accounting</span>
        <span class="pill">TFP</span>
        <span class="pill">Vietnam 2020-2025</span>
    </div>
    """,
    unsafe_allow_html=True
)

if theta <= 0:
    st.error(
        "Tổng α + β + γ + δ đã lớn hơn hoặc bằng 1, khiến θ ≤ 0. "
        "Hãy giảm một hoặc vài hệ số ở sidebar."
    )
    st.stop()


# Nạp dữ liệu
df_raw, data_source = load_macro_data(uploaded_file)

# Tính toán mô hình
model_df, A_mean, mape = compute_cobb_douglas(
    df_raw,
    alpha=alpha,
    beta=beta,
    gamma=gamma,
    delta=delta,
    theta=theta
)

growth_table, growth_summary, avg_log_growth = compute_growth_accounting(
    model_df,
    alpha=alpha,
    beta=beta,
    gamma=gamma,
    delta=delta,
    theta=theta
)

forecast_df = forecast_to_2030(
    model_df,
    alpha=alpha,
    beta=beta,
    gamma=gamma,
    delta=delta,
    theta=theta,
    target_D=target_D,
    target_AI=target_AI,
    target_H=target_H,
    growth_K=growth_K,
    growth_L=growth_L,
    growth_A=growth_A
)


# =========================
# 7. TABS NỘI DUNG
# =========================

tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📌 Mô hình",
    "📄 Dữ liệu",
    "1.4.1 TFP A_t",
    "1.4.2 Dự báo & MAPE",
    "1.4.3 Phân rã tăng trưởng",
    "1.4.4 Dự báo 2030",
    "🧠 Thảo luận chính sách"
])


# -------------------------
# TAB 0: MÔ HÌNH
# -------------------------

with tab0:
    st.markdown("## 1. Mô hình toán học")

    st.markdown(
        """
        <div class="formula-box">
        Hàm sản xuất tổng hợp được mở rộng để đưa thêm ba yếu tố mới:
        mức độ số hóa D, năng lực AI và vốn nhân lực số H.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.latex(r"Y_t = A_t \cdot K_t^\alpha \cdot L_t^\beta \cdot D_t^\gamma \cdot AI_t^\delta \cdot H_t^\theta")

    st.latex(r"\alpha + \beta + \gamma + \delta + \theta = 1")

    st.markdown("### Ý nghĩa biến")
    meaning_df = pd.DataFrame({
        "Ký hiệu": ["Y_t", "A_t", "K_t", "L_t", "D_t", "AI_t", "H_t"],
        "Ý nghĩa": [
            "GDP, sản lượng nền kinh tế, đơn vị nghìn tỷ VND",
            "Năng suất nhân tố tổng hợp TFP",
            "Vốn vật chất / vốn tích lũy",
            "Lao động, đơn vị triệu người",
            "Mức độ số hóa, đại diện bởi tỷ trọng kinh tế số/GDP",
            "Năng lực AI, đại diện bởi số doanh nghiệp công nghệ số",
            "Vốn nhân lực số, đại diện bởi tỷ lệ lao động qua đào tạo"
        ]
    })
    st.dataframe(meaning_df, use_container_width=True)

    st.markdown("### Tham số đang sử dụng")
    param_df = pd.DataFrame({
        "Tham số": ["α", "β", "γ", "δ", "θ"],
        "Yếu tố": ["K", "L", "D", "AI", "H"],
        "Giá trị": [alpha, beta, gamma, delta, theta]
    })
    st.dataframe(param_df, use_container_width=True)

    st.info(
        "Với bộ mặc định của đề bài: α = 0.33, β = 0.42, γ = 0.10, δ = 0.08, θ = 0.07. "
        "Các hệ số có thể điều chỉnh ở sidebar để phân tích nhạy cảm."
    )


# -------------------------
# TAB 1: DỮ LIỆU
# -------------------------

with tab1:
    st.markdown("## 2. Dữ liệu Việt Nam 2020-2025")

    st.caption(f"Nguồn đang dùng: {data_source}")

    display_data = df_raw.rename(columns={
        "Y": "Y - GDP, nghìn tỷ VND",
        "K": "K - Vốn tích lũy",
        "L": "L - Triệu lao động",
        "D": "D - Kinh tế số/GDP, %",
        "AI": "AI - Nghìn DN số",
        "H": "H - Lao động qua đào tạo, %"
    })

    st.dataframe(format_df(display_data, 3), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "GDP 2020",
        f"{df_raw['Y'].iloc[0]:,.1f}",
        "nghìn tỷ VND"
    )

    c2.metric(
        "GDP 2025",
        f"{df_raw['Y'].iloc[-1]:,.1f}",
        "nghìn tỷ VND"
    )

    gdp_cagr = (df_raw["Y"].iloc[-1] / df_raw["Y"].iloc[0]) ** (1 / (len(df_raw) - 1)) - 1
    c3.metric("CAGR GDP 2020-2025", f"{gdp_cagr * 100:.2f}%")

    digital_cagr = (df_raw["D"].iloc[-1] / df_raw["D"].iloc[0]) ** (1 / (len(df_raw) - 1)) - 1
    c4.metric("CAGR D 2020-2025", f"{digital_cagr * 100:.2f}%")

    fig_data = px.line(
        df_raw,
        x="Năm",
        y=["Y", "K"],
        markers=True,
        title="GDP và vốn tích lũy 2020-2025"
    )
    st.plotly_chart(fig_data, use_container_width=True)

    fig_inputs = px.line(
        df_raw,
        x="Năm",
        y=["L", "D", "AI", "H"],
        markers=True,
        title="L, D, AI, H giai đoạn 2020-2025"
    )
    st.plotly_chart(fig_inputs, use_container_width=True)


# -------------------------
# TAB 2: 1.4.1 TFP
# -------------------------

with tab2:
    st.markdown("## Câu 1.4.1 — Ước lượng TFP A_t")

    st.markdown("Từ hàm sản xuất, giải ngược TFP:")

    st.latex(r"A_t = \frac{Y_t}{K_t^\alpha L_t^\beta D_t^\gamma AI_t^\delta H_t^\theta}")

    tfp_df = model_df[["Năm", "Y", "K", "L", "D", "AI", "H", "TFP_A_t"]].copy()
    st.dataframe(format_df(tfp_df, 4), use_container_width=True)

    c1, c2, c3 = st.columns(3)

    tfp_start = model_df["TFP_A_t"].iloc[0]
    tfp_end = model_df["TFP_A_t"].iloc[-1]
    tfp_change = (tfp_end / tfp_start - 1) * 100
    tfp_cagr = (tfp_end / tfp_start) ** (1 / (len(model_df) - 1)) - 1

    c1.metric("TFP A_2020", f"{tfp_start:.4f}")
    c2.metric("TFP A_2025", f"{tfp_end:.4f}", f"{tfp_change:.2f}% so với 2020")
    c3.metric("CAGR TFP", f"{tfp_cagr * 100:.2f}%/năm")

    fig_tfp = px.line(
        model_df,
        x="Năm",
        y="TFP_A_t",
        markers=True,
        title="Năng suất nhân tố tổng hợp A_t theo năm"
    )
    fig_tfp.update_traces(line=dict(width=4), marker=dict(size=10))
    st.plotly_chart(fig_tfp, use_container_width=True)

    if tfp_end > tfp_start:
        st.success(
            "Bình luận: TFP có xu hướng tăng trong giai đoạn 2020-2025. "
            "Điều này cho thấy phần tăng trưởng không chỉ đến từ mở rộng vốn và lao động, "
            "mà còn đến từ hiệu quả sử dụng nguồn lực, đổi mới công nghệ, số hóa và cải thiện năng lực tổ chức."
        )
    else:
        st.warning(
            "Bình luận: TFP có xu hướng giảm trong giai đoạn 2020-2025. "
            "Điều này hàm ý tăng trưởng có thể đang phụ thuộc nhiều hơn vào mở rộng đầu vào, "
            "trong khi hiệu quả tổng hợp của nền kinh tế chưa được cải thiện rõ."
        )


# -------------------------
# TAB 3: 1.4.2 DỰ BÁO & MAPE
# -------------------------

with tab3:
    st.markdown("## Câu 1.4.2 — Dự báo GDP và tính MAPE")

    st.markdown("Dùng TFP trung bình giai đoạn 2020-2025:")

    st.latex(r"\bar{A} = \frac{1}{T}\sum_{t=1}^{T} A_t")

    st.latex(r"\hat{Y}_t = \bar{A} \cdot K_t^\alpha L_t^\beta D_t^\gamma AI_t^\delta H_t^\theta")

    st.latex(r"MAPE = \frac{1}{T}\sum_t \left|\frac{Y_t - \hat{Y}_t}{Y_t}\right| \times 100")

    c1, c2, c3 = st.columns(3)
    c1.metric("A trung bình", f"{A_mean:.4f}")
    c2.metric("MAPE", f"{mape:.2f}%")
    c3.metric("Sai số TB tuyệt đối", f"{model_df['Sai_số_tuyệt_đối'].mean():,.2f}")

    pred_df = model_df[["Năm", "Y", "Y_dự_báo", "Sai_số_tuyệt_đối", "APE_%"]].copy()
    st.dataframe(format_df(pred_df, 3), use_container_width=True)

    fig_compare = go.Figure()

    fig_compare.add_trace(go.Scatter(
        x=model_df["Năm"],
        y=model_df["Y"],
        mode="lines+markers",
        name="Y thực tế",
        line=dict(width=4)
    ))

    fig_compare.add_trace(go.Scatter(
        x=model_df["Năm"],
        y=model_df["Y_dự_báo"],
        mode="lines+markers",
        name="Y dự báo",
        line=dict(width=4, dash="dash")
    ))

    fig_compare.update_layout(
        title="So sánh GDP thực tế và GDP dự báo",
        xaxis_title="Năm",
        yaxis_title="GDP, nghìn tỷ VND"
    )

    st.plotly_chart(fig_compare, use_container_width=True)

    fig_ape = px.bar(
        model_df,
        x="Năm",
        y="APE_%",
        text=model_df["APE_%"].round(2),
        title="Sai số phần trăm tuyệt đối APE theo năm"
    )
    st.plotly_chart(fig_ape, use_container_width=True)

    if mape < 5:
        st.success(
            f"MAPE = {mape:.2f}% là mức sai số thấp. "
            "Mô hình Cobb-Douglas mở rộng khớp khá tốt với dữ liệu thực tế trong giai đoạn quan sát."
        )
    elif mape < 10:
        st.info(
            f"MAPE = {mape:.2f}% là mức sai số chấp nhận được cho một mô hình vĩ mô đơn giản."
        )
    else:
        st.warning(
            f"MAPE = {mape:.2f}% tương đối cao. "
            "Cần xem xét hiệu chỉnh hệ số, bổ sung biến hoặc dùng mô hình động/phức tạp hơn."
        )


# -------------------------
# TAB 4: 1.4.3 PHÂN RÃ TĂNG TRƯỞNG
# -------------------------

with tab4:
    st.markdown("## Câu 1.4.3 — Phân rã tăng trưởng 2020-2025")

    st.markdown("Phương trình phân rã tăng trưởng:")

    st.latex(
        r"\Delta \ln Y_t = \Delta \ln A_t + "
        r"\alpha \Delta \ln K_t + \beta \Delta \ln L_t + "
        r"\gamma \Delta \ln D_t + \delta \Delta \ln AI_t + \theta \Delta \ln H_t"
    )

    c1, c2 = st.columns(2)
    c1.metric("Tăng trưởng GDP log bình quân", f"{avg_log_growth * 100:.2f}%/năm")
    c2.metric(
        "Tổng tỷ trọng đóng góp",
        f"{growth_summary['Tỷ trọng trong tăng trưởng GDP, %'].sum():.2f}%"
    )

    yearly_display = growth_table.copy()

    for col in [
        "Tăng trưởng GDP log", "TFP", "K", "L", "D", "AI", "H",
        "Tổng đóng góp kiểm tra", "Sai lệch kiểm tra"
    ]:
        yearly_display[col] = yearly_display[col] * 100

    st.markdown("### Bảng phân rã theo từng giai đoạn, đơn vị: điểm % log")
    st.dataframe(format_df(yearly_display, 4), use_container_width=True)

    st.markdown("### Đóng góp bình quân 2020-2025")
    st.dataframe(format_df(growth_summary, 3), use_container_width=True)

    fig_contrib = px.bar(
        growth_summary.sort_values("Tỷ trọng trong tăng trưởng GDP, %"),
        x="Tỷ trọng trong tăng trưởng GDP, %",
        y="Yếu tố",
        orientation="h",
        title="Tỷ trọng đóng góp vào tăng trưởng GDP bình quân 2020-2025"
    )
    st.plotly_chart(fig_contrib, use_container_width=True)

    long_yearly = yearly_display.melt(
        id_vars=["Giai đoạn"],
        value_vars=["TFP", "K", "L", "D", "AI", "H"],
        var_name="Yếu tố",
        value_name="Đóng góp, điểm % log"
    )

    fig_yearly = px.bar(
        long_yearly,
        x="Giai đoạn",
        y="Đóng góp, điểm % log",
        color="Yếu tố",
        barmode="relative",
        title="Phân rã tăng trưởng theo từng năm"
    )
    st.plotly_chart(fig_yearly, use_container_width=True)

    digital_part = growth_summary[growth_summary["Yếu tố"].isin(["D", "AI", "H"])].copy()
    top_digital = digital_part.sort_values(
        "Tỷ trọng trong tăng trưởng GDP, %",
        ascending=False
    ).iloc[0]

    st.success(
        f"Trong ba yếu tố mới D, AI, H, yếu tố đóng góp lớn nhất theo cấu hình hiện tại là "
        f"{top_digital['Yếu tố']} với tỷ trọng khoảng "
        f"{top_digital['Tỷ trọng trong tăng trưởng GDP, %']:.2f}% trong tăng trưởng GDP bình quân."
    )


# -------------------------
# TAB 5: 1.4.4 DỰ BÁO 2030
# -------------------------

with tab5:
    st.markdown("## Câu 1.4.4 — Mô phỏng và dự báo GDP Việt Nam năm 2030")

    st.markdown(
        """
        Kịch bản mặc định theo đề bài:
        - D tăng lên 30%.
        - AI = 100 nghìn doanh nghiệp.
        - H = 35%.
        - K và L tăng 6%/năm.
        - TFP tăng 1,2%/năm.
        """
    )

    forecast_2030 = forecast_df[forecast_df["Năm"] == 2030].iloc[0]
    base_2025 = model_df.sort_values("Năm").iloc[-1]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "GDP dự báo 2030",
        f"{forecast_2030['GDP_dự_báo']:,.1f}",
        "nghìn tỷ VND"
    )

    c2.metric(
        "Tăng so với 2025",
        f"{forecast_2030['Tăng so với 2025, %']:.2f}%"
    )

    c3.metric(
        "D 2030",
        f"{forecast_2030['D']:.1f}%"
    )

    c4.metric(
        "AI 2030",
        f"{forecast_2030['AI']:.1f} nghìn DN"
    )

    st.dataframe(format_df(forecast_df, 3), use_container_width=True)

    fig_forecast = px.line(
        forecast_df,
        x="Năm",
        y="GDP_dự_báo",
        markers=True,
        title="Quỹ đạo GDP dự báo 2025-2030"
    )
    fig_forecast.update_traces(line=dict(width=4), marker=dict(size=10))
    st.plotly_chart(fig_forecast, use_container_width=True)

    fig_inputs_2030 = px.line(
        forecast_df,
        x="Năm",
        y=["K", "L", "D", "AI", "H", "A_TFP"],
        markers=True,
        title="Quỹ đạo đầu vào mô phỏng 2025-2030"
    )
    st.plotly_chart(fig_inputs_2030, use_container_width=True)

    observed_D_cagr = (model_df["D"].iloc[-1] / model_df["D"].iloc[0]) ** (1 / (len(model_df) - 1)) - 1
    required_D_cagr = (target_D / model_df["D"].iloc[-1]) ** (1 / 5) - 1

    c5, c6 = st.columns(2)
    c5.metric("CAGR D quan sát 2020-2025", f"{observed_D_cagr * 100:.2f}%/năm")
    c6.metric("CAGR D cần đạt 2025-2030", f"{required_D_cagr * 100:.2f}%/năm")

    if required_D_cagr <= observed_D_cagr:
        st.success(
            "Mục tiêu D = 30% vào năm 2030 có vẻ khả thi về mặt tốc độ tăng trưởng số hóa, "
            "vì tốc độ cần đạt giai đoạn 2025-2030 không vượt tốc độ quan sát 2020-2025."
        )
    else:
        st.warning(
            "Mục tiêu D = 30% vào năm 2030 đòi hỏi tốc độ tăng nhanh hơn giai đoạn 2020-2025. "
            "Cần thêm đầu tư hạ tầng số, thể chế dữ liệu, nhân lực số và năng lực hấp thụ công nghệ."
        )


# -------------------------
# TAB 6: THẢO LUẬN CHÍNH SÁCH
# -------------------------

with tab6:
    st.markdown("## 1.5. Câu hỏi thảo luận chính sách")

    tfp_start = model_df["TFP_A_t"].iloc[0]
    tfp_end = model_df["TFP_A_t"].iloc[-1]
    tfp_cagr = (tfp_end / tfp_start) ** (1 / (len(model_df) - 1)) - 1

    digital_part = growth_summary[growth_summary["Yếu tố"].isin(["D", "AI", "H"])].copy()
    top_digital = digital_part.sort_values(
        "Tỷ trọng trong tăng trưởng GDP, %",
        ascending=False
    ).iloc[0]

    observed_D_cagr = (model_df["D"].iloc[-1] / model_df["D"].iloc[0]) ** (1 / (len(model_df) - 1)) - 1
    required_D_cagr = (target_D / model_df["D"].iloc[-1]) ** (1 / 5) - 1

    st.markdown("### a) TFP tăng hay giảm? Điều đó nói gì về chất lượng tăng trưởng?")

    if tfp_end > tfp_start:
        st.success(
            f"TFP tăng từ {tfp_start:.4f} năm 2020 lên {tfp_end:.4f} năm 2025, "
            f"tương ứng CAGR khoảng {tfp_cagr * 100:.2f}%/năm. "
            "Điều này cho thấy chất lượng tăng trưởng có cải thiện: cùng với vốn và lao động, "
            "nền kinh tế còn hưởng lợi từ hiệu quả tổng hợp, đổi mới công nghệ, số hóa và năng lực quản trị."
        )
    else:
        st.warning(
            f"TFP giảm từ {tfp_start:.4f} năm 2020 xuống {tfp_end:.4f} năm 2025. "
            "Điều này hàm ý tăng trưởng có thể đang dựa nhiều vào tích lũy đầu vào, "
            "trong khi hiệu quả tổng hợp chưa tăng tương xứng."
        )

    st.markdown("### b) Trong D, AI, H, yếu tố nào đóng góp nhiều nhất?")

    st.info(
        f"Trong ba yếu tố mới D, AI và H, yếu tố đóng góp lớn nhất theo mô hình hiện tại là "
        f"{top_digital['Yếu tố']}, với tỷ trọng khoảng "
        f"{top_digital['Tỷ trọng trong tăng trưởng GDP, %']:.2f}% trong tăng trưởng GDP bình quân. "
        "Kết quả này phụ thuộc vào hai thành phần: tốc độ tăng thực tế của biến đó và hệ số co giãn gắn với biến đó."
    )

    st.markdown("### c) Mục tiêu kinh tế số đạt 30% GDP vào năm 2030 có khả thi không?")

    if required_D_cagr <= observed_D_cagr:
        feasibility_text = (
            "Có tính khả thi tương đối về mặt số học, vì tốc độ tăng D cần đạt trong giai đoạn 2025-2030 "
            "không cao hơn tốc độ quan sát trong giai đoạn 2020-2025."
        )
    else:
        feasibility_text = (
            "Có thách thức lớn hơn, vì tốc độ tăng D cần đạt trong giai đoạn 2025-2030 "
            "cao hơn tốc độ quan sát giai đoạn 2020-2025."
        )

    st.warning(
        f"{feasibility_text} Tuy nhiên, mô hình Cobb-Douglas chỉ cho thấy khả năng kỹ thuật theo dữ liệu đầu vào. "
        "Để mục tiêu này bền vững, cần thêm các ràng buộc chính sách: ngân sách đầu tư số, chất lượng hạ tầng, "
        "năng lực nhân lực số, an toàn dữ liệu, khả năng hấp thụ AI của doanh nghiệp và giảm chênh lệch vùng miền."
    )

    st.markdown("### Tải kết quả")
    output_df = model_df.copy()
    csv = output_df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="⬇️ Tải bảng kết quả Bài 1 dạng CSV",
        data=csv,
        file_name="bai01_cobb_douglas_results.csv",
        mime="text/csv"
    )

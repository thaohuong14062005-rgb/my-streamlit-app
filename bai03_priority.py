# bai03_priority.py
# Bài 3 — Tính chỉ số ưu tiên ngành Priority_i cho 10 ngành Việt Nam
# Module dùng được với streamlit_app.py có cơ chế gọi module.render()

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# =====================================================
# 1. DỮ LIỆU MẶC ĐỊNH THEO ĐỀ BÀI
# =====================================================

def build_default_sector_data() -> pd.DataFrame:
    return pd.DataFrame({
        "Ngành": [
            "Nông-Lâm-Thủy sản",
            "CN chế biến chế tạo",
            "Xây dựng",
            "Khai khoáng",
            "Bán buôn-bán lẻ",
            "Tài chính-Ngân hàng",
            "Logistics-Vận tải",
            "CNTT-Truyền thông",
            "Giáo dục-Đào tạo",
            "Y tế"
        ],
        "Tăng trưởng": [3.27, 9.64, 7.45, -1.20, 7.10, 7.36, 9.93, 7.85, 6.42, 6.85],
        "Năng suất": [103.4, 241.2, 168.8, 1290.5, 145.3, 1072.4, 321.4, 713.8, 205.7, 437.1],
        "Lan tỏa": [0.35, 0.78, 0.42, 0.30, 0.55, 0.85, 0.72, 0.92, 0.65, 0.60],
        "Xuất khẩu": [40.5, 290.9, 2.5, 8.2, 5.5, 1.2, 3.1, 178.0, 0.0, 0.0],
        "Việc làm": [13.20, 11.50, 4.80, 0.30, 7.80, 0.55, 1.95, 0.62, 2.15, 0.75],
        "AI Readiness": [15, 55, 20, 30, 48, 72, 42, 88, 38, 45],
        "Rủi ro TĐH": [18, 42, 25, 55, 38, 52, 35, 28, 22, 18]
    })


# =====================================================
# 2. HÀM ĐỌC VÀ CHUẨN HÓA FILE CSV NẾU CÓ
# =====================================================

def clean_number(x):
    if pd.isna(x):
        return np.nan

    if isinstance(x, (int, float, np.number)):
        return float(x)

    text = str(x).strip().replace(" ", "")

    if "," in text and "." in text:
        # Ví dụ: 1.290,5 -> 1290.5
        text = text.replace(".", "").replace(",", ".")
    elif "," in text:
        # Ví dụ: 7,45 -> 7.45
        text = text.replace(",", ".")

    try:
        return float(text)
    except ValueError:
        return np.nan


def standardize_sector_columns(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    lower_cols = {str(c).strip().lower(): c for c in df.columns}

    candidates = {
        "Ngành": [
            "sector_name_vi", "sector", "nganh", "ngành", "ten_nganh", "tên ngành"
        ],
        "Tăng trưởng": [
            "growth_rate_2024_pct", "growth", "tang_truong", "tăng trưởng"
        ],
        "Năng suất": [
            "productivity", "labor_productivity", "nang_suat", "năng suất",
            "gdp_share_2024_pct"
        ],
        "Lan tỏa": [
            "spillover_coef_0_1", "spillover", "lan_toa", "lan tỏa"
        ],
        "Xuất khẩu": [
            "export_billion_USD", "export", "xuat_khau", "xuất khẩu", "xk"
        ],
        "Việc làm": [
            "labor_million", "employment", "viec_lam", "việc làm", "lao_dong"
        ],
        "AI Readiness": [
            "ai_readiness_0_100", "ai_readiness", "ai readiness"
        ],
        "Rủi ro TĐH": [
            "automation_risk_pct", "automation_risk", "risk", "rui_ro", "rủi ro"
        ]
    }

    selected = {}

    for target, cand_list in candidates.items():
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

        selected[target] = found

    if any(v is None for v in selected.values()):
        missing = [k for k, v in selected.items() if v is None]
        raise ValueError(f"Không nhận diện được các cột: {missing}")

    out = pd.DataFrame({
        "Ngành": df[selected["Ngành"]],
        "Tăng trưởng": df[selected["Tăng trưởng"]],
        "Năng suất": df[selected["Năng suất"]],
        "Lan tỏa": df[selected["Lan tỏa"]],
        "Xuất khẩu": df[selected["Xuất khẩu"]],
        "Việc làm": df[selected["Việc làm"]],
        "AI Readiness": df[selected["AI Readiness"]],
        "Rủi ro TĐH": df[selected["Rủi ro TĐH"]],
    })

    for col in [
        "Tăng trưởng", "Năng suất", "Lan tỏa", "Xuất khẩu",
        "Việc làm", "AI Readiness", "Rủi ro TĐH"
    ]:
        out[col] = out[col].apply(clean_number)

    out = out.dropna().reset_index(drop=True)
    return out


def load_sector_data(uploaded_file=None):
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            return standardize_sector_columns(raw_df), "CSV upload từ giao diện"
        except Exception as e:
            st.warning(f"Không đọc được file upload. App dùng dữ liệu mặc định. Lỗi: {e}")
            return build_default_sector_data(), "Dữ liệu mặc định từ đề bài"

    candidate_paths = [
        Path("vietnam_sectors_2024.csv"),
        Path("data") / "vietnam_sectors_2024.csv"
    ]

    for p in candidate_paths:
        if p.exists():
            try:
                raw_df = pd.read_csv(p)
                return standardize_sector_columns(raw_df), f"File CSV: {p}"
            except Exception as e:
                st.warning(f"Tìm thấy {p}, nhưng không chuẩn hóa được. App dùng dữ liệu mặc định. Lỗi: {e}")
                return build_default_sector_data(), "Dữ liệu mặc định từ đề bài"

    return build_default_sector_data(), "Dữ liệu mặc định từ đề bài"


# =====================================================
# 3. HÀM TÍNH TOÁN PRIORITY
# =====================================================

GOOD_COLS = [
    "Tăng trưởng",
    "Năng suất",
    "Lan tỏa",
    "Xuất khẩu",
    "Việc làm",
    "AI Readiness"
]

RISK_COL = "Rủi ro TĐH"

CRITERIA_FOR_SCORE = [
    "Tăng trưởng_norm",
    "Năng suất_norm",
    "Lan tỏa_norm",
    "Xuất khẩu_norm",
    "Việc làm_norm",
    "AI Readiness_norm",
    "Rủi ro thấp_norm"
]


def norm_good(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    denom = s.max() - s.min()
    if denom == 0:
        return pd.Series(np.ones(len(s)), index=s.index)
    return (s - s.min()) / denom


def norm_bad(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    denom = s.max() - s.min()
    if denom == 0:
        return pd.Series(np.ones(len(s)), index=s.index)
    return (s.max() - s) / denom


def normalize_weights(raw_weights: dict) -> dict:
    total = sum(raw_weights.values())
    if total <= 0:
        raise ValueError("Tổng trọng số phải lớn hơn 0.")
    return {k: v / total for k, v in raw_weights.items()}


def compute_priority(df: pd.DataFrame, raw_weights: dict):
    work = df.copy()

    norm_df = pd.DataFrame()
    norm_df["Ngành"] = work["Ngành"]

    for col in GOOD_COLS:
        norm_df[f"{col}_norm"] = norm_good(work[col])

    # Rủi ro là tiêu chí xấu, nên đảo dấu.
    # Rủi ro càng thấp thì điểm chuẩn hóa càng cao.
    norm_df["Rủi ro thấp_norm"] = norm_bad(work[RISK_COL])

    weights = normalize_weights(raw_weights)

    score = np.zeros(len(norm_df))

    for col in CRITERIA_FOR_SCORE:
        score += norm_df[col] * weights[col]

    result = work.copy()
    result["Priority"] = score
    result["Xếp hạng"] = result["Priority"].rank(ascending=False, method="min").astype(int)

    result = result.sort_values("Priority", ascending=False).reset_index(drop=True)
    result["Xếp hạng"] = range(1, len(result) + 1)

    return result, norm_df, weights


def compute_ai_sensitivity(df: pd.DataFrame, base_raw_weights: dict):
    rows = []
    rank_matrix = {}

    ai_values = np.round(np.arange(0.05, 0.401, 0.05), 2)

    for ai_w in ai_values:
        temp_weights = base_raw_weights.copy()
        temp_weights["AI Readiness_norm"] = float(ai_w)

        result, _, normalized_weights = compute_priority(df, temp_weights)

        top3 = result.head(3)["Ngành"].tolist()

        rows.append({
            "a6 AI Readiness thô": ai_w,
            "a6 sau chuẩn hóa": normalized_weights["AI Readiness_norm"],
            "Top 1": top3[0],
            "Top 2": top3[1],
            "Top 3": top3[2],
            "Top-3": " | ".join(top3)
        })

        for _, r in result.iterrows():
            sector = r["Ngành"]
            rank = int(r["Xếp hạng"])
            rank_matrix.setdefault(sector, {})
            rank_matrix[sector][ai_w] = rank

    sens_df = pd.DataFrame(rows)
    heat_df = pd.DataFrame(rank_matrix).T
    heat_df = heat_df[ai_values]

    return sens_df, heat_df


def compare_weight_scenarios(df: pd.DataFrame):
    scenarios = {
        "Mặc định": {
            "Tăng trưởng_norm": 0.15,
            "Năng suất_norm": 0.15,
            "Lan tỏa_norm": 0.20,
            "Xuất khẩu_norm": 0.15,
            "Việc làm_norm": 0.10,
            "AI Readiness_norm": 0.20,
            "Rủi ro thấp_norm": 0.15,
        },
        "Định hướng tăng trưởng": {
            "Tăng trưởng_norm": 0.25,
            "Năng suất_norm": 0.25,
            "Lan tỏa_norm": 0.10,
            "Xuất khẩu_norm": 0.20,
            "Việc làm_norm": 0.05,
            "AI Readiness_norm": 0.10,
            "Rủi ro thấp_norm": 0.05,
        },
        "Định hướng bao trùm": {
            "Tăng trưởng_norm": 0.10,
            "Năng suất_norm": 0.10,
            "Lan tỏa_norm": 0.25,
            "Xuất khẩu_norm": 0.05,
            "Việc làm_norm": 0.25,
            "AI Readiness_norm": 0.10,
            "Rủi ro thấp_norm": 0.15,
        }
    }

    all_rows = []

    for scenario_name, weights in scenarios.items():
        result, _, normalized_weights = compute_priority(df, weights)

        for _, row in result.iterrows():
            all_rows.append({
                "Kịch bản trọng số": scenario_name,
                "Ngành": row["Ngành"],
                "Priority": row["Priority"],
                "Xếp hạng": row["Xếp hạng"]
            })

    compare_df = pd.DataFrame(all_rows)

    top3_rows = []

    for scenario_name in scenarios.keys():
        temp = compare_df[compare_df["Kịch bản trọng số"] == scenario_name]
        temp = temp.sort_values("Xếp hạng").head(3)

        top3_rows.append({
            "Kịch bản trọng số": scenario_name,
            "Top 1": temp.iloc[0]["Ngành"],
            "Top 2": temp.iloc[1]["Ngành"],
            "Top 3": temp.iloc[2]["Ngành"],
            "Top-3": " | ".join(temp["Ngành"].tolist())
        })

    top3_df = pd.DataFrame(top3_rows)

    return compare_df, top3_df, scenarios


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
            <h1>📊 Bài 3 — Tính chỉ số ưu tiên ngành Priorityᵢ cho 10 ngành Việt Nam</h1>
            <p>
            Module này chuẩn hóa dữ liệu 10 ngành, tính chỉ số ưu tiên chuyển đổi số và AI,
            xếp hạng ngành, phân tích độ nhạy trọng số AI Readiness và so sánh các định hướng chính sách.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "Risk là tiêu chí xấu nên được đảo dấu thành “Rủi ro thấp_norm”. "
        "Điểm Priority càng cao thì ngành càng được ưu tiên đẩy mạnh chuyển đổi số và AI."
    )

    with st.sidebar:
        st.markdown("### ⚙️ Tham số Bài 3")

        uploaded_file = st.file_uploader(
            "Upload vietnam_sectors_2024.csv nếu có",
            type=["csv"],
            key="bai3_upload"
        )

        st.markdown("#### Trọng số thô mặc định")

        w_growth = st.number_input("a1 - Tăng trưởng", value=0.15, step=0.01, format="%.2f")
        w_productivity = st.number_input("a2 - Năng suất", value=0.15, step=0.01, format="%.2f")
        w_spillover = st.number_input("a3 - Lan tỏa", value=0.20, step=0.01, format="%.2f")
        w_export = st.number_input("a4 - Xuất khẩu", value=0.15, step=0.01, format="%.2f")
        w_employment = st.number_input("a5 - Việc làm", value=0.10, step=0.01, format="%.2f")
        w_ai = st.number_input("a6 - AI Readiness", value=0.20, step=0.01, format="%.2f")
        w_risk = st.number_input("a7 - Giảm rủi ro TĐH", value=0.15, step=0.01, format="%.2f")

    raw_weights = {
        "Tăng trưởng_norm": w_growth,
        "Năng suất_norm": w_productivity,
        "Lan tỏa_norm": w_spillover,
        "Xuất khẩu_norm": w_export,
        "Việc làm_norm": w_employment,
        "AI Readiness_norm": w_ai,
        "Rủi ro thấp_norm": w_risk,
    }

    df, data_source = load_sector_data(uploaded_file)
    result, norm_df, normalized_weights = compute_priority(df, raw_weights)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📌 Mô hình",
        "📄 Dữ liệu",
        "3.4.1 Chuẩn hóa",
        "3.4.2 Priority & Xếp hạng",
        "3.4.3 Độ nhạy AI",
        "3.4.4 So sánh trọng số"
    ])

    # =====================================================
    # TAB 1 — MÔ HÌNH
    # =====================================================

    with tab1:
        st.header("1. Mô hình toán học")

        st.markdown("### Công thức Priorityᵢ")

        st.latex(
            r"Priority_i = a_1 Growth_i + a_2 Productivity_i + a_3 Spillover_i "
            r"+ a_4 Export_i + a_5 Employment_i + a_6 AIReadiness_i - a_7 Risk_i"
        )

        st.markdown("### Chuẩn hóa min-max")

        st.latex(
            r"\tilde{x}_i = \frac{x_i - \min(x)}{\max(x) - \min(x)}"
        )

        st.markdown("### Đảo dấu tiêu chí rủi ro")

        st.latex(
            r"\widetilde{RiskLow}_i = \frac{\max(Risk) - Risk_i}{\max(Risk) - \min(Risk)}"
        )

        st.markdown("### Trọng số đang sử dụng")

        weights_df = pd.DataFrame({
            "Tiêu chí": list(raw_weights.keys()),
            "Trọng số thô": list(raw_weights.values()),
            "Trọng số sau chuẩn hóa tổng = 1": [normalized_weights[k] for k in raw_weights.keys()]
        })

        st.dataframe(weights_df.round(4), use_container_width=True)

        st.warning(
            "Bộ trọng số mặc định trong đề có tổng thô bằng 1,10. "
            "Vì vậy trong app này, toàn bộ trọng số được chuẩn hóa lại để tổng bằng 1 trước khi tính Priority."
        )

    # =====================================================
    # TAB 2 — DỮ LIỆU
    # =====================================================

    with tab2:
        st.header("2. Dữ liệu 10 ngành Việt Nam 2024")

        st.caption(f"Nguồn dữ liệu đang dùng: {data_source}")

        st.dataframe(df, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Số ngành", f"{len(df)}")
        c2.metric("Tăng trưởng cao nhất", f"{df['Tăng trưởng'].max():.2f}%")
        c3.metric("AI Readiness cao nhất", f"{df['AI Readiness'].max():.0f}")
        c4.metric("Rủi ro TĐH cao nhất", f"{df['Rủi ro TĐH'].max():.0f}%")

        fig_growth = px.bar(
            df.sort_values("Tăng trưởng", ascending=False),
            x="Ngành",
            y="Tăng trưởng",
            title="Tăng trưởng theo ngành năm 2024",
            text="Tăng trưởng"
        )
        fig_growth.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig_growth, use_container_width=True)

        fig_ai_risk = px.scatter(
            df,
            x="AI Readiness",
            y="Rủi ro TĐH",
            size="Việc làm",
            color="Ngành",
            hover_name="Ngành",
            title="AI Readiness, rủi ro tự động hóa và quy mô việc làm"
        )
        st.plotly_chart(fig_ai_risk, use_container_width=True)

    # =====================================================
    # TAB 3 — CHUẨN HÓA
    # =====================================================

    with tab3:
        st.header("Câu 3.4.1 — Chuẩn hóa min-max 7 tiêu chí")

        st.markdown(
            "Sáu tiêu chí tốt gồm: Tăng trưởng, Năng suất, Lan tỏa, Xuất khẩu, Việc làm, AI Readiness. "
            "Tiêu chí Rủi ro TĐH được đảo dấu thành Rủi ro thấp."
        )

        st.dataframe(norm_df.round(4), use_container_width=True)

        heat_data = norm_df.set_index("Ngành")[CRITERIA_FOR_SCORE]

        fig_heat = px.imshow(
            heat_data,
            text_auto=".2f",
            aspect="auto",
            title="Ma trận chuẩn hóa min-max của 10 ngành",
            labels=dict(x="Tiêu chí chuẩn hóa", y="Ngành", color="Điểm")
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # =====================================================
    # TAB 4 — PRIORITY
    # =====================================================

    with tab4:
        st.header("Câu 3.4.2 — Tính Priorityᵢ và xếp hạng 10 ngành")

        display_cols = [
            "Xếp hạng",
            "Ngành",
            "Priority",
            "Tăng trưởng",
            "Năng suất",
            "Lan tỏa",
            "Xuất khẩu",
            "Việc làm",
            "AI Readiness",
            "Rủi ro TĐH"
        ]

        st.dataframe(result[display_cols].round(4), use_container_width=True)

        top3 = result.head(3)

        c1, c2, c3 = st.columns(3)

        c1.metric("Top 1", top3.iloc[0]["Ngành"], f"Priority = {top3.iloc[0]['Priority']:.4f}")
        c2.metric("Top 2", top3.iloc[1]["Ngành"], f"Priority = {top3.iloc[1]['Priority']:.4f}")
        c3.metric("Top 3", top3.iloc[2]["Ngành"], f"Priority = {top3.iloc[2]['Priority']:.4f}")

        fig_rank = px.bar(
            result.sort_values("Priority", ascending=True),
            x="Priority",
            y="Ngành",
            orientation="h",
            text=result.sort_values("Priority", ascending=True)["Priority"].round(4),
            title="Xếp hạng ngành theo Priority"
        )
        st.plotly_chart(fig_rank, use_container_width=True)

        st.success(
            f"Theo bộ trọng số hiện tại, 3 ngành ưu tiên cao nhất là: "
            f"{top3.iloc[0]['Ngành']}, {top3.iloc[1]['Ngành']} và {top3.iloc[2]['Ngành']}."
        )

    # =====================================================
    # TAB 5 — ĐỘ NHẠY AI
    # =====================================================

    with tab5:
        st.header("Câu 3.4.3 — Phân tích độ nhạy trọng số AI Readiness")

        st.markdown(
            "Thay đổi trọng số thô a6 của AI Readiness từ 0,05 đến 0,40 với bước 0,05. "
            "Sau mỗi lần thay đổi, toàn bộ trọng số được chuẩn hóa lại để tổng bằng 1."
        )

        sens_df, heat_df = compute_ai_sensitivity(df, raw_weights)

        st.markdown("### Top-3 theo từng mức trọng số AI")
        st.dataframe(sens_df, use_container_width=True)

        base_top3 = set(sens_df.iloc[0][["Top 1", "Top 2", "Top 3"]].tolist())
        sens_df["Top-3 thay đổi so với a6=0.05"] = sens_df.apply(
            lambda r: set([r["Top 1"], r["Top 2"], r["Top 3"]]) != base_top3,
            axis=1
        )

        changed_count = sens_df["Top-3 thay đổi so với a6=0.05"].sum()

        if changed_count == 0:
            st.success(
                "Top-3 ổn định khi thay đổi trọng số AI Readiness trong khoảng 0,05 đến 0,40."
            )
        else:
            st.warning(
                f"Top-3 có thay đổi ở {changed_count} mức trọng số AI. "
                "Điều này cho thấy kết quả nhạy với ưu tiên chính sách dành cho AI Readiness."
            )

        st.markdown("### Heatmap xếp hạng theo trọng số AI")

        fig_sens = px.imshow(
            heat_df,
            text_auto=True,
            aspect="auto",
            title="Heatmap thứ hạng ngành khi thay đổi trọng số AI Readiness",
            labels=dict(x="Trọng số AI Readiness thô", y="Ngành", color="Thứ hạng")
        )
        st.plotly_chart(fig_sens, use_container_width=True)

    # =====================================================
    # TAB 6 — SO SÁNH TRỌNG SỐ
    # =====================================================

    with tab6:
        st.header("Câu 3.4.4 — So sánh hai bộ trọng số chính sách")

        compare_df, top3_df, scenarios = compare_weight_scenarios(df)

        st.markdown("### Top-3 theo từng định hướng chính sách")
        st.dataframe(top3_df, use_container_width=True)

        st.markdown("### Bảng điểm và thứ hạng theo từng kịch bản trọng số")
        st.dataframe(compare_df.round(4), use_container_width=True)

        fig_compare = px.bar(
            compare_df.sort_values(["Kịch bản trọng số", "Xếp hạng"]),
            x="Ngành",
            y="Priority",
            color="Kịch bản trọng số",
            barmode="group",
            title="So sánh Priority theo ba bộ trọng số"
        )
        fig_compare.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig_compare, use_container_width=True)

        pivot_rank = compare_df.pivot(
            index="Ngành",
            columns="Kịch bản trọng số",
            values="Xếp hạng"
        )

        fig_rank_scenario = px.imshow(
            pivot_rank,
            text_auto=True,
            aspect="auto",
            title="Heatmap thứ hạng ngành theo từng định hướng trọng số",
            labels=dict(x="Kịch bản trọng số", y="Ngành", color="Thứ hạng")
        )
        st.plotly_chart(fig_rank_scenario, use_container_width=True)

        default_top3 = top3_df[top3_df["Kịch bản trọng số"] == "Mặc định"]["Top-3"].iloc[0]
        growth_top3 = top3_df[top3_df["Kịch bản trọng số"] == "Định hướng tăng trưởng"]["Top-3"].iloc[0]
        inclusive_top3 = top3_df[top3_df["Kịch bản trọng số"] == "Định hướng bao trùm"]["Top-3"].iloc[0]

        st.markdown("### Nhận xét chính sách")

        st.info(
            f"Với bộ trọng số mặc định, Top-3 là: {default_top3}. "
            f"Với định hướng tăng trưởng, Top-3 là: {growth_top3}. "
            f"Với định hướng bao trùm, Top-3 là: {inclusive_top3}."
        )

        mining_row = result[result["Ngành"] == "Khai khoáng"]

        if len(mining_row) > 0:
            mining_rank = int(mining_row["Xếp hạng"].iloc[0])
            mining_priority = float(mining_row["Priority"].iloc[0])

            st.warning(
                f"Ngành Khai khoáng có năng suất rất cao nhưng xếp hạng {mining_rank} "
                f"với Priority = {mining_priority:.4f}. "
                "Nguyên nhân thường là tăng trưởng thấp hoặc âm, việc làm nhỏ, lan tỏa hạn chế, "
                "rủi ro tự động hóa cao và mức độ phù hợp với mục tiêu chuyển đổi số bao trùm không lớn."
            )

        st.success(
            "Về quản trị chính sách, trọng số không nên chỉ do chuyên gia kỹ thuật quyết định. "
            "Một hội đồng chính sách có tham vấn doanh nghiệp, địa phương, chuyên gia lao động, "
            "chuyên gia công nghệ và đại diện xã hội sẽ giúp bộ trọng số có tính chính danh cao hơn."
        )


# Alias để streamlit_app.py có thể gọi module.run() nếu cần
def run():
    render()

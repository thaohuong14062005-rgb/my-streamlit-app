# bai03_priority.py

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


BRAND = "#053151"

MULTI_COLORS = {
    "Tăng trưởng": "#053151",
    "GDP share": "#E76F51",
    "Lan tỏa": "#2A9D8F",
    "Xuất khẩu": "#F4A261",
    "Việc làm": "#8E44AD",
    "AI readiness": "#457B9D",
    "Risk đảo chiều": "#A7C957",
    "Mặc định": "#053151",
    "Định hướng tăng trưởng": "#E76F51",
    "Định hướng bao trùm": "#2A9D8F",
}


# =========================================================
# BÀI 3 — XẾP HẠNG ƯU TIÊN CHUYỂN ĐỔI SỐ VÀ AI THEO NGÀNH
# =========================================================


FEATURES = [
    "growth_rate_2024_pct",
    "gdp_share_2024_pct",
    "spillover_coef_0_1",
    "export_billion_USD",
    "labor_million",
    "ai_readiness_0_100",
    "automation_risk_pct",
]

FEATURE_LABELS = {
    "growth_rate_2024_pct": "Tăng trưởng",
    "gdp_share_2024_pct": "GDP share",
    "spillover_coef_0_1": "Lan tỏa",
    "export_billion_USD": "Xuất khẩu",
    "labor_million": "Việc làm",
    "ai_readiness_0_100": "AI readiness",
    "automation_risk_pct": "Risk đảo chiều",
}

DEFAULT_WEIGHTS_RAW = np.array(
    [0.15, 0.15, 0.20, 0.15, 0.10, 0.20, 0.15],
    dtype=float,
)

GROWTH_WEIGHTS = np.array(
    [0.25, 0.20, 0.10, 0.20, 0.05, 0.15, 0.05],
    dtype=float,
)

INCLUSIVE_WEIGHTS = np.array(
    [0.10, 0.10, 0.20, 0.05, 0.25, 0.10, 0.20],
    dtype=float,
)


def fallback_sector_data():
    return pd.DataFrame(
        {
            "sector_id": list(range(1, 11)),
            "sector_name_vi": [
                "Nông nghiệp",
                "Công nghiệp chế biến chế tạo",
                "Xây dựng",
                "Bán buôn bán lẻ",
                "Vận tải kho bãi",
                "Thông tin và truyền thông",
                "Tài chính ngân hàng",
                "Bất động sản",
                "Giáo dục đào tạo",
                "Y tế và trợ giúp xã hội",
            ],
            "gdp_share_2024_pct": [11.9, 24.2, 6.2, 9.7, 4.6, 4.1, 5.4, 3.9, 4.0, 3.2],
            "growth_rate_2024_pct": [3.3, 8.4, 7.6, 8.2, 9.1, 9.8, 7.3, 2.5, 4.1, 5.8],
            "labor_million": [13.8, 11.5, 4.8, 7.2, 3.1, 1.3, 0.9, 0.8, 1.7, 1.5],
            "export_billion_USD": [55.0, 310.0, 2.5, 5.2, 3.5, 8.0, 1.0, 0.2, 0.1, 0.1],
            "digital_index_0_100": [45, 62, 48, 61, 55, 86, 82, 50, 58, 63],
            "ai_readiness_0_100": [35, 68, 42, 60, 52, 88, 84, 49, 54, 61],
            "fdi_attraction_billion_USD": [1.2, 18.5, 2.3, 4.0, 2.2, 5.1, 1.8, 3.2, 0.5, 0.7],
            "spillover_coef_0_1": [0.52, 0.86, 0.58, 0.64, 0.61, 0.89, 0.78, 0.45, 0.67, 0.70],
            "automation_risk_pct": [34, 48, 43, 38, 46, 22, 31, 27, 18, 20],
            "rd_intensity_pct": [0.15, 0.62, 0.18, 0.22, 0.10, 1.10, 0.55, 0.08, 0.30, 0.35],
        }
    )


def find_file(filename):
    possible_paths = [
        filename,
        f"./{filename}",
        f"data/{filename}",
        f"./data/{filename}",
        f"datasets/{filename}",
        f"./datasets/{filename}",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    return None


def load_data():
    path = find_file("vietnam_sectors_2024.csv")

    if path is None:
        return fallback_sector_data()

    try:
        df = pd.read_csv(path)

        required_cols = ["sector_name_vi"] + FEATURES
        missing = [col for col in required_cols if col not in df.columns]

        if missing:
            return fallback_sector_data()

        for col in FEATURES:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=required_cols).reset_index(drop=True)

        if len(df) == 0:
            return fallback_sector_data()

        return df

    except Exception:
        return fallback_sector_data()


def norm_good(series):
    min_v = series.min()
    max_v = series.max()

    if np.isclose(max_v, min_v):
        return pd.Series(np.zeros(len(series)), index=series.index)

    return (series - min_v) / (max_v - min_v)


def norm_bad(series):
    min_v = series.min()
    max_v = series.max()

    if np.isclose(max_v, min_v):
        return pd.Series(np.zeros(len(series)), index=series.index)

    return (max_v - series) / (max_v - min_v)


def normalize_matrix(df):
    out = pd.DataFrame(index=df.index)
    out["Ngành"] = df["sector_name_vi"]

    for col in FEATURES:
        label = FEATURE_LABELS[col]

        if col == "automation_risk_pct":
            out[label] = norm_bad(df[col])
        else:
            out[label] = norm_good(df[col])

    return out


def normalized_weights(raw_weights):
    weights = np.array(raw_weights, dtype=float)
    total = weights.sum()

    if np.isclose(total, 0):
        return np.ones_like(weights) / len(weights)

    return weights / total


def score_priority(norm_df, raw_weights):
    labels = [FEATURE_LABELS[col] for col in FEATURES]
    weights = normalized_weights(raw_weights)

    scores = norm_df[labels].values @ weights

    ranked = pd.DataFrame(
        {
            "Ngành": norm_df["Ngành"],
            "Priority": scores,
        }
    )

    ranked["Rank"] = ranked["Priority"].rank(ascending=False, method="min").astype(int)
    ranked = ranked.sort_values(["Priority", "Ngành"], ascending=[False, True]).reset_index(drop=True)

    return ranked, weights


def build_weight_table(raw_weights, normalized):
    labels = [FEATURE_LABELS[col] for col in FEATURES]

    return pd.DataFrame(
        {
            "Tiêu chí": labels,
            "Trọng số gốc": raw_weights,
            "Trọng số chuẩn hóa": normalized,
        }
    )


def sensitivity_ai_weight(norm_df):
    rows = []

    for ai_weight in np.arange(0.05, 0.401, 0.05):
        raw = DEFAULT_WEIGHTS_RAW.copy()
        raw[5] = ai_weight

        ranked, _ = score_priority(norm_df, raw)

        for _, row in ranked.iterrows():
            rows.append(
                {
                    "a6_AI": round(float(ai_weight), 2),
                    "Ngành": row["Ngành"],
                    "Priority": row["Priority"],
                    "Rank": int(row["Rank"]),
                }
            )

    return pd.DataFrame(rows)


def compare_weight_sets(norm_df):
    default_ranked, _ = score_priority(norm_df, DEFAULT_WEIGHTS_RAW)
    growth_ranked, _ = score_priority(norm_df, GROWTH_WEIGHTS)
    inclusive_ranked, _ = score_priority(norm_df, INCLUSIVE_WEIGHTS)

    default_ranked["Bộ trọng số"] = "Mặc định"
    growth_ranked["Bộ trọng số"] = "Định hướng tăng trưởng"
    inclusive_ranked["Bộ trọng số"] = "Định hướng bao trùm"

    combined = pd.concat(
        [default_ranked, growth_ranked, inclusive_ranked],
        ignore_index=True,
    )

    weights_df = pd.DataFrame(
        {
            "Tiêu chí": [FEATURE_LABELS[col] for col in FEATURES],
            "Mặc định": normalized_weights(DEFAULT_WEIGHTS_RAW),
            "Định hướng tăng trưởng": normalized_weights(GROWTH_WEIGHTS),
            "Định hướng bao trùm": normalized_weights(INCLUSIVE_WEIGHTS),
        }
    )

    return combined, weights_df


def make_styled_table(df, decimals=3):
    show_df = df.copy()

    format_dict = {}
    for col in show_df.columns:
        if pd.api.types.is_numeric_dtype(show_df[col]):
            if str(col).lower() in ["rank", "xếp hạng"]:
                format_dict[col] = "{:.0f}"
            else:
                format_dict[col] = "{:." + str(decimals) + "f}"

    styler = show_df.style.format(format_dict)

    try:
        styler = styler.hide(axis="index")
    except Exception:
        try:
            styler = styler.hide_index()
        except Exception:
            pass

    styler = styler.set_properties(
        **{
            "background-color": "#ffffff",
            "color": BRAND,
            "border": f"1px solid {BRAND}",
            "padding": "10px 12px",
            "font-size": "16px",
        }
    )

    styler = styler.set_table_styles(
        [
            {
                "selector": "thead th",
                "props": [
                    ("background-color", "#ffffff"),
                    ("color", BRAND),
                    ("font-weight", "800"),
                    ("border", f"1px solid {BRAND}"),
                    ("padding", "12px 12px"),
                    ("font-size", "16px"),
                    ("text-align", "left"),
                ],
            },
            {
                "selector": "tbody td",
                "props": [
                    ("background-color", "#ffffff"),
                    ("color", BRAND),
                    ("border", f"1px solid {BRAND}"),
                ],
            },
            {
                "selector": "table",
                "props": [
                    ("border-collapse", "collapse"),
                    ("width", "100%"),
                    ("background-color", "#ffffff"),
                    ("color", BRAND),
                ],
            },
        ]
    )

    return styler


def show_table(df, decimals=3):
    st.table(make_styled_table(df, decimals=decimals))


def style_base_fig(fig, height=430):
    fig.update_layout(
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color=BRAND, size=15),
        title_font=dict(color=BRAND, size=20),
        xaxis=dict(
            showline=True,
            linecolor=BRAND,
            tickfont=dict(color=BRAND),
            title_font=dict(color=BRAND),
        ),
        yaxis=dict(
            showline=True,
            linecolor=BRAND,
            gridcolor="rgba(5,49,81,0.18)",
            tickfont=dict(color=BRAND),
            title_font=dict(color=BRAND),
        ),
        legend=dict(font=dict(color=BRAND)),
    )
    return fig


def render():
    st.title("Bài 3. Xếp hạng ưu tiên chuyển đổi số và AI theo ngành")
    st.caption("Chuẩn hóa min-max, tính Priority, phân tích độ nhạy AI readiness và so sánh bộ trọng số")

    df = load_data()
    norm_df = normalize_matrix(df)

    ranked_default, weights_default = score_priority(norm_df, DEFAULT_WEIGHTS_RAW)
    weight_table = build_weight_table(DEFAULT_WEIGHTS_RAW, weights_default)

    sens_df = sensitivity_ai_weight(norm_df)
    compare_df, compare_weights_df = compare_weight_sets(norm_df)

    top3_default = ranked_default.head(3)
    top1 = top3_default.iloc[0]

    tabs = st.tabs(
        [
            "Ma trận chuẩn hóa",
            "Xếp hạng ngành",
            "Độ nhạy AI",
            "So sánh trọng số",
            "Thảo luận Chính sách",
        ]
    )

    # =====================================================
    # 3.4.1
    # =====================================================
    with tabs[0]:
      

        c1, c2, c3 = st.columns(3)
        c1.metric("Số ngành", f"{len(df)}")
        c2.metric("Số tiêu chí", "7")
        c3.metric("Risk", "Đảo chiều")

        st.subheader("Dữ liệu gốc")

        raw_display = df[["sector_name_vi"] + FEATURES].rename(
            columns={
                "sector_name_vi": "Ngành",
                "growth_rate_2024_pct": "Tăng trưởng",
                "gdp_share_2024_pct": "GDP share",
                "spillover_coef_0_1": "Lan tỏa",
                "export_billion_USD": "Xuất khẩu",
                "labor_million": "Việc làm",
                "ai_readiness_0_100": "AI readiness",
                "automation_risk_pct": "Risk",
            }
        )

        show_table(raw_display, decimals=3)

        st.subheader("Ma trận đã chuẩn hóa")
        show_table(norm_df, decimals=4)

        st.success(
            "Đã chuẩn hóa min-max 7 tiêu chí. Riêng Automation Risk được đảo chiều để rủi ro thấp có điểm cao hơn."
        )

    # =====================================================
    # 3.4.2
    # =====================================================
    with tabs[1]:
       

        c1, c2, c3 = st.columns(3)
        c1.metric("Ngành top 1", top1["Ngành"])
        c2.metric("Priority top 1", f"{top1['Priority']:.3f}")
        c3.metric("Bộ trọng số", "Mặc định")

        st.subheader("Trọng số sử dụng")
        show_table(weight_table, decimals=4)

        st.subheader("Xếp hạng 10 ngành")
        ranked_display = ranked_default[["Rank", "Ngành", "Priority"]].rename(
            columns={"Rank": "Xếp hạng"}
        )
        show_table(ranked_display, decimals=4)

        rank_plot_df = ranked_default.sort_values("Priority", ascending=True).copy()
        rank_plot_df["Nhãn"] = rank_plot_df["Priority"].round(3)

        fig = px.bar(
            rank_plot_df,
            x="Priority",
            y="Ngành",
            orientation="h",
            text="Nhãn",
            title="Priority theo ngành",
        )
        fig.update_traces(
            marker_color=BRAND,
            textposition="outside",
            textfont=dict(color=BRAND),
            hovertemplate="<b>%{y}</b><br>Priority: %{x:.4f}<extra></extra>",
        )
        fig.update_layout(
            xaxis_title="Priority",
            yaxis_title="Ngành",
        )
        style_base_fig(fig, height=540)
        st.plotly_chart(fig, use_container_width=True)

        st.success(
            f"Ba ngành ưu tiên cao nhất theo bộ trọng số mặc định là: "
            f"{', '.join(top3_default['Ngành'].tolist())}."
        )

    # =====================================================
    # 3.4.3
    # =====================================================
    with tabs[2]:
       

        top3_rows = []

        for ai_weight in sorted(sens_df["a6_AI"].unique()):
            sub = sens_df[sens_df["a6_AI"] == ai_weight].sort_values("Rank").head(3)

            top3_rows.append(
                {
                    "a6 AI": ai_weight,
                    "Top 1": sub.iloc[0]["Ngành"],
                    "Top 2": sub.iloc[1]["Ngành"],
                    "Top 3": sub.iloc[2]["Ngành"],
                }
            )

        top3_df = pd.DataFrame(top3_rows)
        unique_top3_sets = top3_df[["Top 1", "Top 2", "Top 3"]].drop_duplicates()
        change_text = "Có thay đổi" if len(unique_top3_sets) > 1 else "Không thay đổi"

        c1, c2, c3 = st.columns(3)
        c1.metric("a6 nhỏ nhất", "0.05")
        c2.metric("a6 lớn nhất", "0.40")
        c3.metric("Top-3", change_text)

        show_table(top3_df, decimals=2)

        rank_matrix = sens_df.pivot(index="Ngành", columns="a6_AI", values="Rank")
        sector_order = ranked_default["Ngành"].tolist()
        rank_matrix = rank_matrix.reindex(sector_order)

        fig_heat = px.imshow(
            rank_matrix,
            aspect="auto",
            color_continuous_scale="RdYlGn_r",
            title="Heatmap xếp hạng khi thay đổi trọng số AI readiness",
            labels=dict(x="Trọng số a6", y="Ngành", color="Rank"),
        )
        fig_heat.update_traces(
            text=rank_matrix.values,
            texttemplate="%{text:.0f}",
            hovertemplate="<b>%{y}</b><br>a6=%{x}<br>Rank=%{z:.0f}<extra></extra>",
        )
        fig_heat.update_layout(
            height=620,
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color=BRAND),
            title_font=dict(color=BRAND, size=20),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("### Kiểm thử nhanh một giá trị a6")

        custom_ai_weight = st.slider(
            "Trọng số a6 - AI readiness",
            0.05,
            0.40,
            0.20,
            0.05,
        )

        custom_raw = DEFAULT_WEIGHTS_RAW.copy()
        custom_raw[5] = custom_ai_weight

        custom_ranked, custom_weights = score_priority(norm_df, custom_raw)

        c4, c5 = st.columns(2)
        c4.metric("a6 đang chọn", f"{custom_ai_weight:.2f}")
        c5.metric("Top 1", custom_ranked.iloc[0]["Ngành"])

        show_table(
            custom_ranked[["Rank", "Ngành", "Priority"]].rename(columns={"Rank": "Xếp hạng"}),
            decimals=4,
        )

        with st.expander("Xem ma trận Priority theo a6"):
            priority_matrix = sens_df.pivot(index="Ngành", columns="a6_AI", values="Priority")
            priority_matrix = priority_matrix.reindex(sector_order)
            show_table(priority_matrix.reset_index(), decimals=4)

        st.success(
            f"Kết quả độ nhạy cho thấy top-3 khi thay đổi trọng số AI readiness: {change_text.lower()}."
        )

    # =====================================================
    # 3.4.4
    # =====================================================
    with tabs[3]:
      

        st.subheader("Hai bộ trọng số")
        show_table(compare_weights_df, decimals=3)

        top_compare = compare_df[compare_df["Rank"] <= 3].copy()
        top_compare = top_compare.sort_values(["Bộ trọng số", "Rank"])

        st.subheader("Top-3 theo từng bộ trọng số")
        top_compare_display = top_compare[["Bộ trọng số", "Rank", "Ngành", "Priority"]].rename(
            columns={"Rank": "Xếp hạng"}
        )
        show_table(top_compare_display, decimals=4)

        top_growth = top_compare[
            top_compare["Bộ trọng số"] == "Định hướng tăng trưởng"
        ]["Ngành"].tolist()

        top_inclusive = top_compare[
            top_compare["Bộ trọng số"] == "Định hướng bao trùm"
        ]["Ngành"].tolist()

        c1, c2 = st.columns(2)
        c1.metric("Top-3 tăng trưởng", " / ".join(top_growth))
        c2.metric("Top-3 bao trùm", " / ".join(top_inclusive))

        union_sectors = sorted(set(top_growth + top_inclusive))

        plot_compare = compare_df[
            (compare_df["Ngành"].isin(union_sectors))
            & (compare_df["Bộ trọng số"].isin(["Định hướng tăng trưởng", "Định hướng bao trùm"]))
        ].copy()

        fig = px.bar(
            plot_compare,
            x="Ngành",
            y="Priority",
            color="Bộ trọng số",
            barmode="group",
            title="Priority của các ngành top-3 dưới hai định hướng",
            color_discrete_map=MULTI_COLORS,
        )
        fig.update_layout(
            xaxis_title="Ngành",
            yaxis_title="Priority",
        )
        fig.update_traces(
            marker_line_color="white",
            marker_line_width=1,
        )
        style_base_fig(fig, height=470)
        st.plotly_chart(fig, use_container_width=True)

        if set(top_growth) == set(top_inclusive):
            st.success("Hai bộ trọng số cho ra cùng nhóm top-3, dù thứ tự có thể khác.")
        else:
            st.success("Hai bộ trọng số tạo ra khác biệt trong nhóm top-3, phản ánh ưu tiên chính sách khác nhau.")

    # =====================================================
    # 3.5
    # =====================================================
    with tabs[4]:
     

        top3_names = top3_default["Ngành"].tolist()
        top3_text = ", ".join(top3_names)

        st.markdown("### a) Ba ngành nào nên ưu tiên chuyển đổi số và AI trước?")

        st.success(
            f"Theo bộ trọng số mặc định, ba ngành nên ưu tiên là: **{top3_text}**."
        )

        st.markdown(
            f"""
            Kết quả này phản ánh cách tiếp cận đa tiêu chí: ngành được ưu tiên không chỉ vì tăng trưởng cao,
            mà còn vì có khả năng lan tỏa, quy mô xuất khẩu, năng lực sẵn sàng AI, đóng góp việc làm
            và mức rủi ro tự động hóa đã được xử lý theo hướng giảm rủi ro.

            Về định hướng chính sách, nhóm ngành top-3 nhìn chung phù hợp với tinh thần ưu tiên khoa học,
            công nghệ, đổi mới sáng tạo và chuyển đổi số. Nếu một ngành vừa có năng lực AI cao, vừa có lan tỏa lớn
            và đóng góp đáng kể cho tăng trưởng hoặc xuất khẩu, thì đầu tư chuyển đổi số vào ngành đó có thể tạo hiệu ứng
            kéo theo cho nhiều khu vực khác của nền kinh tế.

            Tuy nhiên, kết quả xếp hạng không nên được hiểu là danh sách cố định. Nó phụ thuộc vào bộ trọng số,
            chất lượng dữ liệu và mục tiêu chính sách trong từng thời kỳ.
            """
        )

        st.markdown("### b) Vì sao Khai khoáng có năng suất cao nhưng không nằm trong nhóm ưu tiên?")

        st.markdown(
            """
            Khai khoáng có thể có năng suất lao động hoặc giá trị gia tăng trên lao động rất cao,
            nhưng điều đó không tự động khiến ngành này trở thành ưu tiên hàng đầu cho chuyển đổi số và AI.

            Có ba lý do chính:

            - **Tác động lan tỏa có thể thấp hơn** so với các ngành như chế biến chế tạo, thông tin truyền thông,
            logistics hoặc tài chính. Một ngành có năng suất cao nhưng ít liên kết chuỗi giá trị sẽ tạo hiệu ứng lan tỏa hạn chế.

            - **Quy mô việc làm không lớn.** Nếu mục tiêu chính sách bao gồm cả bao trùm xã hội và chuyển đổi lực lượng lao động,
            các ngành sử dụng nhiều lao động hoặc có khả năng nâng cấp kỹ năng diện rộng thường được ưu tiên hơn.

            - **Rủi ro môi trường và tính bền vững.** Khai khoáng có thể tạo giá trị kinh tế lớn,
            nhưng không nhất thiết phù hợp với định hướng tăng trưởng xanh, đổi mới sáng tạo và kinh tế số dài hạn.

            Vì vậy, năng suất cao chỉ là một tiêu chí. Khi xét thêm lan tỏa, AI readiness, xuất khẩu, việc làm
            và rủi ro tự động hóa, Khai khoáng có thể không nằm trong nhóm ưu tiên cao nhất.
            """
        )

        st.markdown("### c) Bộ trọng số nên do ai quyết định?")

        st.markdown(
            """
            Bộ trọng số không nên chỉ do một nhóm kỹ thuật hoặc một cơ quan hành chính quyết định.
            Trọng số phản ánh ưu tiên phát triển, phân bổ nguồn lực và đánh đổi chính sách,
            nên đây là vấn đề vừa mang tính kỹ thuật, vừa mang tính chính trị - xã hội.

            **Chuyên gia kỹ thuật** có vai trò quan trọng trong việc thiết kế chỉ số, kiểm định dữ liệu,
            lựa chọn phương pháp chuẩn hóa và đánh giá độ nhạy. Họ giúp bảo đảm mô hình nhất quán,
            minh bạch và tránh sai lệch kỹ thuật.

            **Hội đồng chính sách** cần quyết định định hướng ưu tiên dựa trên chiến lược phát triển quốc gia,
            nguồn lực ngân sách, năng lực thực thi và các mục tiêu dài hạn như tăng trưởng, đổi mới sáng tạo,
            an sinh xã hội và phát triển bền vững.

            **Quy trình đối thoại công khai** lại rất cần thiết để bảo đảm tính chính danh.
            Khi trọng số ảnh hưởng đến việc ngành nào được ưu tiên, doanh nghiệp, người lao động, địa phương,
            hiệp hội ngành nghề và người dân cần có cơ hội phản hồi. Điều này giúp giảm rủi ro thiên lệch,
            tăng tính chấp nhận xã hội và làm cho chính sách dễ thực thi hơn.

            Cách tốt nhất là kết hợp cả ba: chuyên gia xây dựng phương pháp, hội đồng chính sách xác lập mục tiêu,
            và đối thoại công khai để kiểm tra tính hợp lý, minh bạch và chính danh của bộ trọng số.

            **Kết luận:** bộ trọng số nên được quyết định thông qua một cơ chế governance đa bên,
            vừa dựa trên bằng chứng kỹ thuật, vừa phản ánh ưu tiên phát triển và có sự tham gia của các bên liên quan.
            """
        )

        st.info(
            f"Kết luận: theo bộ trọng số mặc định, top-3 là **{top3_text}**; "
            "tuy nhiên thứ hạng có thể thay đổi khi trọng số AI readiness hoặc định hướng chính sách thay đổi."
        )


def run():
    render()

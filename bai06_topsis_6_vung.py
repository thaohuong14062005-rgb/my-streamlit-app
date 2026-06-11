# bai06_topsis_6_vung.py

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


BRAND = "#053151"

MULTI_COLORS = {
    "TOPSIS chuyên gia": "#053151",
    "TOPSIS Entropy": "#E76F51",
    "TOPSIS AHP": "#2A9D8F",
    "GRDP/người": "#053151",
    "FDI": "#E76F51",
    "Digital Index": "#2A9D8F",
    "AI Readiness": "#F4A261",
    "LĐ đào tạo": "#8E44AD",
    "R&D": "#457B9D",
    "Internet": "#A7C957",
    "Gini": "#BC4749",
}


# =========================================================
# BÀI 6 — TOPSIS XẾP HẠNG VÙNG CHO TRUNG TÂM AI
# =========================================================


CRITERIA = [
    "grdp_per_capita_million_VND",
    "fdi_registered_billion_USD",
    "digital_index_0_100",
    "ai_readiness_0_100",
    "trained_labor_pct",
    "rd_intensity_pct",
    "internet_penetration_pct",
    "gini_coef",
]

CRITERIA_LABELS = {
    "grdp_per_capita_million_VND": "GRDP/người",
    "fdi_registered_billion_USD": "FDI",
    "digital_index_0_100": "Digital Index",
    "ai_readiness_0_100": "AI Readiness",
    "trained_labor_pct": "LĐ đào tạo",
    "rd_intensity_pct": "R&D",
    "internet_penetration_pct": "Internet",
    "gini_coef": "Gini",
}

IS_BENEFIT = np.array([True, True, True, True, True, True, True, False])

EXPERT_WEIGHTS = np.array(
    [0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10],
    dtype=float,
)

AHP_BASE_PRIORITIES = np.array(
    [2, 2, 3, 4, 3, 3, 1, 2],
    dtype=float,
)


def fallback_region_data():
    return pd.DataFrame(
        {
            "region_code": ["NMM", "RRD", "NCC", "CH", "SE", "MD"],
            "region_name_vi": [
                "Trung du và miền núi Bắc Bộ",
                "Đồng bằng sông Hồng",
                "Bắc Trung Bộ và Duyên hải miền Trung",
                "Tây Nguyên",
                "Đông Nam Bộ",
                "Đồng bằng sông Cửu Long",
            ],
            "grdp_per_capita_million_VND": [68, 142, 74, 58, 196, 72],
            "fdi_registered_billion_USD": [3.2, 17.8, 5.4, 1.1, 23.5, 2.6],
            "digital_index_0_100": [48, 78, 55, 42, 82, 52],
            "ai_readiness_0_100": [41, 77, 49, 36, 84, 45],
            "trained_labor_pct": [24, 42, 29, 22, 45, 25],
            "rd_intensity_pct": [0.22, 0.85, 0.35, 0.18, 0.95, 0.25],
            "internet_penetration_pct": [72, 89, 76, 68, 91, 74],
            "gini_coef": [0.39, 0.36, 0.37, 0.40, 0.38, 0.35],
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
    path = find_file("vietnam_regions_2024.csv")

    if path is None:
        return fallback_region_data()

    try:
        df = pd.read_csv(path)

        if "region_name_vi" not in df.columns:
            return fallback_region_data()

        missing = [col for col in CRITERIA if col not in df.columns]

        if missing:
            return fallback_region_data()

        for col in CRITERIA:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["region_name_vi"] + CRITERIA).reset_index(drop=True)

        if len(df) == 0:
            return fallback_region_data()

        if "region_code" not in df.columns:
            df["region_code"] = [f"R{i + 1}" for i in range(len(df))]

        return df

    except Exception:
        return fallback_region_data()


def normalize_weights(weights):
    weights = np.array(weights, dtype=float)
    total = weights.sum()

    if np.isclose(total, 0):
        return np.ones_like(weights) / len(weights)

    return weights / total


def topsis_score(df, weights):
    X = df[CRITERIA].values.astype(float)
    w = normalize_weights(weights)

    denom = np.sqrt((X ** 2).sum(axis=0))
    denom = np.where(np.isclose(denom, 0), 1, denom)

    R = X / denom
    V = R * w

    A_star = np.where(IS_BENEFIT, V.max(axis=0), V.min(axis=0))
    A_neg = np.where(IS_BENEFIT, V.min(axis=0), V.max(axis=0))

    S_star = np.sqrt(((V - A_star) ** 2).sum(axis=1))
    S_neg = np.sqrt(((V - A_neg) ** 2).sum(axis=1))

    C_star = S_neg / (S_star + S_neg + 1e-12)

    out = pd.DataFrame(
        {
            "Vùng": df["region_name_vi"],
            "TOPSIS_score": C_star,
        }
    )

    out["Rank"] = out["TOPSIS_score"].rank(ascending=False, method="min").astype(int)
    out = out.sort_values(["TOPSIS_score", "Vùng"], ascending=[False, True]).reset_index(drop=True)

    return out, R, V


def minmax_for_entropy(df):
    X = df[CRITERIA].values.astype(float)
    Xn = np.zeros_like(X, dtype=float)

    for j in range(X.shape[1]):
        col = X[:, j]
        min_v = col.min()
        max_v = col.max()

        if np.isclose(max_v, min_v):
            Xn[:, j] = 0
        else:
            if IS_BENEFIT[j]:
                Xn[:, j] = (col - min_v) / (max_v - min_v)
            else:
                Xn[:, j] = (max_v - col) / (max_v - min_v)

    return Xn


def entropy_weights(df):
    Xn = minmax_for_entropy(df)
    Xp = Xn + 1e-12

    P = Xp / Xp.sum(axis=0, keepdims=True)
    k = 1.0 / np.log(len(Xp))

    E = -k * np.nansum(P * np.log(P + 1e-12), axis=0)
    d = 1 - E

    if np.isclose(d.sum(), 0):
        w = np.ones(Xn.shape[1]) / Xn.shape[1]
    else:
        w = d / d.sum()

    entropy_df = pd.DataFrame(
        {
            "Tiêu chí": [CRITERIA_LABELS[c] for c in CRITERIA],
            "Entropy E": E,
            "Độ phân biệt d": d,
            "Trọng số Entropy": w,
        }
    )

    return w, entropy_df


def ahp_weights_from_priorities(priorities):
    priorities = np.array(priorities, dtype=float)
    n = len(priorities)

    pairwise = priorities.reshape(-1, 1) / priorities.reshape(1, -1)

    eigvals, eigvecs = np.linalg.eig(pairwise)
    max_idx = np.argmax(eigvals.real)
    lambda_max = eigvals.real[max_idx]
    weights = eigvecs[:, max_idx].real
    weights = np.abs(weights)
    weights = weights / weights.sum()

    ci = (lambda_max - n) / (n - 1) if n > 1 else 0

    ri_table = {
        1: 0.00,
        2: 0.00,
        3: 0.58,
        4: 0.90,
        5: 1.12,
        6: 1.24,
        7: 1.32,
        8: 1.41,
        9: 1.45,
        10: 1.49,
    }

    ri = ri_table.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0

    pairwise_df = pd.DataFrame(
        pairwise,
        index=[CRITERIA_LABELS[c] for c in CRITERIA],
        columns=[CRITERIA_LABELS[c] for c in CRITERIA],
    )

    weight_df = pd.DataFrame(
        {
            "Tiêu chí": [CRITERIA_LABELS[c] for c in CRITERIA],
            "Ưu tiên AHP": priorities,
            "Trọng số AHP": weights,
        }
    )

    return weights, pairwise_df, weight_df, float(lambda_max), float(ci), float(cr)


def sensitivity_ai_weight(df):
    rows = []

    for ai_weight in np.arange(0.10, 0.401, 0.05):
        w = EXPERT_WEIGHTS.copy()
        w[3] = ai_weight
        w = normalize_weights(w)

        ranked, _, _ = topsis_score(df, w)

        for _, row in ranked.iterrows():
            rows.append(
                {
                    "w_AI": round(float(ai_weight), 2),
                    "Vùng": row["Vùng"],
                    "TOPSIS_score": row["TOPSIS_score"],
                    "Rank": int(row["Rank"]),
                }
            )

    return pd.DataFrame(rows)


def compare_rankings(expert_rank, entropy_rank, ahp_rank):
    expert = expert_rank[["Vùng", "TOPSIS_score", "Rank"]].rename(
        columns={"TOPSIS_score": "Score chuyên gia", "Rank": "Rank chuyên gia"}
    )
    entropy = entropy_rank[["Vùng", "TOPSIS_score", "Rank"]].rename(
        columns={"TOPSIS_score": "Score Entropy", "Rank": "Rank Entropy"}
    )
    ahp = ahp_rank[["Vùng", "TOPSIS_score", "Rank"]].rename(
        columns={"TOPSIS_score": "Score AHP", "Rank": "Rank AHP"}
    )

    out = expert.merge(entropy, on="Vùng", how="left").merge(ahp, on="Vùng", how="left")
    out["Thay đổi Rank Entropy"] = out["Rank Entropy"] - out["Rank chuyên gia"]
    out["Thay đổi Rank AHP"] = out["Rank AHP"] - out["Rank chuyên gia"]

    return out.sort_values("Rank chuyên gia").reset_index(drop=True)


def weight_table(expert_w, entropy_w, ahp_w):
    return pd.DataFrame(
        {
            "Tiêu chí": [CRITERIA_LABELS[c] for c in CRITERIA],
            "Chuyên gia": normalize_weights(expert_w),
            "Entropy": normalize_weights(entropy_w),
            "AHP": normalize_weights(ahp_w),
        }
    )


def make_styled_table(df, decimals=3):
    show_df = df.copy()

    format_dict = {}
    for col in show_df.columns:
        if pd.api.types.is_numeric_dtype(show_df[col]):
            if str(col).lower() in ["rank", "rank chuyên gia", "rank entropy", "rank ahp"]:
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
    st.title("Bài 6. TOPSIS xếp hạng vùng triển khai AI")
    st.caption("TOPSIS chuyên gia, Entropy weight, độ nhạy AI readiness và AHP đơn giản")

    df = load_data()

    expert_rank, R, V = topsis_score(df, EXPERT_WEIGHTS)
    entropy_w, entropy_df = entropy_weights(df)
    entropy_rank, _, _ = topsis_score(df, entropy_w)

    ahp_w, pairwise_df, ahp_weight_df, lambda_max, ci, cr = ahp_weights_from_priorities(AHP_BASE_PRIORITIES)
    ahp_rank, _, _ = topsis_score(df, ahp_w)

    compare_df = compare_rankings(expert_rank, entropy_rank, ahp_rank)
    weights_compare = weight_table(EXPERT_WEIGHTS, entropy_w, ahp_w)

    sens_df = sensitivity_ai_weight(df)

    top_expert = expert_rank.iloc[0]
    top3_expert = expert_rank.head(3)["Vùng"].tolist()

    tabs = st.tabs(
        [
            "6.4.1 TOPSIS",
            "6.4.2 Entropy",
            "6.4.3 Độ nhạy AI",
            "6.4.4 AHP",
            "6.5 Chính sách",
        ]
    )

    # =====================================================
    # 6.4.1
    # =====================================================
    with tabs[0]:
        st.header("6.4.1. TOPSIS với trọng số chuyên gia")

        c1, c2, c3 = st.columns(3)
        c1.metric("Số vùng", f"{len(df)}")
        c2.metric("Số tiêu chí", "8")
        c3.metric("Vùng dẫn đầu", top_expert["Vùng"])

        raw_display = df[["region_name_vi"] + CRITERIA].rename(
            columns={
                "region_name_vi": "Vùng",
                "grdp_per_capita_million_VND": "GRDP/người",
                "fdi_registered_billion_USD": "FDI",
                "digital_index_0_100": "Digital Index",
                "ai_readiness_0_100": "AI Readiness",
                "trained_labor_pct": "LĐ đào tạo",
                "rd_intensity_pct": "R&D",
                "internet_penetration_pct": "Internet",
                "gini_coef": "Gini",
            }
        )

        st.subheader("Dữ liệu đầu vào")
        show_table(raw_display, decimals=3)

        expert_weight_df = pd.DataFrame(
            {
                "Tiêu chí": [CRITERIA_LABELS[c] for c in CRITERIA],
                "Trọng số chuyên gia": normalize_weights(EXPERT_WEIGHTS),
                "Loại tiêu chí": ["Lợi ích" if b else "Chi phí" for b in IS_BENEFIT],
            }
        )

        st.subheader("Trọng số chuyên gia")
        show_table(expert_weight_df, decimals=3)

        rank_display = expert_rank.rename(columns={"Rank": "Xếp hạng"})
        st.subheader("Xếp hạng TOPSIS")
        show_table(rank_display, decimals=4)

        plot_df = expert_rank.sort_values("TOPSIS_score", ascending=True).copy()
        plot_df["Nhãn"] = plot_df["TOPSIS_score"].round(3)

        fig = px.bar(
            plot_df,
            x="TOPSIS_score",
            y="Vùng",
            orientation="h",
            text="Nhãn",
            title="TOPSIS score theo vùng - trọng số chuyên gia",
        )
        fig.update_traces(
            marker_color=BRAND,
            textposition="outside",
            textfont=dict(color=BRAND),
            hovertemplate="<b>%{y}</b><br>C*: %{x:.4f}<extra></extra>",
        )
        fig.update_layout(xaxis_title="Cᵢ*", yaxis_title="Vùng")
        style_base_fig(fig, height=460)
        st.plotly_chart(fig, use_container_width=True)

        st.success(
            f"Vùng dẫn đầu theo TOPSIS chuyên gia là {top_expert['Vùng']} với C* = {top_expert['TOPSIS_score']:.4f}."
        )

    # =====================================================
    # 6.4.2
    # =====================================================
    with tabs[1]:
        st.header("6.4.2. TOPSIS với trọng số Entropy")

        c1, c2, c3 = st.columns(3)
        c1.metric("Top 1 chuyên gia", expert_rank.iloc[0]["Vùng"])
        c2.metric("Top 1 Entropy", entropy_rank.iloc[0]["Vùng"])

        biggest_change_row = compare_df.iloc[
            compare_df["Thay đổi Rank Entropy"].abs().idxmax()
        ]
        c3.metric("Thay đổi lớn nhất", biggest_change_row["Vùng"])

        st.subheader("Trọng số Entropy")
        show_table(entropy_df, decimals=4)

        st.subheader("So sánh xếp hạng")
        compare_display = compare_df[
            ["Vùng", "Rank chuyên gia", "Rank Entropy", "Thay đổi Rank Entropy", "Score chuyên gia", "Score Entropy"]
        ]
        show_table(compare_display, decimals=4)

        plot_compare = compare_display.melt(
            id_vars="Vùng",
            value_vars=["Rank chuyên gia", "Rank Entropy"],
            var_name="Phương pháp",
            value_name="Rank",
        )

        fig = px.bar(
            plot_compare,
            x="Vùng",
            y="Rank",
            color="Phương pháp",
            barmode="group",
            title="So sánh rank TOPSIS chuyên gia và Entropy",
            color_discrete_map={
                "Rank chuyên gia": "#053151",
                "Rank Entropy": "#E76F51",
            },
        )
        fig.update_layout(
            xaxis_title="Vùng",
            yaxis_title="Rank",
            yaxis_autorange="reversed",
        )
        style_base_fig(fig, height=460)
        st.plotly_chart(fig, use_container_width=True)

        st.success(
            f"Vùng có thay đổi xếp hạng lớn nhất khi dùng Entropy là {biggest_change_row['Vùng']}."
        )

    # =====================================================
    # 6.4.3
    # =====================================================
    with tabs[2]:
        st.header("6.4.3. Độ nhạy theo trọng số AI readiness")

        top3_rows = []

        for ai_w in sorted(sens_df["w_AI"].unique()):
            sub = sens_df[sens_df["w_AI"] == ai_w].sort_values("Rank").head(3)

            top3_rows.append(
                {
                    "w_AI": ai_w,
                    "Top 1": sub.iloc[0]["Vùng"],
                    "Top 2": sub.iloc[1]["Vùng"],
                    "Top 3": sub.iloc[2]["Vùng"],
                }
            )

        top3_df = pd.DataFrame(top3_rows)
        unique_top3 = top3_df[["Top 1", "Top 2", "Top 3"]].drop_duplicates()
        stable_text = "Ổn định" if len(unique_top3) == 1 else "Có thay đổi"

        c1, c2, c3 = st.columns(3)
        c1.metric("w_AI nhỏ nhất", "0.10")
        c2.metric("w_AI lớn nhất", "0.40")
        c3.metric("Top-3", stable_text)

        show_table(top3_df, decimals=2)

        rank_matrix = sens_df.pivot(index="Vùng", columns="w_AI", values="Rank")
        rank_matrix = rank_matrix.reindex(expert_rank["Vùng"].tolist())

        fig_heat = px.imshow(
            rank_matrix,
            aspect="auto",
            color_continuous_scale="RdYlGn_r",
            title="Heatmap rank khi thay đổi trọng số AI readiness",
            labels=dict(x="w_AI", y="Vùng", color="Rank"),
        )
        fig_heat.update_traces(
            text=rank_matrix.values,
            texttemplate="%{text:.0f}",
            hovertemplate="<b>%{y}</b><br>w_AI=%{x}<br>Rank=%{z:.0f}<extra></extra>",
        )
        fig_heat.update_layout(
            height=560,
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color=BRAND),
            title_font=dict(color=BRAND, size=20),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("### Kiểm thử nhanh w_AI")

        custom_ai_w = st.slider(
            "Trọng số AI readiness",
            0.10,
            0.40,
            0.20,
            0.05,
        )

        custom_w = EXPERT_WEIGHTS.copy()
        custom_w[3] = custom_ai_w
        custom_rank, _, _ = topsis_score(df, custom_w)

        c4, c5 = st.columns(2)
        c4.metric("w_AI đang chọn", f"{custom_ai_w:.2f}")
        c5.metric("Top 1", custom_rank.iloc[0]["Vùng"])

        show_table(custom_rank.rename(columns={"Rank": "Xếp hạng"}), decimals=4)

        st.success(
            f"Kết quả độ nhạy cho thấy top-3 theo w_AI: {stable_text.lower()}."
        )

    # =====================================================
    # 6.4.4
    # =====================================================
    with tabs[3]:
        st.header("6.4.4. AHP đơn giản và so sánh với TOPSIS")

        c1, c2, c3 = st.columns(3)
        c1.metric("λ max", f"{lambda_max:.3f}")
        c2.metric("CI", f"{ci:.4f}")
        c3.metric("CR", f"{cr:.4f}")

        st.subheader("Trọng số AHP")
        show_table(ahp_weight_df, decimals=4)

        st.subheader("Ma trận so sánh cặp AHP")
        show_table(pairwise_df.reset_index().rename(columns={"index": "Tiêu chí"}), decimals=3)

        st.subheader("So sánh trọng số")
        show_table(weights_compare, decimals=4)

        st.subheader("So sánh xếp hạng TOPSIS chuyên gia, Entropy và AHP")
        show_table(compare_df, decimals=4)

        top_union = set()
        top_union.update(expert_rank.head(3)["Vùng"].tolist())
        top_union.update(entropy_rank.head(3)["Vùng"].tolist())
        top_union.update(ahp_rank.head(3)["Vùng"].tolist())

        plot_rows = []

        for method, ranked in [
            ("TOPSIS chuyên gia", expert_rank),
            ("TOPSIS Entropy", entropy_rank),
            ("TOPSIS AHP", ahp_rank),
        ]:
            sub = ranked[ranked["Vùng"].isin(top_union)].copy()
            sub["Phương pháp"] = method
            plot_rows.append(sub)

        plot_df = pd.concat(plot_rows, ignore_index=True)

        fig = px.bar(
            plot_df,
            x="Vùng",
            y="TOPSIS_score",
            color="Phương pháp",
            barmode="group",
            title="So sánh TOPSIS score theo ba cách xác định trọng số",
            color_discrete_map=MULTI_COLORS,
        )
        fig.update_traces(marker_line_color="white", marker_line_width=1)
        fig.update_layout(xaxis_title="Vùng", yaxis_title="TOPSIS score")
        style_base_fig(fig, height=480)
        st.plotly_chart(fig, use_container_width=True)

        if cr < 0.10:
            st.success("Ma trận AHP có CR < 0.10, mức nhất quán chấp nhận được.")
        else:
            st.warning("CR >= 0.10, nên rà soát lại ma trận so sánh cặp AHP.")

    # =====================================================
    # 6.5
    # =====================================================
    with tabs[4]:
        st.header("6.5. Thảo luận chính sách")

        top3_text = ", ".join(top3_expert)
        entropy_top = entropy_rank.iloc[0]["Vùng"]

        biggest_change = compare_df.iloc[
            compare_df["Thay đổi Rank Entropy"].abs().idxmax()
        ]

        st.markdown("### a) Vùng nào dẫn đầu theo TOPSIS chuyên gia?")

        st.success(
            f"Vùng dẫn đầu là **{top_expert['Vùng']}**. Top-3 gồm: **{top3_text}**."
        )

        st.markdown(
            """
            Vùng dẫn đầu theo TOPSIS là vùng có khoảng cách gần nhất với nghiệm lý tưởng:
            GRDP/người, FDI, chỉ số số hóa, AI readiness, lao động qua đào tạo, R&D và internet cao,
            đồng thời Gini thấp hơn tương đối.

            Đây là ứng viên mạnh để triển khai trung tâm AI quốc gia đầu tiên vì có năng lực hấp thụ công nghệ,
            hệ sinh thái doanh nghiệp và hạ tầng số tốt hơn. Tuy nhiên, quyết định chính sách không nên chỉ dựa vào điểm TOPSIS.
            Cần xét thêm năng lực đất đai, hạ tầng điện, an ninh dữ liệu, liên kết đại học - doanh nghiệp
            và vai trò lan tỏa vùng.
            """
        )

        st.markdown("### b) Khi dùng Entropy, vùng nào thay đổi xếp hạng lớn nhất? Vì sao?")

        st.success(
            f"Vùng thay đổi xếp hạng lớn nhất là **{biggest_change['Vùng']}**, "
            f"với thay đổi rank = **{biggest_change['Thay đổi Rank Entropy']:.0f}**."
        )

        st.markdown(
            f"""
            Trọng số Entropy dựa trên độ phân tán dữ liệu: tiêu chí nào khác biệt mạnh giữa các vùng
            sẽ được gán trọng số cao hơn. Vì vậy, nếu một vùng mạnh ở tiêu chí có độ phân tán lớn,
            vùng đó có thể tăng hạng; ngược lại, nếu vùng mạnh ở tiêu chí ít phân tán hoặc bị giảm trọng số,
            thứ hạng có thể giảm.

            Do đó, sự thay đổi xếp hạng không nhất thiết là mô hình sai,
            mà phản ánh khác biệt giữa **ưu tiên chủ quan của chuyên gia** và **trọng số khách quan theo dữ liệu**.
            """
        )

        st.markdown("### c) Tương quan cao giữa AI Readiness và Internet penetration ảnh hưởng thế nào?")

        st.markdown(
            """
            TOPSIS giả định các tiêu chí đóng góp độc lập tuyến tính vào khoảng cách đến nghiệm lý tưởng.
            Nếu AI Readiness và Internet penetration tương quan rất cao, mô hình có thể bị **đếm trùng thông tin**.
            Một vùng mạnh về hạ tầng internet thường cũng có điểm AI readiness cao, nên lợi thế đó bị khuếch đại hai lần.

            Hệ quả là các vùng phát triển sẵn có có thể được xếp hạng cao hơn mức hợp lý,
            trong khi các vùng đang cần đầu tư bắt kịp có thể bị đánh giá thấp.

            Có thể xử lý bằng một số cách:

            - Kiểm tra ma trận tương quan trước khi chấm điểm.
            - Gộp các tiêu chí tương quan cao thành một chỉ số tổng hợp.
            - Giảm trọng số của một trong hai tiêu chí.
            - Dùng PCA hoặc factor analysis để rút gọn tiêu chí.
            - Thực hiện phân tích độ nhạy để xem top-3 có ổn định không.
            """
        )

        st.markdown("### d) Chọn 3 vùng xây dựng trung tâm AI lớn")

        st.success(
            f"Dựa trên TOPSIS chuyên gia, có thể chọn 3 vùng: **{top3_text}**."
        )

        st.markdown(
            """
            Nếu mục tiêu là tối đa hóa năng lực triển khai nhanh, top-3 theo TOPSIS là lựa chọn hợp lý
            vì các vùng này có nền tảng tốt hơn về số hóa, FDI, nhân lực và AI readiness.

            Tuy nhiên, nếu đặt trong mục tiêu quốc gia về 3 trung tâm AI lớn,
            cần cân nhắc thêm tiêu chí địa - chính trị. Ba trung tâm không nên chỉ tập trung ở các vùng đã phát triển,
            mà cần bảo đảm khả năng lan tỏa Bắc - Trung - Nam, an ninh dữ liệu, khả năng kết nối vùng,
            quốc phòng - an ninh, cân bằng phát triển và giảm khoảng cách số.

            Vì vậy, kết quả TOPSIS nên được xem là cơ sở kỹ thuật ban đầu.
            Quyết định cuối cùng có thể điều chỉnh bằng các tiêu chí bổ sung như:
            vị trí chiến lược, vai trò liên kết vùng, hạ tầng điện - nước cho trung tâm dữ liệu,
            khả năng thu hút nhân tài, và tính đại diện vùng miền.

            **Kết luận:** nên chọn top-3 theo TOPSIS làm danh sách đề xuất ban đầu,
            sau đó hiệu chỉnh bằng ràng buộc địa - chính trị và mục tiêu cân bằng phát triển quốc gia.
            """
        )

        st.info(
            f"Kết luận: TOPSIS chuyên gia chọn top-3 là **{top3_text}**; "
            f"Entropy có thể làm thay đổi thứ hạng do nhấn mạnh tiêu chí có độ phân tán cao."
        )


def run():
    render()

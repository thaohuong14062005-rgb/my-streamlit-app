# bai09_lao_dong_ai.py

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

try:
    import pulp
    PULP_AVAILABLE = True
except Exception:
    PULP_AVAILABLE = False


BRAND = "#053151"

MULTI_COLORS = {
    "AI": "#053151",
    "Đào tạo lại": "#E76F51",
    "NewJob": "#2A9D8F",
    "Upgrade": "#F4A261",
    "Displaced": "#BC4749",
    "RetrainCap": "#457B9D",
    "NetJob": "#053151",
    "Mô hình gốc": "#053151",
    "Có ràng buộc 5%": "#E76F51",
}


# =========================================================
# BÀI 9 — AI, VIỆC LÀM VÀ ĐÀO TẠO LẠI LAO ĐỘNG
# =========================================================


SECTORS = list(range(1, 9))

SECTOR_NAMES = {
    1: "Nông - Lâm - Thủy sản",
    2: "CN chế biến chế tạo",
    3: "Xây dựng",
    4: "Bán buôn bán lẻ",
    5: "Tài chính - Ngân hàng",
    6: "Vận tải - Logistics",
    7: "Thông tin - Truyền thông",
    8: "Giáo dục - Y tế - Dịch vụ xã hội",
}

RISK = {
    1: 0.18,
    2: 0.42,
    3: 0.25,
    4: 0.38,
    5: 0.52,
    6: 0.35,
    7: 0.28,
    8: 0.22,
}

A1 = {
    1: 8.5,
    2: 32.5,
    3: 12.8,
    4: 22.4,
    5: 45.8,
    6: 28.5,
    7: 62.5,
    8: 18.5,
}

B1 = {
    1: 45,
    2: 28,
    3: 35,
    4: 32,
    5: 22,
    6: 30,
    7: 20,
    8: 55,
}

C1 = {
    1: 5.2,
    2: 62.4,
    3: 18.5,
    4: 48.2,
    5: 72.5,
    6: 42.8,
    7: 32.5,
    8: 12.5,
}

D1 = {
    1: 50,
    2: 32,
    3: 42,
    4: 38,
    5: 26,
    6: 36,
    7: 24,
    8: 62,
}

# Quy mô lao động giả định, đơn vị: nghìn lao động.
# Dùng cho ràng buộc mở rộng: DisplacedJob_i <= 5% * L_i.
LABOR_THOUSAND = {
    1: 13800,
    2: 11500,
    3: 4800,
    4: 7200,
    5: 900,
    6: 3100,
    7: 1300,
    8: 3200,
}


def sector_data():
    rows = []

    for i in SECTORS:
        displaced_coef = C1[i] * RISK[i]
        net_ai_coef = A1[i] - displaced_coef

        rows.append(
            {
                "Ngành": i,
                "Tên ngành": SECTOR_NAMES[i],
                "Risk": RISK[i],
                "a1 - NewJob/AI": A1[i],
                "b1 - Upgrade/H": B1[i],
                "c1 - Displaced/AI": C1[i],
                "d1 - RetrainCap/H": D1[i],
                "Hệ số Displaced thực": displaced_coef,
                "NetJob biên của AI": net_ai_coef,
                "L_i": LABOR_THOUSAND[i],
                "5% L_i": 0.05 * LABOR_THOUSAND[i],
            }
        )

    return pd.DataFrame(rows)


def solve_job_model(
    total_budget=30000.0,
    ai_min_budget=0.0,
    max_loss_constraint=False,
    max_loss_pct=0.05,
):
    if not PULP_AVAILABLE:
        return {
            "success": False,
            "status": "PuLP chưa được cài",
            "objective": np.nan,
            "x_ai": {},
            "x_h": {},
        }

    model = pulp.LpProblem("VN_AI_Jobs_Transition", pulp.LpMaximize)

    x_ai = pulp.LpVariable.dicts("x_AI", SECTORS, lowBound=0)
    x_h = pulp.LpVariable.dicts("x_H", SECTORS, lowBound=0)

    net_expr = {}

    for i in SECTORS:
        new_job = A1[i] * x_ai[i]
        upgrade = B1[i] * x_h[i]
        displaced = C1[i] * RISK[i] * x_ai[i]

        net_expr[i] = new_job + upgrade - displaced

    model += pulp.lpSum(net_expr[i] for i in SECTORS), "Total_NetJob"

    model += (
        pulp.lpSum(x_ai[i] + x_h[i] for i in SECTORS) <= total_budget,
        "Total_budget",
    )

    if ai_min_budget > 0:
        model += (
            pulp.lpSum(x_ai[i] for i in SECTORS) >= ai_min_budget,
            "AI_min_budget",
        )

    for i in SECTORS:
        new_job = A1[i] * x_ai[i]
        upgrade = B1[i] * x_h[i]
        displaced = C1[i] * RISK[i] * x_ai[i]
        retrain_cap = D1[i] * x_h[i]

        model += new_job + upgrade - displaced >= 0, f"NetJob_nonnegative_sector_{i}"
        model += displaced <= retrain_cap, f"Retrain_capacity_sector_{i}"

        if max_loss_constraint:
            model += (
                displaced <= max_loss_pct * LABOR_THOUSAND[i],
                f"Max_5pct_displaced_sector_{i}",
            )

    solver = pulp.PULP_CBC_CMD(msg=False)
    status_code = model.solve(solver)
    status = pulp.LpStatus[status_code]

    if status != "Optimal":
        return {
            "success": False,
            "status": status,
            "objective": np.nan,
            "x_ai": {},
            "x_h": {},
        }

    x_ai_val = {i: float(pulp.value(x_ai[i])) for i in SECTORS}
    x_h_val = {i: float(pulp.value(x_h[i])) for i in SECTORS}

    return {
        "success": True,
        "status": status,
        "objective": float(pulp.value(model.objective)),
        "x_ai": x_ai_val,
        "x_h": x_h_val,
    }


def result_table(result):
    rows = []

    if not result["success"]:
        return pd.DataFrame(
            {
                "Thông tin": ["Trạng thái"],
                "Giá trị": [result["status"]],
            }
        )

    for i in SECTORS:
        x_ai = result["x_ai"][i]
        x_h = result["x_h"][i]

        new_job = A1[i] * x_ai
        upgrade = B1[i] * x_h
        displaced = C1[i] * RISK[i] * x_ai
        retrain_cap = D1[i] * x_h
        net_job = new_job + upgrade - displaced

        rows.append(
            {
                "Ngành": i,
                "Tên ngành": SECTOR_NAMES[i],
                "x_AI": x_ai,
                "x_H": x_h,
                "NewJob": new_job,
                "Upgrade": upgrade,
                "Displaced": displaced,
                "RetrainCap": retrain_cap,
                "NetJob": net_job,
                "Tổng đầu tư": x_ai + x_h,
            }
        )

    return pd.DataFrame(rows)


def summary_table(result, label="Mô hình"):
    if not result["success"]:
        return pd.DataFrame(
            {
                "Chỉ tiêu": ["Kịch bản", "Trạng thái"],
                "Giá trị": [label, result["status"]],
            }
        )

    df = result_table(result)

    return pd.DataFrame(
        {
            "Chỉ tiêu": [
                "Kịch bản",
                "Trạng thái",
                "Tổng đầu tư",
                "Tổng x_AI",
                "Tổng x_H",
                "Tổng NewJob",
                "Tổng Upgrade",
                "Tổng Displaced",
                "Tổng NetJob",
            ],
            "Giá trị": [
                label,
                result["status"],
                df["Tổng đầu tư"].sum(),
                df["x_AI"].sum(),
                df["x_H"].sum(),
                df["NewJob"].sum(),
                df["Upgrade"].sum(),
                df["Displaced"].sum(),
                df["NetJob"].sum(),
            ],
        }
    )


def compare_results(base_result, capped_result):
    rows = []

    for label, result in [
        ("Mô hình gốc", base_result),
        ("Có ràng buộc 5%", capped_result),
    ]:
        if result["success"]:
            df = result_table(result)
            rows.append(
                {
                    "Kịch bản": label,
                    "Trạng thái": result["status"],
                    "Tổng đầu tư": df["Tổng đầu tư"].sum(),
                    "Tổng x_AI": df["x_AI"].sum(),
                    "Tổng x_H": df["x_H"].sum(),
                    "Tổng Displaced": df["Displaced"].sum(),
                    "Tổng NetJob": df["NetJob"].sum(),
                }
            )
        else:
            rows.append(
                {
                    "Kịch bản": label,
                    "Trạng thái": result["status"],
                    "Tổng đầu tư": np.nan,
                    "Tổng x_AI": np.nan,
                    "Tổng x_H": np.nan,
                    "Tổng Displaced": np.nan,
                    "Tổng NetJob": np.nan,
                }
            )

    return pd.DataFrame(rows)


def threshold_sector2(x_ai_value):
    i = 2

    displaced_coef = C1[i] * RISK[i]
    net_ai_coef = A1[i] - displaced_coef

    if net_ai_coef >= 0:
        h_for_netjob = 0.0
    else:
        h_for_netjob = ((displaced_coef - A1[i]) / B1[i]) * x_ai_value

    h_for_retrain = (displaced_coef / D1[i]) * x_ai_value

    return pd.DataFrame(
        {
            "Chỉ tiêu": [
                "Ngành",
                "x_AI giả định",
                "Hệ số NewJob AI",
                "Hệ số Displaced thực",
                "NetJob biên của AI",
                "x_H tối thiểu để NetJob₂ ≥ 0",
                "x_H tối thiểu để Displaced₂ ≤ RetrainCap₂",
                "Tổng ngân sách cần nếu đảm bảo retrain",
            ],
            "Giá trị": [
                SECTOR_NAMES[i],
                x_ai_value,
                A1[i],
                displaced_coef,
                net_ai_coef,
                h_for_netjob,
                h_for_retrain,
                x_ai_value + h_for_retrain,
            ],
        }
    )


def max_feasible_ai_sector2(total_budget=30000.0):
    i = 2
    displaced_coef = C1[i] * RISK[i]
    h_ratio_retrain = displaced_coef / D1[i]

    max_ai_with_retrain = total_budget / (1 + h_ratio_retrain)
    min_h = h_ratio_retrain * max_ai_with_retrain

    return pd.DataFrame(
        {
            "Chỉ tiêu": [
                "Tổng ngân sách",
                "Tỷ lệ x_H/x_AI tối thiểu để đủ RetrainCap",
                "x_AI tối đa khả thi nếu chỉ xét ngành 2",
                "x_H đi kèm tối thiểu",
            ],
            "Giá trị": [
                total_budget,
                h_ratio_retrain,
                max_ai_with_retrain,
                min_h,
            ],
        }
    )


def simulate_vulnerable_lanes(total_ai=6000.0, training_ratio=0.75):
    vulnerable = [1, 3, 4]

    labor_sum = sum(LABOR_THOUSAND[i] for i in vulnerable)

    rows = []

    for i in vulnerable:
        weight = LABOR_THOUSAND[i] / labor_sum
        x_ai = total_ai * weight
        x_h = x_ai * training_ratio

        displaced = C1[i] * RISK[i] * x_ai
        retrain_cap = D1[i] * x_h
        retrained = min(displaced, retrain_cap)
        unsupported = max(displaced - retrain_cap, 0)
        new_job = A1[i] * x_ai
        upgrade = B1[i] * x_h

        rows.append(
            {
                "Ngành": i,
                "Tên ngành": SECTOR_NAMES[i],
                "x_AI mô phỏng": x_ai,
                "x_H mô phỏng": x_h,
                "Displaced": displaced,
                "Đào tạo lại được": retrained,
                "Cần hỗ trợ an sinh": unsupported,
                "NewJob": new_job,
                "Upgrade": upgrade,
            }
        )

    return pd.DataFrame(rows)


def sankey_figure(vulnerable_df):
    labels = []

    for sector in vulnerable_df["Tên ngành"]:
        labels.append(f"LĐ phổ thông - {sector}")

    for sector in vulnerable_df["Tên ngành"]:
        labels.append(f"Bị thay thế - {sector}")

    labels += [
        "Được đào tạo lại",
        "Cần hỗ trợ an sinh",
        "Việc làm mới từ AI",
        "Nâng cấp kỹ năng",
    ]

    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    sources = []
    targets = []
    values = []

    for _, row in vulnerable_df.iterrows():
        source_label = f"LĐ phổ thông - {row['Tên ngành']}"
        displaced_label = f"Bị thay thế - {row['Tên ngành']}"

        sources.append(label_to_idx[source_label])
        targets.append(label_to_idx[displaced_label])
        values.append(max(row["Displaced"], 0.001))

        sources.append(label_to_idx[displaced_label])
        targets.append(label_to_idx["Được đào tạo lại"])
        values.append(max(row["Đào tạo lại được"], 0.001))

        if row["Cần hỗ trợ an sinh"] > 0:
            sources.append(label_to_idx[displaced_label])
            targets.append(label_to_idx["Cần hỗ trợ an sinh"])
            values.append(row["Cần hỗ trợ an sinh"])

        sources.append(label_to_idx[source_label])
        targets.append(label_to_idx["Việc làm mới từ AI"])
        values.append(max(row["NewJob"], 0.001))

        sources.append(label_to_idx[source_label])
        targets.append(label_to_idx["Nâng cấp kỹ năng"])
        values.append(max(row["Upgrade"], 0.001))

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=18,
                    thickness=18,
                    line=dict(color=BRAND, width=0.5),
                    label=labels,
                    color="rgba(5,49,81,0.85)",
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color="rgba(5,49,81,0.25)",
                ),
            )
        ]
    )

    fig.update_layout(
        title_text="Swimming lane/Sankey: luồng dịch chuyển lao động dễ bị tổn thương",
        height=620,
        paper_bgcolor="white",
        font=dict(color=BRAND, size=14),
        title_font=dict(color=BRAND, size=20),
    )

    return fig


def make_styled_table(df, decimals=3):
    show_df = df.copy()

    format_dict = {}

    for col in show_df.columns:
        if pd.api.types.is_numeric_dtype(show_df[col]):
            if str(col).lower() in ["ngành"]:
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
    st.title("Bài 9. AI, việc làm và đào tạo lại lao động")
    st.caption("Mô hình tuyến tính bằng PuLP/CBC, NetJob, ngưỡng đào tạo, Sankey lao động và ràng buộc an sinh")

    if not PULP_AVAILABLE:
        st.error("Chưa cài PuLP. Hãy thêm `pulp` vào requirements.txt rồi redeploy lại.")
        return

    base_result = solve_job_model(
        total_budget=30000,
        ai_min_budget=0,
        max_loss_constraint=False,
    )

    capped_result = solve_job_model(
        total_budget=30000,
        ai_min_budget=0,
        max_loss_constraint=True,
        max_loss_pct=0.05,
    )

    tabs = st.tabs(
        [
            "9.4.1 PuLP/CBC",
            "9.4.2 Ngưỡng ngành 2",
            "9.4.3 Sankey lao động",
            "9.4.4 Ràng buộc 5%",
            "9.5 Chính sách",
        ]
    )

    # =====================================================
    # 9.4.1
    # =====================================================
    with tabs[0]:
        st.header("9.4.1. Giải mô hình bằng PuLP/CBC")

        st.subheader("Dữ liệu đầu vào")
        show_table(sector_data(), decimals=3)

        if not base_result["success"]:
            st.error(f"Mô hình không tối ưu. Trạng thái: {base_result['status']}")
        else:
            result_df = result_table(base_result)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Trạng thái", base_result["status"])
            c2.metric("Tổng NetJob", f"{result_df['NetJob'].sum():,.1f}")
            c3.metric("Tổng x_AI", f"{result_df['x_AI'].sum():,.1f}")
            c4.metric("Tổng x_H", f"{result_df['x_H'].sum():,.1f}")

            st.subheader("Phân bổ tối ưu x_AI và x_H theo ngành")
            show_table(result_df, decimals=3)

            st.subheader("Tổng hợp nghiệm")
            show_table(summary_table(base_result, "Mô hình gốc"), decimals=3)

            invest_long = result_df[["Tên ngành", "x_AI", "x_H"]].rename(
                columns={
                    "x_AI": "AI",
                    "x_H": "Đào tạo lại",
                }
            ).melt(
                id_vars="Tên ngành",
                value_vars=["AI", "Đào tạo lại"],
                var_name="Loại đầu tư",
                value_name="Giá trị",
            )

            fig_inv = px.bar(
                invest_long,
                x="Tên ngành",
                y="Giá trị",
                color="Loại đầu tư",
                barmode="group",
                title="Phân bổ đầu tư AI và đào tạo lại theo ngành",
                color_discrete_map=MULTI_COLORS,
            )
            fig_inv.update_traces(marker_line_color="white", marker_line_width=1)
            fig_inv.update_layout(xaxis_title="Ngành", yaxis_title="Ngân sách")
            style_base_fig(fig_inv, height=520)
            st.plotly_chart(fig_inv, use_container_width=True)

            job_long = result_df[["Tên ngành", "NewJob", "Upgrade", "Displaced", "NetJob"]].melt(
                id_vars="Tên ngành",
                value_vars=["NewJob", "Upgrade", "Displaced", "NetJob"],
                var_name="Chỉ tiêu",
                value_name="Giá trị",
            )

            fig_job = px.bar(
                job_long,
                x="Tên ngành",
                y="Giá trị",
                color="Chỉ tiêu",
                barmode="group",
                title="NewJob, Upgrade, Displaced và NetJob theo ngành",
                color_discrete_map=MULTI_COLORS,
            )
            fig_job.update_traces(marker_line_color="white", marker_line_width=1)
            fig_job.update_layout(xaxis_title="Ngành", yaxis_title="Việc làm")
            style_base_fig(fig_job, height=520)
            st.plotly_chart(fig_job, use_container_width=True)

            st.markdown("### Kiểm thử nhanh yêu cầu ngân sách AI tối thiểu")

            ai_min = st.slider(
                "Ngân sách AI tối thiểu",
                0,
                20000,
                0,
                1000,
            )

            test_result = solve_job_model(
                total_budget=30000,
                ai_min_budget=ai_min,
                max_loss_constraint=False,
            )

            show_table(summary_table(test_result, f"AI tối thiểu {ai_min:,.0f}"), decimals=3)

            st.success(
                "Mô hình đã giải bằng PuLP/CBC. Nếu nghiệm tập trung vào đào tạo lại, đó là do hàm mục tiêu đang tối đa hóa NetJob ròng ngắn hạn."
            )

    # =====================================================
    # 9.4.2
    # =====================================================
    with tabs[1]:
        st.header("9.4.2. Ngưỡng đào tạo tối thiểu ở ngành 2")

        x_ai_sector2 = st.slider(
            "Giả định x_AI ở ngành 2",
            0,
            30000,
            30000,
            1000,
        )

        threshold_df = threshold_sector2(x_ai_sector2)
        feasible_ai_df = max_feasible_ai_sector2(total_budget=30000)

        c1, c2, c3 = st.columns(3)
        c1.metric("Ngành", "CN chế biến chế tạo")
        c2.metric("x_AI giả định", f"{x_ai_sector2:,.0f}")
        c3.metric("Risk ngành 2", f"{RISK[2]:.0%}")

        st.subheader("Ngưỡng x_H theo điều kiện NetJob₂ ≥ 0")
        show_table(threshold_df, decimals=3)

        st.subheader("Nếu đồng thời bảo đảm Displaced₂ ≤ RetrainCap₂")
        show_table(feasible_ai_df, decimals=3)

        st.success(
            "Với dữ liệu đề bài, NetJob biên của AI ở ngành 2 vẫn dương vì NewJob do AI lớn hơn DisplacedJob kỳ vọng. "
            "Do đó, riêng điều kiện NetJob₂ ≥ 0 không đòi hỏi x_H dương. Tuy nhiên, điều kiện an sinh Displaced₂ ≤ RetrainCap₂ "
            "lại yêu cầu đào tạo lại đi kèm khá lớn."
        )

    # =====================================================
    # 9.4.3
    # =====================================================
    with tabs[2]:
        st.header("9.4.3. Mô phỏng nhóm lao động dễ bị tổn thương")

        st.markdown(
            "Nhóm dễ bị tổn thương gồm lao động phổ thông trong các ngành 1, 3, 4: "
            "**Nông - Lâm - Thủy sản, Xây dựng, Bán buôn bán lẻ**."
        )

        c1, c2 = st.columns(2)

        total_ai_vulnerable = c1.slider(
            "Tổng đầu tư AI mô phỏng vào ngành 1, 3, 4",
            1000,
            15000,
            6000,
            1000,
        )

        training_ratio = c2.slider(
            "Tỷ lệ x_H / x_AI mô phỏng",
            0.20,
            1.50,
            0.75,
            0.05,
        )

        vulnerable_df = simulate_vulnerable_lanes(
            total_ai=total_ai_vulnerable,
            training_ratio=training_ratio,
        )

        show_table(vulnerable_df, decimals=3)

        fig_sankey = sankey_figure(vulnerable_df)
        st.plotly_chart(fig_sankey, use_container_width=True)

        flow_long = vulnerable_df[
            ["Tên ngành", "Displaced", "Đào tạo lại được", "Cần hỗ trợ an sinh", "NewJob", "Upgrade"]
        ].melt(
            id_vars="Tên ngành",
            var_name="Luồng",
            value_name="Giá trị",
        )

        fig_flow = px.bar(
            flow_long,
            x="Tên ngành",
            y="Giá trị",
            color="Luồng",
            barmode="group",
            title="Tác động đến nhóm lao động dễ bị tổn thương",
        )
        fig_flow.update_traces(marker_line_color="white", marker_line_width=1)
        fig_flow.update_layout(xaxis_title="Ngành", yaxis_title="Lao động")
        style_base_fig(fig_flow, height=500)
        st.plotly_chart(fig_flow, use_container_width=True)

        st.success(
            "Sankey cho thấy phần lao động bị thay thế có thể được đào tạo lại, nhưng nếu năng lực đào tạo không đủ thì xuất hiện nhóm cần hỗ trợ an sinh."
        )

    # =====================================================
    # 9.4.4
    # =====================================================
    with tabs[3]:
        st.header("9.4.4. Thêm ràng buộc không ngành nào mất quá 5% lao động")

        compare_df = compare_results(base_result, capped_result)

        show_table(compare_df, decimals=3)

        c1, c2, c3 = st.columns(3)
        c1.metric("Mô hình gốc", base_result["status"])
        c2.metric("Có ràng buộc 5%", capped_result["status"])

        if base_result["success"] and capped_result["success"]:
            delta = capped_result["objective"] - base_result["objective"]
            c3.metric("Thay đổi NetJob", f"{delta:,.1f}")
        else:
            c3.metric("Thay đổi NetJob", "Không xác định")

        if capped_result["success"]:
            capped_df = result_table(capped_result)

            st.subheader("Phân bổ khi thêm ràng buộc 5%")
            show_table(capped_df, decimals=3)

            fig = px.bar(
                capped_df,
                x="Tên ngành",
                y="Displaced",
                text=capped_df["Displaced"].round(1),
                title="DisplacedJob theo ngành dưới ràng buộc 5%",
            )
            fig.update_traces(
                marker_color=BRAND,
                textposition="outside",
                textfont=dict(color=BRAND),
            )
            fig.update_layout(xaxis_title="Ngành", yaxis_title="DisplacedJob")
            style_base_fig(fig, height=480)
            st.plotly_chart(fig, use_container_width=True)

            st.success(
                "Bài toán vẫn khả thi khi thêm ràng buộc 5%, nếu mô hình có thể điều chỉnh giảm AI hoặc tăng đào tạo lại để kiểm soát mất việc."
            )
        else:
            st.error(
                "Bài toán không khả thi với ràng buộc 5%. Điều này có nghĩa tốc độ tự động hóa đang vượt quá khả năng bảo vệ việc làm của hệ thống."
            )

        st.markdown("### Kiểm thử thêm: yêu cầu AI tối thiểu + ràng buộc 5%")

        ai_min_capped = st.slider(
            "Ngân sách AI tối thiểu khi có ràng buộc 5%",
            0,
            20000,
            5000,
            1000,
        )

        test_capped = solve_job_model(
            total_budget=30000,
            ai_min_budget=ai_min_capped,
            max_loss_constraint=True,
            max_loss_pct=0.05,
        )

        show_table(summary_table(test_capped, f"AI tối thiểu {ai_min_capped:,.0f} + ràng buộc 5%"), decimals=3)

    # =====================================================
    # 9.5
    # =====================================================
    with tabs[4]:
        st.header("9.5. Thảo luận chính sách")

        base_df = result_table(base_result) if base_result["success"] else pd.DataFrame()

        if len(base_df) > 0:
            top_training = base_df.sort_values("x_H", ascending=False).iloc[0]["Tên ngành"]
            finance_row = base_df[base_df["Ngành"] == 5].iloc[0]
            agri_row = base_df[base_df["Ngành"] == 1].iloc[0]
        else:
            top_training = "Không xác định"
            finance_row = None
            agri_row = None

        st.markdown("### a) Ngành nào cần đầu tư đào tạo lại nhiều nhất?")

        st.success(f"Theo nghiệm tối ưu, ngành nhận đào tạo lại nhiều nhất là **{top_training}**.")

        st.markdown(
            """
            Kết quả này cần được hiểu theo logic của mô hình: đào tạo lại tạo ra UpgradeJob và đồng thời giúp hấp thụ rủi ro do tự động hóa.
            Nếu một ngành có hệ số b1 hoặc d1 cao, mô hình có xu hướng phân bổ nhiều x_H vào ngành đó.

            Về thực tế Việt Nam, nhu cầu đào tạo lại thường lớn ở các ngành đông lao động hoặc có tốc độ tự động hóa nhanh:
            chế biến chế tạo, bán lẻ, logistics, tài chính - ngân hàng và một phần nông nghiệp công nghệ cao.
            Vì vậy, kết quả mô hình nên được đối chiếu thêm với quy mô lao động và khả năng thực thi đào tạo tại từng ngành.
            """
        )

        st.markdown("### b) Ngành Tài chính - Ngân hàng có risk 52% nhưng hệ số tạo việc làm AI cao")

        st.markdown(
            """
            Tài chính - Ngân hàng là ngành có hai mặt rất rõ.
            Một mặt, risk thay thế cao vì nhiều tác vụ có thể tự động hóa: chấm điểm tín dụng, chăm sóc khách hàng,
            phát hiện gian lận, xử lý hồ sơ và giao dịch số.
            Mặt khác, hệ số tạo việc làm mới từ AI cũng cao vì ngành này có thể phát sinh nhiều vị trí mới:
            chuyên gia dữ liệu, kỹ sư mô hình, quản trị rủi ro AI, an ninh dữ liệu, sản phẩm tài chính số.

            Vì vậy, chiến lược phù hợp không phải là tránh AI, mà là **AI đi kèm đào tạo lại bắt buộc**.
            Nghĩa là tốc độ triển khai AI phải gắn với năng lực chuyển đổi kỹ năng của lao động hiện hữu.
            """
        )

        st.markdown("### c) Có nên đầu tư AI vào Nông - Lâm - Thủy sản không?")

        st.markdown(
            """
            Ngành Nông - Lâm - Thủy sản có hệ số tạo việc làm AI thấp hơn nhiều ngành khác,
            nhưng quy mô lao động lớn nên tác động xã hội của tự động hóa có thể rất đáng kể.
            Nếu chỉ tối đa hóa NetJob ngắn hạn, mô hình có thể không ưu tiên x_AI vào ngành này.
            Tuy nhiên, điều đó không có nghĩa là không nên đầu tư AI.

            Với nông nghiệp, AI nên được triển khai theo hướng hỗ trợ năng suất và giảm rủi ro:
            dự báo thời tiết, tối ưu tưới tiêu, truy xuất nguồn gốc, cảnh báo dịch bệnh, logistics nông sản,
            thay vì tự động hóa thay thế lao động quá nhanh.

            Mô hình hàm ý rằng nếu đầu tư AI vào nông nghiệp, cần đi cùng đào tạo kỹ năng số cơ bản,
            khuyến nông số và chính sách hỗ trợ chuyển đổi sinh kế cho lao động dễ bị tổn thương.
            """
        )

        st.markdown("### d) Ràng buộc “tốc độ tự động hóa không vượt quá năng lực đào tạo lại” là ràng buộc nào?")

        st.success("Ràng buộc đó là: **DisplacedJobᵢ ≤ RetrainCapᵢ** cho mọi ngành i.")

        st.markdown(
            """
            Trong mô hình, ràng buộc này được viết dưới dạng:

            `c1_i × risk_i × x_AI_i ≤ d1_i × x_H_i`

            Vế trái là số lao động có nguy cơ bị thay thế do AI.
            Vế phải là năng lực đào tạo lại do đầu tư x_H tạo ra.
            Đây chính là cách mô hình hóa nguyên tắc: tự động hóa không được đi nhanh hơn khả năng tái đào tạo.

            Có thể bổ sung thêm các ràng buộc an sinh xã hội:

            - Không ngành nào mất quá 5% lao động: `DisplacedJob_i ≤ 0.05 × L_i`.
            - Nhóm dễ bị tổn thương phải nhận tối thiểu một tỷ lệ ngân sách đào tạo.
            - Ngành có risk cao phải có tỷ lệ x_H/x_AI tối thiểu.
            - Tổng lao động cần hỗ trợ an sinh không vượt quá năng lực ngân sách xã hội.
            - Ưu tiên đào tạo trước cho ngành đông lao động phổ thông.

            Như vậy, mô hình không chỉ tối đa hóa việc làm ròng mà còn kiểm soát tốc độ chuyển đổi để tránh cú sốc xã hội.
            """
        )

        st.info(
            "Kết luận: AI có thể tạo việc làm ròng dương, nhưng chỉ bền vững khi đầu tư đào tạo lại đủ lớn và đúng ngành."
        )


def run():
    render()

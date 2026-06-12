# bai05_mip_15_du_an.py

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

try:
    import pulp
    PULP_AVAILABLE = True
except Exception:
    PULP_AVAILABLE = False


BRAND = "#053151"

MULTI_COLORS = {
    "Hạ tầng": "#053151",
    "Chính phủ số": "#E76F51",
    "AI/bán dẫn": "#2A9D8F",
    "An ninh/dữ liệu": "#F4A261",
    "Khác": "#8E44AD",
    "Gốc": "#053151",
    "Ngân sách 100.000": "#E76F51",
    "Bắt buộc P1 & P2": "#2A9D8F",
    "Có rủi ro": "#F4A261",
}


# =========================================================
# BÀI 5 — LỰA CHỌN DANH MỤC DỰ ÁN SỐ
# =========================================================


PROJECTS = list(range(1, 16))

C = {
    1: 12000, 2: 11500, 3: 18000, 4: 4500, 5: 3200,
    6: 5800, 7: 6500, 8: 15000, 9: 2500, 10: 7200,
    11: 4800, 12: 8500, 13: 20000, 14: 3800, 15: 1500,
}

C1 = {
    1: 8500, 2: 7500, 3: 12000, 4: 3500, 5: 2500,
    6: 4000, 7: 4500, 8: 9000, 9: 1800, 10: 5000,
    11: 3500, 12: 5500, 13: 13000, 14: 2800, 15: 1200,
}

B = {
    1: 21500, 2: 20800, 3: 32500, 4: 9200, 5: 6800,
    6: 11400, 7: 12200, 8: 28500, 9: 5800, 10: 13800,
    11: 8500, 12: 16200, 13: 35000, 14: 7500, 15: 3800,
}


PROJECT_META = {
    1: ("P1", "Hạ tầng số backbone A", "Hạ tầng"),
    2: ("P2", "Hạ tầng số backbone B", "Hạ tầng"),
    3: ("P3", "Trung tâm dữ liệu quốc gia", "Hạ tầng"),
    4: ("P4", "Dịch vụ công số lõi", "Chính phủ số"),
    5: ("P5", "Định danh và hồ sơ số", "Chính phủ số"),
    6: ("P6", "Nền tảng tích hợp dữ liệu", "Chính phủ số"),
    7: ("P7", "Hỗ trợ SME chuyển đổi số", "Khác"),
    8: ("P8", "AI quốc gia", "AI/bán dẫn"),
    9: ("P9", "Thí điểm đô thị thông minh", "Khác"),
    10: ("P10", "Kỹ năng số lực lượng lao động", "Khác"),
    11: ("P11", "Y tế số", "Khác"),
    12: ("P12", "Nền tảng dữ liệu dùng chung", "An ninh/dữ liệu"),
    13: ("P13", "Bán dẫn chiến lược", "AI/bán dẫn"),
    14: ("P14", "An ninh mạng quốc gia", "An ninh/dữ liệu"),
    15: ("P15", "Open Data", "An ninh/dữ liệu"),
}


def risk_probability(project_id):
    category = PROJECT_META[project_id][2]

    if category == "Hạ tầng":
        return 0.85
    if category == "Chính phủ số":
        return 0.75
    if category == "AI/bán dẫn":
        return 0.65

    return 0.80


def project_data():
    rows = []

    for i in PROJECTS:
        code, name, category = PROJECT_META[i]
        p = risk_probability(i)

        rows.append(
            {
                "Dự án": code,
                "Tên dự án": name,
                "Nhóm": category,
                "Chi phí C": C[i],
                "Chi phí C1": C1[i],
                "Lợi ích B": B[i],
                "B/C": B[i] / C[i],
                "p_i": p,
                "Lợi ích kỳ vọng": p * B[i],
            }
        )

    return pd.DataFrame(rows)


def solve_project_model(
    budget=80000,
    c1_budget=40000,
    force_p1_p2=False,
    remove_p1_p2_exclusion=False,
    require_p14=True,
    objective_mode="benefit",
):
    if not PULP_AVAILABLE:
        return {
            "success": False,
            "status": "PuLP chưa được cài",
            "selected": [],
            "objective": np.nan,
            "cost": np.nan,
            "cost_c1": np.nan,
            "model": None,
        }

    model = pulp.LpProblem("VN_Project_Selection", pulp.LpMaximize)
    y = pulp.LpVariable.dicts("y", PROJECTS, cat="Binary")

    if objective_mode == "risk":
        model += pulp.lpSum(risk_probability(i) * B[i] * y[i] for i in PROJECTS), "Expected_Z"
    else:
        model += pulp.lpSum(B[i] * y[i] for i in PROJECTS), "Z"

    model += pulp.lpSum(C[i] * y[i] for i in PROJECTS) <= budget, "Total_budget"
    model += pulp.lpSum(C1[i] * y[i] for i in PROJECTS) <= c1_budget, "C1_budget"

    if not remove_p1_p2_exclusion:
        model += y[1] + y[2] <= 1, "P1_P2_mutual_exclusion"

    model += y[8] <= y[12], "P8_requires_P12"
    model += y[13] <= y[12], "P13_requires_P12"
    model += y[4] + y[5] >= 1, "P4_or_P5"
    
    if require_p14:
        model += y[14] >= 1, "P14_required"

    model += pulp.lpSum(y[i] for i in PROJECTS) >= 7, "Min_projects"
    model += pulp.lpSum(y[i] for i in PROJECTS) <= 11, "Max_projects"

    if force_p1_p2:
        model += y[1] == 1, "Force_P1"
        model += y[2] == 1, "Force_P2"

    solver = pulp.PULP_CBC_CMD(msg=False)
    status_code = model.solve(solver)
    status = pulp.LpStatus[status_code]

    if status != "Optimal":
        return {
            "success": False,
            "status": status,
            "selected": [],
            "objective": np.nan,
            "cost": np.nan,
            "cost_c1": np.nan,
            "model": model,
        }

    selected = [i for i in PROJECTS if y[i].value() is not None and y[i].value() > 0.5]
    objective = float(pulp.value(model.objective))
    cost = sum(C[i] for i in selected)
    cost_c1 = sum(C1[i] for i in selected)

    return {
        "success": True,
        "status": status,
        "selected": selected,
        "objective": objective,
        "cost": cost,
        "cost_c1": cost_c1,
        "model": model,
    }


def selected_table(result, objective_mode="benefit"):
    data = project_data()
    selected_codes = [f"P{i}" for i in result["selected"]]

    out = data[data["Dự án"].isin(selected_codes)].copy()

    if objective_mode == "risk":
        out["Giá trị mục tiêu"] = out["Lợi ích kỳ vọng"]
    else:
        out["Giá trị mục tiêu"] = out["Lợi ích B"]

    out = out.sort_values("Dự án").reset_index(drop=True)

    return out


def summary_table(result, label="Kịch bản"):
    if not result["success"]:
        return pd.DataFrame(
            {
                "Chỉ tiêu": ["Kịch bản", "Trạng thái"],
                "Giá trị": [label, result["status"]],
            }
        )

    npv_ratio = result["objective"] / result["cost"] if result["cost"] > 0 else np.nan

    return pd.DataFrame(
        {
            "Chỉ tiêu": [
                "Kịch bản",
                "Trạng thái",
                "Số dự án chọn",
                "Tổng chi phí",
                "Tổng chi phí C1",
                "Giá trị mục tiêu Z*",
                "NPV biên Z*/chi phí",
            ],
            "Giá trị": [
                label,
                result["status"],
                len(result["selected"]),
                result["cost"],
                result["cost_c1"],
                result["objective"],
                npv_ratio,
            ],
        }
    )


def compare_solutions(results_dict):
    rows = []

    for label, res in results_dict.items():
        if res["success"]:
            rows.append(
                {
                    "Kịch bản": label,
                    "Trạng thái": res["status"],
                    "Tập dự án": ", ".join([f"P{i}" for i in res["selected"]]),
                    "Số dự án": len(res["selected"]),
                    "Tổng chi phí": res["cost"],
                    "Tổng C1": res["cost_c1"],
                    "Z*": res["objective"],
                    "Z*/chi phí": res["objective"] / res["cost"],
                }
            )
        else:
            rows.append(
                {
                    "Kịch bản": label,
                    "Trạng thái": res["status"],
                    "Tập dự án": "",
                    "Số dự án": np.nan,
                    "Tổng chi phí": np.nan,
                    "Tổng C1": np.nan,
                    "Z*": np.nan,
                    "Z*/chi phí": np.nan,
                }
            )

    return pd.DataFrame(rows)


def make_styled_table(df, decimals=3):
    show_df = df.copy()

    format_dict = {}
    for col in show_df.columns:
        if pd.api.types.is_numeric_dtype(show_df[col]):
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


def plot_selected_projects(df, title, value_col="Lợi ích B"):
    fig = px.bar(
        df,
        x="Dự án",
        y=value_col,
        color="Nhóm",
        text=df[value_col].round(1),
        title=title,
        color_discrete_map=MULTI_COLORS,
        hover_data=["Tên dự án", "Chi phí C", "Chi phí C1", "B/C"],
    )

    fig.update_traces(
        textposition="outside",
        marker_line_color="white",
        marker_line_width=1,
    )
    fig.update_layout(
        xaxis_title="Dự án",
        yaxis_title=value_col,
    )
    style_base_fig(fig, height=450)

    return fig


def render():
    st.title("Bài 5. Lựa chọn danh mục dự án số")
    st.caption("Bài toán nhị phân 0-1 bằng PuLP/CBC, phân tích ngân sách, redundancy và rủi ro dự án")

    if not PULP_AVAILABLE:
        st.error("Chưa cài PuLP. Hãy thêm `pulp` vào requirements.txt.")
        return

    base_res = solve_project_model(
        budget=80000,
        c1_budget=40000,
        force_p1_p2=False,
        remove_p1_p2_exclusion=False,
        require_p14=True,
        objective_mode="benefit",
    )

    budget100_res = solve_project_model(
        budget=100000,
        c1_budget=40000,
        force_p1_p2=False,
        remove_p1_p2_exclusion=False,
        require_p14=True,
        objective_mode="benefit",
    )

    both_conflict_res = solve_project_model(
        budget=80000,
        c1_budget=40000,
        force_p1_p2=True,
        remove_p1_p2_exclusion=False,
        require_p14=True,
        objective_mode="benefit",
    )

    both_override_res = solve_project_model(
        budget=80000,
        c1_budget=40000,
        force_p1_p2=True,
        remove_p1_p2_exclusion=True,
        require_p14=True,
        objective_mode="benefit",
    )

    risk_res = solve_project_model(
        budget=80000,
        c1_budget=40000,
        force_p1_p2=False,
        remove_p1_p2_exclusion=False,
        require_p14=True,
        objective_mode="risk",
    )

    no_p14_res = solve_project_model(
        budget=80000,
        c1_budget=40000,
        force_p1_p2=False,
        remove_p1_p2_exclusion=False,
        require_p14=False,
        objective_mode="benefit",
    )

    tabs = st.tabs(
        [
            "Lựa chọn dự án",
            "Ngân sách 100.000",
            "5.4.3 Bắt buộc P1 & P2",
            "Rủi ro dự án",
            "Thảo luận chính sách",
        ]
    )

    # =====================================================
    # 5.4.1
    # =====================================================
    with tabs[0]:
      

        st.subheader("Dữ liệu dự án")
        show_table(project_data(), decimals=3)

        if not base_res["success"]:
            st.error(f"Mô hình không tối ưu. Trạng thái: {base_res['status']}")
        else:
            selected_df = selected_table(base_res, objective_mode="benefit")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Trạng thái", "Optimal")
            c2.metric("Số dự án", f"{len(base_res['selected'])}")
            c3.metric("Z*", f"{base_res['objective']:,.0f}")
            c4.metric("Z*/chi phí", f"{base_res['objective'] / base_res['cost']:.3f}")

            st.subheader("Dự án được chọn")
            show_table(selected_df, decimals=3)

            st.subheader("Tổng hợp nghiệm")
            show_table(summary_table(base_res, "Ngân sách 80.000"), decimals=3)

            fig = plot_selected_projects(
                selected_df,
                "Lợi ích các dự án được chọn",
                value_col="Lợi ích B",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.success(
                f"PuLP/CBC chọn {len(base_res['selected'])} dự án, tổng chi phí "
                f"{base_res['cost']:,.0f}, tổng lợi ích Z* = {base_res['objective']:,.0f}, "
                f"NPV biên Z*/chi phí = {base_res['objective'] / base_res['cost']:.3f}."
            )

    # =====================================================
    # 5.4.2
    # =====================================================
    with tabs[1]:
        

        compare_df = compare_solutions(
            {
                "Gốc 80.000": base_res,
                "Ngân sách 100.000": budget100_res,
            }
        )

        show_table(compare_df, decimals=3)

        if base_res["success"] and budget100_res["success"]:
            base_set = set(base_res["selected"])
            new_set = set(budget100_res["selected"])

            added = sorted(list(new_set - base_set))
            removed = sorted(list(base_set - new_set))

            c1, c2, c3 = st.columns(3)
            c1.metric("Z* gốc", f"{base_res['objective']:,.0f}")
            c2.metric("Z* 100.000", f"{budget100_res['objective']:,.0f}")
            c3.metric("Thay đổi Z*", f"{budget100_res['objective'] - base_res['objective']:,.0f}")

            compare_plot = pd.DataFrame(
                {
                    "Kịch bản": ["Gốc 80.000", "Ngân sách 100.000"],
                    "Z*": [base_res["objective"], budget100_res["objective"]],
                    "Tổng chi phí": [base_res["cost"], budget100_res["cost"]],
                    "Tổng C1": [base_res["cost_c1"], budget100_res["cost_c1"]],
                }
            )

            fig = px.bar(
                compare_plot,
                x="Kịch bản",
                y="Z*",
                color="Kịch bản",
                text=compare_plot["Z*"].round(0),
                title="So sánh Z* khi nới ngân sách",
                color_discrete_map=MULTI_COLORS,
            )
            fig.update_traces(textposition="outside", marker_line_color="white", marker_line_width=1)
            fig.update_layout(xaxis_title="Kịch bản", yaxis_title="Z*")
            style_base_fig(fig, height=420)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Điều chỉnh nhanh ngân sách tổng")

            custom_budget = st.slider(
                "Ngân sách tổng",
                70000,
                130000,
                100000,
                5000,
            )

            custom_res = solve_project_model(
                budget=custom_budget,
                c1_budget=40000,
                force_p1_p2=False,
                remove_p1_p2_exclusion=False,
                require_p14=True,
                objective_mode="benefit",
            )

            show_table(
                compare_solutions(
                    {
                        "Gốc 80.000": base_res,
                        f"Tùy chỉnh {custom_budget:,.0f}": custom_res,
                    }
                ),
                decimals=3,
            )

            if len(added) == 0 and len(removed) == 0:
                st.success(
                    "Khi nới ngân sách tổng lên 100.000, tập dự án không thay đổi. "
                    "Điều này cho thấy ràng buộc C1 hoặc các ràng buộc logic đang là nút thắt chính."
                )
            else:
                st.success(
                    f"Khi nới ngân sách, dự án thêm mới: {', '.join([f'P{i}' for i in added]) or 'không có'}; "
                    f"dự án bị loại: {', '.join([f'P{i}' for i in removed]) or 'không có'}."
                )

    # =====================================================
    # 5.4.3
    # =====================================================
    with tabs[2]:
     

        compare_df = compare_solutions(
            {
                "Gốc": base_res,
                "Thêm P1=P2=1, giữ loại trừ": both_conflict_res,
                "Redundancy: bỏ loại trừ P1/P2": both_override_res,
            }
        )

        show_table(compare_df, decimals=3)

        c1, c2, c3 = st.columns(3)
        c1.metric("Gốc", base_res["status"])
        c2.metric("Giữ y1+y2<=1", both_conflict_res["status"])
        c3.metric("Bỏ loại trừ", both_override_res["status"])

        if both_override_res["success"]:
            selected_df = selected_table(both_override_res, objective_mode="benefit")

            fig = plot_selected_projects(
                selected_df,
                "Dự án được chọn khi yêu cầu redundancy P1 và P2",
                value_col="Lợi ích B",
            )
            st.plotly_chart(fig, use_container_width=True)

        if not both_conflict_res["success"]:
            st.warning(
                "Nếu vẫn giữ ràng buộc y1 + y2 <= 1, yêu cầu bắt buộc cả P1 và P2 làm bài toán không khả thi."
            )

        if both_override_res["success"]:
            delta_z = both_override_res["objective"] - base_res["objective"]

            st.success(
                f"Nếu Quốc hội cho phép bỏ ràng buộc loại trừ P1/P2 để phục vụ redundancy, "
                f"bài toán khả thi. Z* thay đổi {delta_z:,.0f} so với nghiệm gốc."
            )

    # =====================================================
    # 5.4.4
    # =====================================================
    with tabs[3]:
       

        risk_df = project_data()[["Dự án", "Tên dự án", "Nhóm", "Lợi ích B", "p_i", "Lợi ích kỳ vọng"]].copy()
        show_table(risk_df, decimals=3)

        if not risk_res["success"]:
            st.error(f"Mô hình rủi ro không tối ưu. Trạng thái: {risk_res['status']}")
        else:
            selected_risk_df = selected_table(risk_res, objective_mode="risk")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Trạng thái", "Optimal")
            c2.metric("Số dự án", f"{len(risk_res['selected'])}")
            c3.metric("E[Z]*", f"{risk_res['objective']:,.0f}")
            c4.metric("E[Z]/chi phí", f"{risk_res['objective'] / risk_res['cost']:.3f}")

            st.subheader("Dự án được chọn theo lợi ích kỳ vọng")
            show_table(selected_risk_df, decimals=3)

            compare_risk_df = compare_solutions(
                {
                    "Gốc": base_res,
                    "Có rủi ro": risk_res,
                }
            )
            show_table(compare_risk_df, decimals=3)

            fig = plot_selected_projects(
                selected_risk_df,
                "Lợi ích kỳ vọng của các dự án được chọn",
                value_col="Lợi ích kỳ vọng",
            )
            st.plotly_chart(fig, use_container_width=True)

            base_set = set(base_res["selected"])
            risk_set = set(risk_res["selected"])
            added = sorted(list(risk_set - base_set))
            removed = sorted(list(base_set - risk_set))

            st.success(
                f"Khi xét rủi ro, hàm mục tiêu chuyển thành E[Z]. "
                f"Dự án thêm mới: {', '.join([f'P{i}' for i in added]) or 'không có'}; "
                f"dự án bị loại: {', '.join([f'P{i}' for i in removed]) or 'không có'}."
            )

    # =====================================================
    # 5.5
    # =====================================================
    with tabs[4]:
     

        p15_selected = base_res["success"] and 15 in base_res["selected"]

        st.markdown("### a) Vì sao mô hình có thể bỏ qua P15 dù B/C cao?")

        if p15_selected:
            st.success("Với bộ dữ liệu và ràng buộc hiện tại, P15 đang được chọn trong nghiệm tối ưu.")
        else:
            st.warning("Trong nghiệm hiện tại, P15 không được chọn.")

        st.markdown(
            """
            P15 có tỷ suất lợi ích/chi phí cao vì chi phí nhỏ và lợi ích tương đối tốt.
            Tuy nhiên, bài toán không tối đa hóa riêng tỷ suất B/C, mà tối đa hóa **tổng lợi ích Z**
            dưới nhiều ràng buộc đồng thời: ngân sách tổng, ngân sách C1, quan hệ phụ thuộc giữa dự án,
            yêu cầu bắt buộc P14, giới hạn số lượng dự án và ràng buộc chọn P4 hoặc P5.

            Vì vậy, một dự án có B/C cao vẫn có thể bị loại nếu nó không giúp giải quyết nút thắt chính,
            không đủ đóng góp tuyệt đối vào Z, hoặc làm mất chỗ của dự án có lợi ích tổng lớn hơn.
            Ngược lại, nếu P15 được chọn, điều đó phản ánh vai trò của dự án nhỏ nhưng hiệu quả,
            nhất là khi mô hình có giới hạn ngân sách và cần đủ số lượng dự án.

            Về chính sách, không nên chỉ dùng B/C để quyết định. Các dự án nền tảng như Open Data
            có thể tạo hiệu ứng lan tỏa dài hạn mà lợi ích trực tiếp trong mô hình chưa phản ánh hết.
            """
        )

        st.markdown("### b) Ràng buộc bắt buộc P14 có làm giảm Z* không?")

        if base_res["success"] and no_p14_res["success"]:
            p14_cost = no_p14_res["objective"] - base_res["objective"]
            p14_pct = p14_cost / no_p14_res["objective"] * 100

            st.success(
                f"Khi bỏ ràng buộc bắt buộc P14, Z* thay đổi khoảng **{p14_cost:,.0f}**, "
                f"tương đương **{p14_pct:.2f}%** so với mô hình không bắt buộc P14."
            )
        else:
            p14_cost = np.nan

        st.markdown(
            """
            Ràng buộc bắt buộc P14 có thể làm giảm Z* nếu P14 không phải dự án có lợi ích biên cao nhất
            trong danh mục. Đây là chi phí cơ hội của việc bắt buộc đầu tư an ninh mạng.

            Tuy nhiên, bắt buộc P14 là hợp lý về mặt chính sách. An ninh mạng là điều kiện nền tảng
            cho mọi chương trình chuyển đổi số. Nếu không có lớp bảo vệ này, các dự án dữ liệu, AI,
            chính phủ số và Open Data có thể tạo rủi ro hệ thống: rò rỉ dữ liệu, tấn công hạ tầng số,
            mất niềm tin công dân và gián đoạn dịch vụ công.

            Do đó, P14 giống một khoản đầu tư bảo hiểm hệ thống. Nó có thể làm giảm lợi ích tối đa ngắn hạn,
            nhưng làm tăng độ an toàn và tính bền vững của toàn bộ danh mục.
            """
        )

        st.markdown("### c) Mô hình hóa cộng hưởng giữa P8 và P13 như thế nào?")

        st.markdown(
            """
            Mô hình hiện tại giả định lợi ích các dự án là độc lập:
            tổng lợi ích bằng tổng riêng lẻ của từng dự án. Nhưng trên thực tế,
            P8 về AI quốc gia và P13 về bán dẫn có thể cộng hưởng mạnh:
            năng lực bán dẫn hỗ trợ phần cứng AI, còn AI tạo nhu cầu ứng dụng và thiết kế chip.

            Có thể mô hình hóa hiệu ứng cộng hưởng bằng cách thêm một biến nhị phân mới:

            - Gọi `s_8_13 = 1` nếu cả P8 và P13 cùng được chọn.
            - Thêm các ràng buộc tuyến tính:
              `s_8_13 <= y8`,
              `s_8_13 <= y13`,
              `s_8_13 >= y8 + y13 - 1`.
            - Cộng thêm vào hàm mục tiêu một khoản lợi ích cộng hưởng:
              `phi * s_8_13`.

            Khi đó hàm mục tiêu trở thành:

            `Max Z = Σ B_i y_i + phi * s_8_13`

            Trong đó `phi` là lợi ích cộng hưởng ước tính giữa AI quốc gia và bán dẫn.
            Cách này vẫn giữ bài toán là mô hình tuyến tính nguyên nhị phân, có thể giải bằng PuLP/CBC.
            """
        )

        st.info(
            "Kết luận: mô hình chọn danh mục tối ưu theo lợi ích tổng, nhưng chính sách thực tế cần xét thêm rủi ro, "
            "an ninh hệ thống và hiệu ứng cộng hưởng giữa các dự án chiến lược."
        )


def run():
    render()

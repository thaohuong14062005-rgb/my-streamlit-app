# bai02_linprog_pulp.py

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

try:
    from scipy.optimize import linprog
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    import pulp
    PULP_AVAILABLE = True
except Exception:
    PULP_AVAILABLE = False


BRAND = "#053151"

MULTI_COLORS = {
    "Hạ tầng số": "#053151",
    "AI": "#E76F51",
    "Nhân lực số": "#2A9D8F",
    "R&D": "#F4A261",
    "SciPy": "#053151",
    "PuLP": "#E76F51",
}


# =========================================================
# BÀI 2 — QUY HOẠCH TUYẾN TÍNH PHÂN BỔ NGÂN SÁCH
# =========================================================


def get_problem_data():
    return pd.DataFrame(
        {
            "Biến": ["x1", "x2", "x3", "x4"],
            "Hạng mục": ["Hạ tầng số", "AI", "Nhân lực số", "R&D"],
            "Hệ số tác động": [0.85, 1.20, 0.95, 1.35],
            "Mức tối thiểu": [25.0, 15.0, 20.0, 10.0],
        }
    )


def build_scipy_matrices(budget=100.0, x3_min=20.0):
    c = [-0.85, -1.20, -0.95, -1.35]

    A_ub = [
        [1, 1, 1, 1],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1],
        [0.35, -0.65, 0.35, -0.65],
    ]

    b_ub = [
        budget,
        -25,
        -15,
        -x3_min,
        -10,
        0,
    ]

    return c, A_ub, b_ub


def solve_with_scipy(budget=100.0, x3_min=20.0):
    if not SCIPY_AVAILABLE:
        return {
            "success": False,
            "message": "Chưa cài scipy.",
            "x": None,
            "objective": np.nan,
        }

    c, A_ub, b_ub = build_scipy_matrices(budget=budget, x3_min=x3_min)

    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=[(0, None)] * 4,
        method="highs",
    )

    if not res.success:
        return {
            "success": False,
            "message": res.message,
            "x": None,
            "objective": np.nan,
            "raw": res,
        }

    return {
        "success": True,
        "message": res.message,
        "x": res.x,
        "objective": -res.fun,
        "raw": res,
    }


def solve_with_pulp(budget=100.0, x3_min=20.0):
    if not PULP_AVAILABLE:
        return {
            "success": False,
            "message": "Chưa cài pulp. Hãy thêm pulp vào requirements.txt.",
            "x": None,
            "objective": np.nan,
            "duals": None,
            "slacks": None,
        }

    prob = pulp.LpProblem("Bai_2_Phan_bo_ngan_sach", pulp.LpMaximize)

    x1 = pulp.LpVariable("x1_Ha_tang_so", lowBound=0)
    x2 = pulp.LpVariable("x2_AI", lowBound=0)
    x3 = pulp.LpVariable("x3_Nhan_luc_so", lowBound=0)
    x4 = pulp.LpVariable("x4_RnD", lowBound=0)

    variables = [x1, x2, x3, x4]

    prob += 0.85 * x1 + 1.20 * x2 + 0.95 * x3 + 1.35 * x4, "Z"

    prob += x1 + x2 + x3 + x4 <= budget, "Ngan_sach_tong"
    prob += x1 >= 25, "Toi_thieu_ha_tang_so"
    prob += x2 >= 15, "Toi_thieu_AI"
    prob += x3 >= x3_min, "Toi_thieu_nhan_luc_so"
    prob += x4 >= 10, "Toi_thieu_RnD"
    prob += 0.35 * (x1 + x3) <= 0.65 * (x2 + x4), "Ty_le_cong_nghe_chien_luoc"

    solver = pulp.PULP_CBC_CMD(msg=False)
    status_code = prob.solve(solver)
    status = pulp.LpStatus[status_code]

    if status != "Optimal":
        return {
            "success": False,
            "message": status,
            "x": None,
            "objective": np.nan,
            "duals": None,
            "slacks": None,
        }

    x_values = np.array([v.varValue for v in variables], dtype=float)
    objective = float(pulp.value(prob.objective))

    dual_rows = []
    for name, constraint in prob.constraints.items():
        dual_rows.append(
            {
                "Ràng buộc": name,
                "Shadow price": getattr(constraint, "pi", np.nan),
                "Slack": getattr(constraint, "slack", np.nan),
            }
        )

    dual_df = pd.DataFrame(dual_rows)

    return {
        "success": True,
        "message": status,
        "x": x_values,
        "objective": objective,
        "duals": dual_df,
        "raw": prob,
    }


def solution_table(x, objective):
    data = get_problem_data().copy()
    data["Phân bổ tối ưu"] = x
    data["Đóng góp Z"] = data["Hệ số tác động"] * data["Phân bổ tối ưu"]

    summary = pd.DataFrame(
        {
            "Chỉ tiêu": ["Tổng ngân sách sử dụng", "Giá trị tối ưu Z"],
            "Giá trị": [data["Phân bổ tối ưu"].sum(), objective],
        }
    )

    return data, summary


def sensitivity_budget_curve(budgets):
    rows = []

    for budget in budgets:
        res = solve_with_scipy(budget=float(budget), x3_min=20.0)

        if res["success"]:
            rows.append(
                {
                    "Ngân sách": float(budget),
                    "Z tối ưu": res["objective"],
                    "x1": res["x"][0],
                    "x2": res["x"][1],
                    "x3": res["x"][2],
                    "x4": res["x"][3],
                }
            )
        else:
            rows.append(
                {
                    "Ngân sách": float(budget),
                    "Z tối ưu": np.nan,
                    "x1": np.nan,
                    "x2": np.nan,
                    "x3": np.nan,
                    "x4": np.nan,
                }
            )

    return pd.DataFrame(rows)


def scenario_x3_priority(x3_min_value):
    base = solve_with_scipy(budget=100.0, x3_min=20.0)
    new = solve_with_scipy(budget=100.0, x3_min=x3_min_value)

    rows = []

    for label, res in [
        ("Gốc: x3 >= 20", base),
        (f"Ưu tiên nhân lực số: x3 >= {x3_min_value:.0f}", new),
    ]:
        if res["success"]:
            rows.append(
                {
                    "Kịch bản": label,
                    "Khả thi": "Có",
                    "Z tối ưu": res["objective"],
                    "x1": res["x"][0],
                    "x2": res["x"][1],
                    "x3": res["x"][2],
                    "x4": res["x"][3],
                }
            )
        else:
            rows.append(
                {
                    "Kịch bản": label,
                    "Khả thi": "Không",
                    "Z tối ưu": np.nan,
                    "x1": np.nan,
                    "x2": np.nan,
                    "x3": np.nan,
                    "x4": np.nan,
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


def render():
    st.title("Bài 2. Quy hoạch tuyến tính phân bổ ngân sách")
    st.caption("SciPy linprog, PuLP, shadow price, độ nhạy ngân sách và ưu tiên nhân lực số")

    if not SCIPY_AVAILABLE:
        st.error("Chưa cài scipy. Hãy thêm scipy vào requirements.txt.")
        return

    alpha_labels = get_problem_data()

    res_scipy = solve_with_scipy(budget=100.0, x3_min=20.0)
    res_pulp = solve_with_pulp(budget=100.0, x3_min=20.0)

    tabs = st.tabs(
        [
            "Phân bổ ngân sách tối ưu",
            "Shadow price",
            "Độ nhạy ngân sách",
            "Ưu tiên nhân lực số x3",
            "Thảo luận chính sách",
        ]
    )

    # =====================================================
    # 2.4.1
    # =====================================================
    with tabs[0]:
    

        show_table(alpha_labels, decimals=3)

        if res_scipy["success"]:
            sol_df, summary_df = solution_table(res_scipy["x"], res_scipy["objective"])

            c1, c2, c3 = st.columns(3)
            c1.metric("Trạng thái", "Optimal")
            c2.metric("Z tối ưu", f"{res_scipy['objective']:.2f}")
            c3.metric("Ngân sách dùng", f"{res_scipy['x'].sum():.1f}")

            show_table(sol_df, decimals=3)
            show_table(summary_df, decimals=3)

            fig = px.bar(
                sol_df,
                x="Hạng mục",
                y="Phân bổ tối ưu",
                text=sol_df["Phân bổ tối ưu"].round(1),
                title="Phân bổ ngân sách tối ưu theo SciPy",
                color="Hạng mục",
                color_discrete_map=MULTI_COLORS,
            )
            fig.update_traces(
                textposition="outside",
                marker_line_color="white",
                marker_line_width=1,
            )
            fig.update_layout(
                xaxis_title="Hạng mục",
                yaxis_title="Nghìn tỷ VND",
                showlegend=False,
            )
            style_base_fig(fig, height=430)
            st.plotly_chart(fig, use_container_width=True)

            st.success(
                f"Phương án tối ưu là phân bổ nhiều nhất cho R&D vì hệ số tác động cao nhất. "
                f"Giá trị tối ưu Z = {res_scipy['objective']:.2f}."
            )
        else:
            st.error(f"Bài toán không giải được: {res_scipy['message']}")

    # =====================================================
    # 2.4.2
    # =====================================================
    with tabs[1]:
       

        if res_pulp["success"]:
            sol_df, summary_df = solution_table(res_pulp["x"], res_pulp["objective"])

            c1, c2, c3 = st.columns(3)
            c1.metric("Trạng thái", "Optimal")
            c2.metric("Z tối ưu", f"{res_pulp['objective']:.2f}")
            c3.metric("Ngân sách dùng", f"{res_pulp['x'].sum():.1f}")

            st.subheader("Phân bổ tối ưu PuLP")
            show_table(sol_df, decimals=3)

            st.subheader("Giá đối ngẫu của ràng buộc")
            dual_df = res_pulp["duals"].copy()
            show_table(dual_df, decimals=4)

            budget_shadow = dual_df.loc[
                dual_df["Ràng buộc"] == "Ngan_sach_tong", "Shadow price"
            ]

            if len(budget_shadow) > 0:
                shadow_value = float(budget_shadow.iloc[0])
            else:
                shadow_value = np.nan

            st.success(
                f"Shadow price của ràng buộc ngân sách tổng là khoảng {shadow_value:.2f}. "
                f"Nghĩa là nếu tăng thêm 1 nghìn tỷ VND ngân sách, Z tối ưu tăng thêm khoảng "
                f"{shadow_value:.2f}, trong vùng nghiệm hiện tại."
            )

            fig = px.bar(
                dual_df,
                x="Ràng buộc",
                y="Shadow price",
                text=dual_df["Shadow price"].round(2),
                title="Shadow price của các ràng buộc",
            )
            fig.update_traces(
                marker_color=BRAND,
                textposition="outside",
                textfont=dict(color=BRAND),
            )
            fig.update_layout(
                xaxis_title="Ràng buộc",
                yaxis_title="Shadow price",
            )
            style_base_fig(fig, height=430)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(res_pulp["message"])
            st.info("Để xem shadow price bằng PuLP, thêm dòng `pulp` vào requirements.txt.")

    # =====================================================
    # 2.4.3
    # =====================================================
    with tabs[2]:
     

        budgets_required = [100, 120, 140]
        sens_required = sensitivity_budget_curve(budgets_required)

        c1, c2, c3 = st.columns(3)
        c1.metric("Z*(100)", f"{sens_required.loc[0, 'Z tối ưu']:.2f}")
        c2.metric("Z*(120)", f"{sens_required.loc[1, 'Z tối ưu']:.2f}")
        c3.metric("Z*(140)", f"{sens_required.loc[2, 'Z tối ưu']:.2f}")

        show_table(sens_required, decimals=3)

        fig = px.line(
            sens_required,
            x="Ngân sách",
            y="Z tối ưu",
            markers=True,
            title="Đường cong Z*(B) theo ngân sách tổng",
        )
        fig.update_traces(
            line=dict(color=BRAND, width=4),
            marker=dict(color=BRAND, size=10),
        )
        fig.update_layout(
            xaxis_title="Ngân sách B, nghìn tỷ VND",
            yaxis_title="Z*(B)",
        )
        style_base_fig(fig, height=430)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Điều chỉnh khoảng ngân sách")

        p1, p2, p3 = st.columns(3)
        with p1:
            b_min = st.slider("Ngân sách nhỏ nhất", 70, 150, 100, 5)
        with p2:
            b_max = st.slider("Ngân sách lớn nhất", 100, 220, 160, 5)
        with p3:
            b_step = st.slider("Bước nhảy", 5, 20, 10, 5)

        if b_max <= b_min:
            st.warning("Ngân sách lớn nhất phải lớn hơn ngân sách nhỏ nhất.")
        else:
            budgets_dynamic = list(range(int(b_min), int(b_max) + 1, int(b_step)))
            sens_dynamic = sensitivity_budget_curve(budgets_dynamic)

            show_table(sens_dynamic, decimals=3)

            fig_dynamic = px.line(
                sens_dynamic,
                x="Ngân sách",
                y="Z tối ưu",
                markers=True,
                title="Đường cong Z*(B) tùy chỉnh",
            )
            fig_dynamic.update_traces(
                line=dict(color=BRAND, width=4),
                marker=dict(color=BRAND, size=9),
            )
            fig_dynamic.update_layout(
                xaxis_title="Ngân sách B, nghìn tỷ VND",
                yaxis_title="Z*(B)",
            )
            style_base_fig(fig_dynamic, height=430)
            st.plotly_chart(fig_dynamic, use_container_width=True)

        st.success(
            "Khi ngân sách tăng, Z* tăng gần tuyến tính vì phần ngân sách tăng thêm được đưa vào hạng mục có hệ số tác động cao nhất."
        )

    # =====================================================
    # 2.4.4
    # =====================================================
    with tabs[3]:
      

        x3_priority = st.slider(
            "Ràng buộc nhân lực số tối thiểu x3",
            20.0,
            70.0,
            30.0,
            1.0,
        )

        scenario_df = scenario_x3_priority(x3_priority)

        show_table(scenario_df, decimals=3)

        feasible_now = scenario_df.iloc[1]["Khả thi"] == "Có"

        if feasible_now:
            z_base = scenario_df.iloc[0]["Z tối ưu"]
            z_new = scenario_df.iloc[1]["Z tối ưu"]
            delta_z = z_new - z_base

            c1, c2, c3 = st.columns(3)
            c1.metric("Khả thi", "Có")
            c2.metric("Z mới", f"{z_new:.2f}")
            c3.metric("Thay đổi Z", f"{delta_z:.2f}")

            plot_df = scenario_df.melt(
                id_vars=["Kịch bản"],
                value_vars=["x1", "x2", "x3", "x4"],
                var_name="Biến",
                value_name="Phân bổ",
            )

            name_map = {
                "x1": "Hạ tầng số",
                "x2": "AI",
                "x3": "Nhân lực số",
                "x4": "R&D",
            }
            plot_df["Hạng mục"] = plot_df["Biến"].map(name_map)

            fig = px.bar(
                plot_df,
                x="Hạng mục",
                y="Phân bổ",
                color="Kịch bản",
                barmode="group",
                title="So sánh phân bổ khi ưu tiên nhân lực số",
            )
            fig.update_layout(
                xaxis_title="Hạng mục",
                yaxis_title="Nghìn tỷ VND",
            )
            style_base_fig(fig, height=430)
            st.plotly_chart(fig, use_container_width=True)

            st.success(
                f"Khi đặt x3 >= {x3_priority:.0f}, bài toán vẫn khả thi. "
                f"Z thay đổi {delta_z:.2f} so với phương án gốc."
            )
        else:
            st.error(
                f"Khi đặt x3 >= {x3_priority:.0f}, bài toán không còn khả thi với ngân sách 100."
            )

    # =====================================================
    # 2.5
    # =====================================================
    with tabs[4]:
     

        if res_pulp["success"]:
            dual_df = res_pulp["duals"]
            budget_shadow_series = dual_df.loc[
                dual_df["Ràng buộc"] == "Ngan_sach_tong", "Shadow price"
            ]
            if len(budget_shadow_series) > 0:
                budget_shadow = float(budget_shadow_series.iloc[0])
            else:
                budget_shadow = 1.35
        else:
            budget_shadow = 1.35

        st.markdown("### a) Khi ngân sách tăng thêm 1 nghìn tỷ VND, GDP kỳ vọng tăng thêm bao nhiêu?")

        st.success(
            f"Shadow price của ngân sách tổng xấp xỉ **{budget_shadow:.2f}**."
        )

        st.markdown(
            f"""
            Điều này có nghĩa là trong vùng nghiệm hiện tại, nếu Chính phủ tăng thêm **1 nghìn tỷ VND**
            vào ngân sách tổng, giá trị mục tiêu kỳ vọng **Z** tăng thêm khoảng **{budget_shadow:.2f}** đơn vị.

            Đây có thể xem là một chỉ báo về **lợi ích biên của vốn công** trong mô hình.
            Tuy nhiên, nó chỉ là cận biên trong phạm vi các giả định tuyến tính và trong vùng nghiệm hiện tại.
            Trong thực tế, khi ngân sách tăng quá nhiều, hiệu quả biên có thể giảm do năng lực hấp thụ vốn,
            độ trễ triển khai, thủ tục đầu tư công, giới hạn nhân lực và rủi ro phân bổ kém hiệu quả.

            Vì vậy, shadow price là một tham chiếu hữu ích để đánh giá chi phí cơ hội,
            nhưng không nên hiểu là lợi ích chắc chắn cho mọi mức tăng ngân sách.
            """
        )

        st.markdown("### b) Vì sao R&D có hệ số tác động cao nhất nhưng ràng buộc tối thiểu thấp nhất?")

        st.markdown(
            """
            R&D có hệ số tác động cao nhất vì đầu tư cho nghiên cứu, đổi mới sáng tạo và công nghệ lõi
            thường tạo hiệu ứng lan tỏa lớn đến năng suất, doanh nghiệp và năng lực cạnh tranh dài hạn.

            Tuy nhiên, ràng buộc tối thiểu của R&D lại thấp có thể được giải thích bởi ba lý do:

            - **R&D có rủi ro cao và thời gian hoàn vốn dài.** Không phải mọi khoản đầu tư R&D đều tạo kết quả ngay.
            - **Năng lực hấp thụ R&D còn giới hạn.** Nếu thiếu nhân lực, phòng thí nghiệm, dữ liệu và doanh nghiệp đủ năng lực,
            tăng vốn R&D quá nhanh có thể không tạo hiệu quả tương ứng.
            - **Ngân sách công phải cân bằng nhiều mục tiêu.** Nhà nước vẫn cần bảo đảm hạ tầng số, AI, nhân lực số
            và các nhiệm vụ xã hội khác.

            Do đó, hệ số cao không đồng nghĩa với việc đặt mức tối thiểu cao.
            R&D nên được ưu tiên tăng thêm ở phần ngân sách linh hoạt, sau khi các mức nền tảng tối thiểu được bảo đảm.
            """
        )

        st.markdown("### c) Tỷ lệ 35% công nghệ chiến lược AI + R&D có khả thi không?")

        st.success(
            "Trong nghiệm tối ưu của mô hình, điều kiện tỷ lệ công nghệ chiến lược là khả thi."
        )

        st.markdown(
            """
            Ràng buộc 35% cho AI + R&D phản ánh mục tiêu ưu tiên công nghệ chiến lược.
            Về mặt mô hình, ràng buộc này khả thi vì tổng mức tối thiểu của các hạng mục vẫn nhỏ hơn ngân sách tổng,
            đồng thời AI và R&D là các hạng mục có hệ số tác động cao.

            Tuy nhiên, trong thực tiễn quản lý ngân sách, mục tiêu này không đơn giản.
            Ngân sách nhà nước Việt Nam vẫn phải ưu tiên hạ tầng giao thông, y tế, giáo dục, an sinh xã hội,
            quốc phòng, an ninh và các chương trình phục hồi kinh tế. Vì vậy, dành tỷ trọng lớn cho AI và R&D
            cần được triển khai theo lộ trình.

            Điều kiện để tỷ lệ 35% khả thi hơn gồm:

            - Có danh mục dự án AI và R&D đủ chất lượng.
            - Có cơ chế giám sát hiệu quả đầu tư.
            - Có nhân lực kỹ thuật đủ năng lực triển khai.
            - Có phối hợp giữa ngân sách nhà nước, doanh nghiệp và vốn tư nhân.
            - Có tiêu chí đo lường kết quả rõ ràng, tránh đầu tư hình thức.

            **Kết luận:** tỷ lệ 35% là khả thi về mặt mô hình, nhưng trong thực tế cần đi kèm năng lực hấp thụ,
            cơ chế lựa chọn dự án tốt và cân đối với các ưu tiên ngân sách khác.
            """
        )


def run():
    render()

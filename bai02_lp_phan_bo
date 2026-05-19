# bai02_lp_phan_bo.py
# Bài 2 — Phân bổ ngân sách đơn giản theo 4 hạng mục đầu tư số
# Module này dùng được với streamlit_app.py có cơ chế gọi module.render()

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


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


def solve_with_scipy(total_budget=100.0, min_x3=20.0):
    """
    Giải bài toán LP bằng scipy.optimize.linprog.

    Bài toán gốc:
    max Z = 0.85*x1 + 1.20*x2 + 0.95*x3 + 1.35*x4

    Ràng buộc:
    x1 + x2 + x3 + x4 <= B
    x1 >= 25
    x2 >= 15
    x3 >= min_x3
    x4 >= 10
    x2 + x4 >= 0.35*(x1+x2+x3+x4)
    xj >= 0

    Do scipy.linprog chỉ giải min, ta đổi max Z thành min -Z.
    """

    if not SCIPY_AVAILABLE:
        return {
            "success": False,
            "message": "scipy chưa được cài đặt trong môi trường.",
            "x": None,
            "objective": None,
            "raw": None
        }

    c = [-0.85, -1.20, -0.95, -1.35]

    # A_ub @ x <= b_ub
    A_ub = [
        [1, 1, 1, 1],                     # ngân sách tổng
        [-1, 0, 0, 0],                    # x1 >= 25
        [0, -1, 0, 0],                    # x2 >= 15
        [0, 0, -1, 0],                    # x3 >= min_x3
        [0, 0, 0, -1],                    # x4 >= 10
        [0.35, -0.65, 0.35, -0.65],       # x2+x4 >= 0.35*sum(x)
    ]

    b_ub = [
        total_budget,
        -25,
        -15,
        -min_x3,
        -10,
        0
    ]

    bounds = [(0, None)] * 4

    res = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs"
    )

    if res.success:
        return {
            "success": True,
            "message": res.message,
            "x": res.x,
            "objective": -res.fun,
            "raw": res
        }

    return {
        "success": False,
        "message": res.message,
        "x": None,
        "objective": None,
        "raw": res
    }


def solve_with_pulp(total_budget=100.0, min_x3=20.0):
    """
    Giải lại bài toán bằng PuLP.
    Lưu ý:
    - PuLP/CBC có thể trả về shadow price qua constraint.pi với bài toán LP liên tục.
    - Nếu môi trường không hỗ trợ dual, app vẫn chạy và hiện thông báo.
    """

    if not PULP_AVAILABLE:
        return {
            "success": False,
            "message": "PuLP chưa được cài đặt trong môi trường.",
            "x": None,
            "objective": None,
            "duals": None,
            "status": None
        }

    model = pulp.LpProblem("Bai2_LP_Ngan_Sach_So", pulp.LpMaximize)

    x1 = pulp.LpVariable("x1_Ha_tang_so", lowBound=0, cat="Continuous")
    x2 = pulp.LpVariable("x2_AI_du_lieu", lowBound=0, cat="Continuous")
    x3 = pulp.LpVariable("x3_Nhan_luc_so", lowBound=0, cat="Continuous")
    x4 = pulp.LpVariable("x4_RD_cong_nghe", lowBound=0, cat="Continuous")

    # Hàm mục tiêu
    model += 0.85 * x1 + 1.20 * x2 + 0.95 * x3 + 1.35 * x4, "Tong_GDP_gain"

    # Ràng buộc
    model += x1 + x2 + x3 + x4 <= total_budget, "C1_Ngan_sach_tong"
    model += x1 >= 25, "C2_Ha_tang_so_toi_thieu"
    model += x2 >= 15, "C3_AI_du_lieu_toi_thieu"
    model += x3 >= min_x3, "C4_Nhan_luc_so_toi_thieu"
    model += x4 >= 10, "C5_RD_toi_thieu"

    # x2 + x4 >= 0.35*(x1+x2+x3+x4)
    # Chuyển vế: 0.35*x1 - 0.65*x2 + 0.35*x3 - 0.65*x4 <= 0
    model += 0.35 * x1 - 0.65 * x2 + 0.35 * x3 - 0.65 * x4 <= 0, "C6_Ty_trong_cong_nghe_chien_luoc"

    solver = pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)

    status = pulp.LpStatus[model.status]

    if status != "Optimal":
        return {
            "success": False,
            "message": f"PuLP không tìm được nghiệm tối ưu. Trạng thái: {status}",
            "x": None,
            "objective": None,
            "duals": None,
            "status": status
        }

    x_values = np.array([
        pulp.value(x1),
        pulp.value(x2),
        pulp.value(x3),
        pulp.value(x4)
    ])

    objective = pulp.value(model.objective)

    dual_rows = []

    for name, constraint in model.constraints.items():
        dual_value = getattr(constraint, "pi", None)
        slack_value = getattr(constraint, "slack", None)

        dual_rows.append({
            "Ràng buộc": name,
            "Shadow price / Dual": dual_value,
            "Slack": slack_value
        })

    duals_df = pd.DataFrame(dual_rows)

    return {
        "success": True,
        "message": "Optimal",
        "x": x_values,
        "objective": objective,
        "duals": duals_df,
        "status": status
    }


def make_solution_table(x, objective):
    names = [
        "Hạ tầng số",
        "AI và dữ liệu",
        "Nhân lực số",
        "R&D công nghệ"
    ]

    symbols = ["x1", "x2", "x3", "x4"]

    coef = [0.85, 1.20, 0.95, 1.35]

    df = pd.DataFrame({
        "Biến": symbols,
        "Hạng mục đầu tư": names,
        "Hệ số tác động GDP": coef,
        "Phân bổ tối ưu": x,
        "GDP gain đóng góp": np.array(coef) * np.array(x)
    })

    total_budget_used = df["Phân bổ tối ưu"].sum()
    strategic_share = (
        df.loc[df["Biến"].isin(["x2", "x4"]), "Phân bổ tối ưu"].sum()
        / total_budget_used
        * 100
        if total_budget_used > 0
        else np.nan
    )

    summary = {
        "Z_star": objective,
        "total_budget_used": total_budget_used,
        "strategic_share": strategic_share
    }

    return df, summary


def analytical_solution(total_budget=100.0, min_x3=20.0):
    """
    Nghiệm phân tích nhanh dùng để giải thích chính sách.

    Vì x4 có hệ số mục tiêu cao nhất 1.35, sau khi thỏa mãn các mức tối thiểu,
    phần ngân sách còn lại sẽ dồn vào x4 nếu không vi phạm ràng buộc tỷ trọng công nghệ.
    Với dữ kiện bài này, ràng buộc tỷ trọng công nghệ không chặt tại nghiệm tối ưu.
    """

    x1 = 25.0
    x2 = 15.0
    x3 = float(min_x3)
    x4 = 10.0

    minimum_sum = x1 + x2 + x3 + x4

    if minimum_sum > total_budget:
        return None, None

    x4 += total_budget - minimum_sum

    x = np.array([x1, x2, x3, x4])
    z = 0.85 * x1 + 1.20 * x2 + 0.95 * x3 + 1.35 * x4

    return x, z


def sensitivity_analysis(budgets, min_x3=20.0):
    rows = []

    for B in budgets:
        scipy_result = solve_with_scipy(total_budget=B, min_x3=min_x3)

        if scipy_result["success"]:
            x = scipy_result["x"]
            z = scipy_result["objective"]
            method = "SciPy linprog"
            feasible = True
        else:
            x, z = analytical_solution(total_budget=B, min_x3=min_x3)
            method = "Analytical fallback"
            feasible = x is not None

        if feasible:
            rows.append({
                "Ngân sách B": B,
                "x1 - Hạ tầng số": x[0],
                "x2 - AI và dữ liệu": x[1],
                "x3 - Nhân lực số": x[2],
                "x4 - R&D công nghệ": x[3],
                "Z*(B)": z,
                "Phương pháp": method,
                "Khả thi": "Có"
            })
        else:
            rows.append({
                "Ngân sách B": B,
                "x1 - Hạ tầng số": np.nan,
                "x2 - AI và dữ liệu": np.nan,
                "x3 - Nhân lực số": np.nan,
                "x4 - R&D công nghệ": np.nan,
                "Z*(B)": np.nan,
                "Phương pháp": method,
                "Khả thi": "Không"
            })

    return pd.DataFrame(rows)


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
            <h1>💰 Bài 2 — Phân bổ ngân sách đơn giản theo 4 hạng mục đầu tư số</h1>
            <p>
            Bài toán quy hoạch tuyến tính LP phân bổ ngân sách trung ương năm 2026 cho:
            hạ tầng số, AI và dữ liệu, nhân lực số, R&D công nghệ.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "Đơn vị của biến quyết định là nghìn tỷ VND. "
        "Hàm mục tiêu đo GDP gain kỳ vọng, cũng theo nghìn tỷ VND."
    )

    with st.sidebar:
        st.markdown("### ⚙️ Tham số Bài 2")

        total_budget = st.number_input(
            "Ngân sách tổng B",
            min_value=50.0,
            max_value=300.0,
            value=100.0,
            step=10.0
        )

        min_x1 = 25.0
        min_x2 = 15.0

        min_x3 = st.number_input(
            "Sàn nhân lực số x3",
            min_value=0.0,
            max_value=100.0,
            value=20.0,
            step=5.0
        )

        min_x4 = 10.0

        st.caption("Trong đề bài gốc: x1 ≥ 25, x2 ≥ 15, x3 ≥ 20, x4 ≥ 10.")

    # Cảnh báo nếu người dùng thay x3 nhưng hàm solver đang dùng min_x3
    min_required_sum = min_x1 + min_x2 + min_x3 + min_x4

    if min_required_sum > total_budget:
        st.error(
            f"Bài toán không khả thi vì tổng mức tối thiểu = {min_required_sum:.1f} "
            f"lớn hơn ngân sách B = {total_budget:.1f}."
        )
        st.stop()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📌 Mô hình",
        "2.4.1 SciPy linprog",
        "2.4.2 PuLP & Shadow price",
        "2.4.3 Độ nhạy ngân sách",
        "2.4.4 Ưu tiên nhân lực số",
        "🧠 Thảo luận chính sách"
    ])

    # =========================
    # TAB 1: MÔ HÌNH
    # =========================

    with tab1:
        st.header("1. Mô hình toán học")

        st.markdown("### Biến quyết định")

        variables_df = pd.DataFrame({
            "Biến": ["x1", "x2", "x3", "x4"],
            "Ý nghĩa": [
                "Đầu tư hạ tầng số",
                "Đầu tư AI và dữ liệu",
                "Đầu tư nhân lực số",
                "Đầu tư R&D công nghệ"
            ],
            "Đơn vị": ["Nghìn tỷ VND"] * 4
        })

        st.dataframe(variables_df, use_container_width=True)

        st.markdown("### Hàm mục tiêu")

        st.latex(r"\max Z = 0.85x_1 + 1.20x_2 + 0.95x_3 + 1.35x_4")

        st.markdown("### Hệ ràng buộc")

        constraints_df = pd.DataFrame({
            "Mã": ["C1", "C2", "C3", "C4", "C5", "C6", "C7"],
            "Ràng buộc": [
                r"x1 + x2 + x3 + x4 ≤ B",
                r"x1 ≥ 25",
                r"x2 ≥ 15",
                r"x3 ≥ 20 hoặc giá trị người dùng chọn",
                r"x4 ≥ 10",
                r"x2 + x4 ≥ 0.35(x1+x2+x3+x4)",
                r"x1, x2, x3, x4 ≥ 0"
            ],
            "Ý nghĩa": [
                "Ngân sách tổng",
                "Hạ tầng số tối thiểu",
                "AI và dữ liệu tối thiểu",
                "Nhân lực số tối thiểu",
                "R&D tối thiểu",
                "Tỷ trọng công nghệ chiến lược tối thiểu 35%",
                "Không âm"
            ]
        })

        st.dataframe(constraints_df, use_container_width=True)

        st.markdown("### Dạng chuẩn dùng cho scipy.optimize.linprog")

        st.latex(r"\min -Z = -0.85x_1 -1.20x_2 -0.95x_3 -1.35x_4")

        st.latex(
            r"0.35x_1 -0.65x_2 +0.35x_3 -0.65x_4 \leq 0"
        )

        st.success(
            "Vì đây là bài toán LP tuyến tính, nghiệm tối ưu thường nằm ở một đỉnh của miền khả thi. "
            "Trong bộ hệ số này, R&D có hệ số cao nhất nên sau khi thỏa mãn các sàn tối thiểu, "
            "phần ngân sách còn lại có xu hướng được phân bổ vào R&D."
        )

    # =========================
    # TAB 2: SCIPY
    # =========================

    with tab2:
        st.header("2.4.1 — Giải bằng scipy.optimize.linprog")

        scipy_result = solve_with_scipy(total_budget=total_budget, min_x3=min_x3)

        if scipy_result["success"]:
            x = scipy_result["x"]
            z = scipy_result["objective"]
            sol_df, summary = make_solution_table(x, z)

            c1, c2, c3 = st.columns(3)
            c1.metric("Trạng thái", "Optimal")
            c2.metric("Giá trị tối ưu Z*", f"{z:,.2f}")
            c3.metric("Tỷ trọng AI + R&D", f"{summary['strategic_share']:.2f}%")

            st.markdown("### Phân bổ tối ưu")
            st.dataframe(sol_df.round(4), use_container_width=True)

            fig = px.bar(
                sol_df,
                x="Hạng mục đầu tư",
                y="Phân bổ tối ưu",
                text=sol_df["Phân bổ tối ưu"].round(2),
                title="Phân bổ ngân sách tối ưu theo SciPy"
            )
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.bar(
                sol_df,
                x="Hạng mục đầu tư",
                y="GDP gain đóng góp",
                text=sol_df["GDP gain đóng góp"].round(2),
                title="Đóng góp vào hàm mục tiêu Z theo từng hạng mục"
            )
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("### Kiểm tra ràng buộc")

            check_df = pd.DataFrame({
                "Ràng buộc": [
                    "Tổng ngân sách",
                    "x1 ≥ 25",
                    "x2 ≥ 15",
                    f"x3 ≥ {min_x3:.0f}",
                    "x4 ≥ 10",
                    "AI + R&D ≥ 35% tổng ngân sách"
                ],
                "Giá trị kiểm tra": [
                    x.sum(),
                    x[0],
                    x[1],
                    x[2],
                    x[3],
                    (x[1] + x[3]) / x.sum() * 100
                ],
                "Ngưỡng": [
                    f"≤ {total_budget}",
                    "≥ 25",
                    "≥ 15",
                    f"≥ {min_x3:.0f}",
                    "≥ 10",
                    "≥ 35%"
                ]
            })

            st.dataframe(check_df, use_container_width=True)

        else:
            st.error("SciPy không giải được bài toán.")
            st.write(scipy_result["message"])

            x, z = analytical_solution(total_budget=total_budget, min_x3=min_x3)

            if x is not None:
                st.warning("Dùng nghiệm phân tích dự phòng.")
                sol_df, summary = make_solution_table(x, z)
                st.dataframe(sol_df.round(4), use_container_width=True)
                st.metric("Z* dự phòng", f"{z:,.2f}")

    # =========================
    # TAB 3: PULP & DUAL
    # =========================

    with tab3:
        st.header("2.4.2 — Giải lại bằng PuLP và phân tích shadow price")

        pulp_result = solve_with_pulp(total_budget=total_budget, min_x3=min_x3)

        if pulp_result["success"]:
            x = pulp_result["x"]
            z = pulp_result["objective"]
            sol_df, summary = make_solution_table(x, z)

            c1, c2, c3 = st.columns(3)
            c1.metric("Trạng thái PuLP", pulp_result["status"])
            c2.metric("Giá trị tối ưu Z*", f"{z:,.2f}")
            c3.metric("Ngân sách sử dụng", f"{summary['total_budget_used']:,.2f}")

            st.markdown("### Phân bổ tối ưu bằng PuLP")
            st.dataframe(sol_df.round(4), use_container_width=True)

            st.markdown("### Shadow price / Dual values")

            duals = pulp_result["duals"].copy()

            st.dataframe(duals.round(6), use_container_width=True)

            budget_dual = None

            if "C1_Ngan_sach_tong" in list(duals["Ràng buộc"]):
                temp = duals[duals["Ràng buộc"] == "C1_Ngan_sach_tong"]
                if len(temp) > 0:
                    budget_dual = temp["Shadow price / Dual"].iloc[0]

            if budget_dual is not None and not pd.isna(budget_dual):
                st.success(
                    f"Shadow price của ràng buộc ngân sách tổng xấp xỉ {budget_dual:.4f}. "
                    "Diễn giải: nếu tăng thêm 1 đơn vị ngân sách, tức 1 nghìn tỷ VND, "
                    f"GDP gain kỳ vọng tăng thêm khoảng {budget_dual:.4f} nghìn tỷ VND, "
                    "miễn là cấu trúc nghiệm tối ưu chưa thay đổi."
                )
            else:
                st.info(
                    "Môi trường CBC/PuLP hiện tại chưa trả về dual value rõ ràng. "
                    "Về mặt kinh tế, với nghiệm hiện tại, ngân sách tăng thêm thường chảy vào R&D, "
                    "nên shadow price ngân sách xấp xỉ hệ số biên của R&D là 1.35."
                )

        else:
            st.warning(pulp_result["message"])

            st.info(
                "Nếu Streamlit Cloud báo thiếu PuLP, hãy thêm dòng `pulp` vào requirements.txt."
            )

    # =========================
    # TAB 4: ĐỘ NHẠY NGÂN SÁCH
    # =========================

    with tab4:
        st.header("2.4.3 — Phân tích độ nhạy ngân sách tổng")

        st.markdown(
            "Theo yêu cầu đề bài, tăng ngân sách tổng từ 100 lên 120, 140 nghìn tỷ "
            "và vẽ đường cong Z*(B)."
        )

        default_budgets = [100, 120, 140]

        extra_budget_max = st.slider(
            "Mở rộng thêm ngân sách tối đa để xem đường cong",
            min_value=140,
            max_value=300,
            value=200,
            step=10
        )

        budget_grid = list(range(100, int(extra_budget_max) + 1, 10))

        sens_required = sensitivity_analysis(default_budgets, min_x3=min_x3)
        sens_grid = sensitivity_analysis(budget_grid, min_x3=min_x3)

        st.markdown("### Bảng theo yêu cầu: B = 100, 120, 140")
        st.dataframe(sens_required.round(4), use_container_width=True)

        fig = px.line(
            sens_grid,
            x="Ngân sách B",
            y="Z*(B)",
            markers=True,
            title="Đường cong giá trị tối ưu Z*(B)"
        )
        st.plotly_chart(fig, use_container_width=True)

        fig_alloc = px.area(
            sens_grid,
            x="Ngân sách B",
            y=[
                "x1 - Hạ tầng số",
                "x2 - AI và dữ liệu",
                "x3 - Nhân lực số",
                "x4 - R&D công nghệ"
            ],
            title="Cơ cấu phân bổ tối ưu khi ngân sách tăng"
        )
        st.plotly_chart(fig_alloc, use_container_width=True)

        if len(sens_grid.dropna()) >= 2:
            z100 = sens_grid[sens_grid["Ngân sách B"] == 100]["Z*(B)"].iloc[0]
            z120 = sens_grid[sens_grid["Ngân sách B"] == 120]["Z*(B)"].iloc[0]
            marginal = (z120 - z100) / 20

            st.success(
                f"Từ B=100 đến B=120, Z* tăng từ {z100:.2f} lên {z120:.2f}. "
                f"Mức tăng bình quân là {marginal:.4f} nghìn tỷ GDP gain cho mỗi 1 nghìn tỷ ngân sách tăng thêm."
            )

    # =========================
    # TAB 5: x3 >= 30
    # =========================

    with tab5:
        st.header("2.4.4 — Ưu tiên nhân lực số: thêm ràng buộc x3 ≥ 30")

        base_case = solve_with_scipy(total_budget=100.0, min_x3=20.0)
        human_case = solve_with_scipy(total_budget=100.0, min_x3=30.0)

        rows = []

        for label, result in [
            ("Gốc: x3 ≥ 20", base_case),
            ("Ưu tiên nhân lực: x3 ≥ 30", human_case)
        ]:
            if result["success"]:
                x = result["x"]
                z = result["objective"]

                rows.append({
                    "Kịch bản": label,
                    "Khả thi": "Có",
                    "x1": x[0],
                    "x2": x[1],
                    "x3": x[2],
                    "x4": x[3],
                    "Z*": z,
                    "AI + R&D / Tổng, %": (x[1] + x[3]) / x.sum() * 100
                })
            else:
                rows.append({
                    "Kịch bản": label,
                    "Khả thi": "Không",
                    "x1": np.nan,
                    "x2": np.nan,
                    "x3": np.nan,
                    "x4": np.nan,
                    "Z*": np.nan,
                    "AI + R&D / Tổng, %": np.nan
                })

        compare_df = pd.DataFrame(rows)

        st.dataframe(compare_df.round(4), use_container_width=True)

        if base_case["success"] and human_case["success"]:
            delta_z = human_case["objective"] - base_case["objective"]

            c1, c2, c3 = st.columns(3)
            c1.metric("Z* gốc", f"{base_case['objective']:,.2f}")
            c2.metric("Z* khi x3 ≥ 30", f"{human_case['objective']:,.2f}")
            c3.metric("Thay đổi Z*", f"{delta_z:,.2f}")

            fig = px.bar(
                compare_df,
                x="Kịch bản",
                y=["x1", "x2", "x3", "x4"],
                barmode="stack",
                title="So sánh phân bổ ngân sách giữa hai kịch bản"
            )
            st.plotly_chart(fig, use_container_width=True)

            if delta_z < 0:
                st.warning(
                    f"Khi tăng yêu cầu nhân lực số từ x3 ≥ 20 lên x3 ≥ 30, "
                    f"Z* giảm {abs(delta_z):.2f}. "
                    "Lý do là một phần ngân sách bị chuyển từ R&D, hạng mục có hệ số 1.35, "
                    "sang nhân lực số, hạng mục có hệ số 0.95."
                )
            elif delta_z == 0:
                st.info(
                    "Z* không đổi trong kịch bản này. Điều đó cho thấy ràng buộc nhân lực mới không làm thay đổi nghiệm tối ưu."
                )
            else:
                st.success(
                    "Z* tăng khi ưu tiên nhân lực số. Trường hợp này có thể xảy ra nếu các ràng buộc khác khiến R&D không còn là nơi hấp thụ ngân sách tốt nhất."
                )
        else:
            st.error(
                "Một trong hai kịch bản không khả thi. Hãy kiểm tra tổng ngân sách và các mức tối thiểu."
            )

    # =========================
    # TAB 6: THẢO LUẬN
    # =========================

    with tab6:
        st.header("2.5 — Câu hỏi thảo luận chính sách")

        st.markdown("### a) Khi ngân sách tổng tăng thêm 1 nghìn tỷ VND, GDP kỳ vọng tăng thêm bao nhiêu?")

        st.write(
            "Trong nghiệm tối ưu của bài toán gốc, sau khi thỏa mãn các sàn tối thiểu, "
            "ngân sách còn lại được phân bổ vào R&D vì R&D có hệ số mục tiêu cao nhất là 1.35. "
            "Do đó, trong vùng nghiệm hiện tại, shadow price của ngân sách tổng xấp xỉ 1.35. "
            "Nói cách khác, tăng thêm 1 nghìn tỷ VND ngân sách có thể làm GDP gain kỳ vọng tăng khoảng "
            "1.35 nghìn tỷ VND, miễn là ràng buộc và cấu trúc nghiệm chưa thay đổi."
        )

        st.markdown("### b) Vì sao R&D có hệ số tác động cao nhất nhưng ràng buộc tối thiểu thấp nhất?")

        st.write(
            "R&D có tác động lan tỏa dài hạn, nhưng cũng có độ trễ, rủi ro thất bại và yêu cầu năng lực hấp thụ cao. "
            "Vì vậy, chính sách có thể đặt sàn tối thiểu thấp để tránh ép đầu tư dàn trải vào R&D khi hệ sinh thái chưa đủ mạnh, "
            "đồng thời để mô hình tự phân bổ thêm nếu R&D thật sự có hiệu quả biên cao."
        )

        st.markdown("### c) Tỷ lệ 35% công nghệ chiến lược AI + R&D có khả thi không?")

        st.write(
            "Trong nghiệm tối ưu, tỷ trọng AI + R&D thường vượt 35%, nên về mặt mô hình là khả thi. "
            "Tuy nhiên, trong thực tiễn ngân sách nhà nước còn phải cạnh tranh với hạ tầng giao thông, y tế, giáo dục, "
            "an sinh xã hội và quốc phòng. Do đó, tỷ lệ 35% nên được hiểu là mục tiêu chính sách có điều kiện, "
            "cần đi kèm lộ trình giải ngân, năng lực hấp thụ, cơ chế giám sát hiệu quả và đánh giá sau đầu tư."
        )

        st.markdown("### Kết luận ngắn")

        st.success(
            "Bài toán LP cho thấy nếu chỉ tối đa hóa GDP gain kỳ vọng theo hệ số biên, ngân sách sẽ ưu tiên mạnh cho R&D. "
            "Tuy nhiên, quyết định chính sách thực tế cần cân bằng giữa hiệu quả kinh tế, khả năng triển khai, "
            "nhân lực bổ trợ và rủi ro dài hạn."
        )

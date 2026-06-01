# bai04_lp_nganh_vung.py
# Bài 4 — Quy hoạch tuyến tính phân bổ ngân sách số theo ngành - vùng
# Module dùng được với streamlit_app.py có cơ chế gọi module.render()

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


try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except Exception:
    CVXPY_AVAILABLE = False


# =====================================================
# 1. DỮ LIỆU BÀI 4 THEO ĐỀ
# =====================================================

REGIONS = ["NMM", "RRD", "NCC", "CH", "SE", "MD"]

REGION_NAMES = {
    "NMM": "Trung du miền núi phía Bắc",
    "RRD": "Đồng bằng sông Hồng",
    "NCC": "Bắc Trung Bộ + DH Trung Bộ",
    "CH": "Tây Nguyên",
    "SE": "Đông Nam Bộ",
    "MD": "Đồng bằng sông Cửu Long",
}

ITEMS = ["I", "D", "AI", "H"]

ITEM_NAMES = {
    "I": "Hạ tầng số",
    "D": "Chuyển đổi số DN",
    "AI": "Năng lực AI",
    "H": "Nhân lực số",
}

BETA = {
    ("NMM", "I"): 1.15, ("NMM", "D"): 0.85, ("NMM", "AI"): 0.55, ("NMM", "H"): 1.30,
    ("RRD", "I"): 0.95, ("RRD", "D"): 1.25, ("RRD", "AI"): 1.40, ("RRD", "H"): 1.05,
    ("NCC", "I"): 1.05, ("NCC", "D"): 0.95, ("NCC", "AI"): 0.85, ("NCC", "H"): 1.15,
    ("CH", "I"): 1.20, ("CH", "D"): 0.75, ("CH", "AI"): 0.45, ("CH", "H"): 1.35,
    ("SE", "I"): 0.90, ("SE", "D"): 1.30, ("SE", "AI"): 1.55, ("SE", "H"): 1.00,
    ("MD", "I"): 1.10, ("MD", "D"): 0.85, ("MD", "AI"): 0.65, ("MD", "H"): 1.25,
}

D0 = {
    "NMM": 38,
    "RRD": 78,
    "NCC": 55,
    "CH": 32,
    "SE": 82,
    "MD": 48,
}


def beta_matrix_df():
    rows = []

    for r in REGIONS:
        row = {
            "Mã vùng": r,
            "Vùng": REGION_NAMES[r],
        }

        for j in ITEMS:
            row[f"{j} - {ITEM_NAMES[j]}"] = BETA[(r, j)]

        rows.append(row)

    return pd.DataFrame(rows)


def d0_df():
    return pd.DataFrame({
        "Mã vùng": REGIONS,
        "Vùng": [REGION_NAMES[r] for r in REGIONS],
        "Digital Index ban đầu D_r": [D0[r] for r in REGIONS],
    })


# =====================================================
# 2. GIẢI BẰNG PULP
# =====================================================

def solve_with_pulp(
    total_budget=50000.0,
    min_region=5000.0,
    max_region=12000.0,
    min_human=12000.0,
    gamma=0.002,
    lam=0.7,
    use_fairness=True,
):
    if not PULP_AVAILABLE:
        return {
            "success": False,
            "message": "PuLP chưa được cài đặt. Hãy thêm `pulp` vào requirements.txt.",
            "allocation": None,
            "objective": None,
            "digital_after": None,
            "status": None,
        }

    model = pulp.LpProblem("VN_Digital_Budget_Region", pulp.LpMaximize)

    x = pulp.LpVariable.dicts(
        "x",
        (REGIONS, ITEMS),
        lowBound=0,
        cat="Continuous"
    )

    # Hàm mục tiêu
    model += pulp.lpSum(
        BETA[(r, j)] * x[r][j]
        for r in REGIONS
        for j in ITEMS
    ), "GDP_gain"

    # C1. Ngân sách tổng
    model += pulp.lpSum(
        x[r][j]
        for r in REGIONS
        for j in ITEMS
    ) <= total_budget, "C1_Ngan_sach_tong"

    # C2, C3. Sàn và trần ngân sách mỗi vùng
    for r in REGIONS:
        model += pulp.lpSum(x[r][j] for j in ITEMS) >= min_region, f"C2_San_vung_{r}"
        model += pulp.lpSum(x[r][j] for j in ITEMS) <= max_region, f"C3_Tran_vung_{r}"

    # C4. Sàn nhân lực số
    model += pulp.lpSum(x[r]["H"] for r in REGIONS) >= min_human, "C4_San_nhan_luc_so"

    # C5. Công bằng vùng miền
    # D_r + gamma*x_D,r >= lambda * max_r(D_r + gamma*x_D,r)
    # Linear hóa bằng biến M.
    M = None

    if use_fairness:
        M = pulp.LpVariable("Dmax", lowBound=0, cat="Continuous")

        for r in REGIONS:
            model += D0[r] + gamma * x[r]["D"] <= M, f"C5a_Dmax_upper_{r}"

        for r in REGIONS:
            model += D0[r] + gamma * x[r]["D"] >= lam * M, f"C5b_Fairness_lower_{r}"

    solver = pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)

    status = pulp.LpStatus[model.status]

    if status != "Optimal":
        return {
            "success": False,
            "message": f"PuLP không tìm được nghiệm tối ưu. Trạng thái: {status}",
            "allocation": None,
            "objective": None,
            "digital_after": None,
            "status": status,
        }

    allocation = pd.DataFrame(index=REGIONS, columns=ITEMS, dtype=float)

    for r in REGIONS:
        for j in ITEMS:
            allocation.loc[r, j] = pulp.value(x[r][j])

    allocation.index.name = "Mã vùng"
    allocation_named = allocation.copy()
    allocation_named.index = [REGION_NAMES[r] for r in allocation.index]
    allocation_named.columns = [f"{j} - {ITEM_NAMES[j]}" for j in allocation.columns]

    objective = pulp.value(model.objective)

    digital_after = pd.DataFrame({
        "Mã vùng": REGIONS,
        "Vùng": [REGION_NAMES[r] for r in REGIONS],
        "D ban đầu": [D0[r] for r in REGIONS],
        "Đầu tư D": [allocation.loc[r, "D"] for r in REGIONS],
        "D sau đầu tư": [D0[r] + gamma * allocation.loc[r, "D"] for r in REGIONS],
    })

    if use_fairness and M is not None:
        digital_after["Dmax M"] = pulp.value(M)
        digital_after["Ngưỡng công bằng λM"] = lam * pulp.value(M)
        digital_after["Đạt C5?"] = digital_after["D sau đầu tư"] >= digital_after["Ngưỡng công bằng λM"] - 1e-6

    dual_rows = []

    for name, c in model.constraints.items():
        dual_rows.append({
            "Ràng buộc": name,
            "Shadow price / Dual": getattr(c, "pi", None),
            "Slack": getattr(c, "slack", None),
        })

    dual_df = pd.DataFrame(dual_rows)

    return {
        "success": True,
        "message": "Optimal",
        "allocation": allocation,
        "allocation_named": allocation_named,
        "objective": objective,
        "digital_after": digital_after,
        "duals": dual_df,
        "status": status,
    }


# =====================================================
# 3. GIẢI BẰNG CVXPY
# =====================================================

def solve_with_cvxpy(
    total_budget=50000.0,
    min_region=5000.0,
    max_region=12000.0,
    min_human=12000.0,
    gamma=0.002,
    lam=0.7,
    use_fairness=True,
):
    if not CVXPY_AVAILABLE:
        return {
            "success": False,
            "message": "CVXPY chưa được cài đặt. Hãy thêm `cvxpy` vào requirements.txt.",
            "allocation": None,
            "objective": None,
            "status": None,
        }

    beta = np.array([[BETA[(r, j)] for j in ITEMS] for r in REGIONS], dtype=float)
    d0 = np.array([D0[r] for r in REGIONS], dtype=float)

    x = cp.Variable((len(REGIONS), len(ITEMS)), nonneg=True)

    constraints = []

    # C1
    constraints.append(cp.sum(x) <= total_budget)

    # C2, C3
    region_sum = cp.sum(x, axis=1)
    constraints.append(region_sum >= min_region)
    constraints.append(region_sum <= max_region)

    # C4
    h_idx = ITEMS.index("H")
    constraints.append(cp.sum(x[:, h_idx]) >= min_human)

    # C5
    if use_fairness:
        d_idx = ITEMS.index("D")
        M = cp.Variable(nonneg=True)

        digital_after = d0 + gamma * x[:, d_idx]

        constraints.append(digital_after <= M)
        constraints.append(digital_after >= lam * M)

    objective = cp.Maximize(cp.sum(cp.multiply(beta, x)))
    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=cp.CLARABEL)
    except Exception:
        try:
            problem.solve(solver=cp.SCS)
        except Exception as e:
            return {
                "success": False,
                "message": f"CVXPY không giải được bài toán. Lỗi: {e}",
                "allocation": None,
                "objective": None,
                "status": None,
            }

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        return {
            "success": False,
            "message": f"CVXPY không tìm được nghiệm tối ưu. Trạng thái: {problem.status}",
            "allocation": None,
            "objective": None,
            "status": problem.status,
        }

    allocation = pd.DataFrame(x.value, index=REGIONS, columns=ITEMS)
    allocation.index.name = "Mã vùng"

    allocation_named = allocation.copy()
    allocation_named.index = [REGION_NAMES[r] for r in allocation.index]
    allocation_named.columns = [f"{j} - {ITEM_NAMES[j]}" for j in allocation.columns]

    digital_after_df = pd.DataFrame({
        "Mã vùng": REGIONS,
        "Vùng": [REGION_NAMES[r] for r in REGIONS],
        "D ban đầu": [D0[r] for r in REGIONS],
        "Đầu tư D": [allocation.loc[r, "D"] for r in REGIONS],
        "D sau đầu tư": [D0[r] + gamma * allocation.loc[r, "D"] for r in REGIONS],
    })

    return {
        "success": True,
        "message": "Optimal",
        "allocation": allocation,
        "allocation_named": allocation_named,
        "objective": float(problem.value),
        "digital_after": digital_after_df,
        "status": problem.status,
    }


# =====================================================
# 4. HÀM PHÂN TÍCH KẾT QUẢ
# =====================================================

def allocation_long_df(allocation: pd.DataFrame):
    df = allocation.copy()
    df.index.name = "Mã vùng"

    long_df = df.reset_index().melt(
        id_vars="Mã vùng",
        value_vars=ITEMS,
        var_name="Hạng mục",
        value_name="Ngân sách"
    )

    long_df["Vùng"] = long_df["Mã vùng"].map(REGION_NAMES)
    long_df["Tên hạng mục"] = long_df["Hạng mục"].map(ITEM_NAMES)

    return long_df


def summarize_allocation(allocation: pd.DataFrame, objective: float):
    region_total = allocation.sum(axis=1).reset_index()
    region_total.columns = ["Mã vùng", "Tổng ngân sách"]
    region_total["Vùng"] = region_total["Mã vùng"].map(REGION_NAMES)

    item_total = allocation.sum(axis=0).reset_index()
    item_total.columns = ["Hạng mục", "Tổng ngân sách"]
    item_total["Tên hạng mục"] = item_total["Hạng mục"].map(ITEM_NAMES)

    top_region_code = region_total.sort_values("Tổng ngân sách", ascending=False).iloc[0]["Mã vùng"]
    top_item_code = item_total.sort_values("Tổng ngân sách", ascending=False).iloc[0]["Hạng mục"]

    summary = {
        "objective": objective,
        "total_budget_used": allocation.values.sum(),
        "human_budget": allocation["H"].sum(),
        "top_region_code": top_region_code,
        "top_region_name": REGION_NAMES[top_region_code],
        "top_item_code": top_item_code,
        "top_item_name": ITEM_NAMES[top_item_code],
    }

    return region_total, item_total, summary


def compare_fairness_cost(
    total_budget=50000.0,
    min_region=5000.0,
    max_region=12000.0,
    min_human=12000.0,
    gamma=0.002,
    lam=0.7,
):
    fair = solve_with_pulp(
        total_budget=total_budget,
        min_region=min_region,
        max_region=max_region,
        min_human=min_human,
        gamma=gamma,
        lam=lam,
        use_fairness=True,
    )

    nofair = solve_with_pulp(
        total_budget=total_budget,
        min_region=min_region,
        max_region=max_region,
        min_human=min_human,
        gamma=gamma,
        lam=lam,
        use_fairness=False,
    )

    if not fair["success"] or not nofair["success"]:
        return fair, nofair, None

    cost = nofair["objective"] - fair["objective"]
    cost_pct = cost / nofair["objective"] * 100 if nofair["objective"] != 0 else np.nan

    compare_df = pd.DataFrame({
        "Mô hình": ["Có ràng buộc công bằng C5", "Không có ràng buộc công bằng C5"],
        "Z* GDP gain": [fair["objective"], nofair["objective"]],
        "Tổng ngân sách dùng": [
            fair["allocation"].values.sum(),
            nofair["allocation"].values.sum()
        ],
        "Ngân sách H": [
            fair["allocation"]["H"].sum(),
            nofair["allocation"]["H"].sum()
        ],
        "Ngân sách D": [
            fair["allocation"]["D"].sum(),
            nofair["allocation"]["D"].sum()
        ],
    })

    result = {
        "compare_df": compare_df,
        "cost": cost,
        "cost_pct": cost_pct,
    }

    return fair, nofair, result


# =====================================================
# 5. GIAO DIỆN STREAMLIT
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
            <h1>🗺️ Bài 4 — Quy hoạch tuyến tính phân bổ ngân sách số theo ngành-vùng</h1>
            <p>
            Module này giải bài toán phân bổ 50.000 tỷ VND ngân sách kinh tế số quốc gia
            cho 6 vùng kinh tế xã hội và 4 hạng mục đầu tư: I, D, AI, H.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "Đơn vị ngân sách trong module là tỷ VND. "
        "Hàm mục tiêu Z đo GDP gain kỳ vọng theo hệ số tác động biên β của từng vùng và hạng mục."
    )

    with st.sidebar:
        st.markdown("### ⚙️ Tham số Bài 4")

        total_budget = st.number_input(
            "Ngân sách tổng",
            min_value=20000.0,
            max_value=100000.0,
            value=50000.0,
            step=1000.0
        )

        min_region = st.number_input(
            "Sàn ngân sách mỗi vùng",
            min_value=0.0,
            max_value=20000.0,
            value=5000.0,
            step=500.0
        )

        max_region = st.number_input(
            "Trần ngân sách mỗi vùng",
            min_value=5000.0,
            max_value=30000.0,
            value=12000.0,
            step=500.0
        )

        min_human = st.number_input(
            "Sàn nhân lực số H toàn quốc",
            min_value=0.0,
            max_value=50000.0,
            value=12000.0,
            step=500.0
        )

        gamma = st.number_input(
            "γ - hiệu quả đầu tư D tới Digital Index",
            min_value=0.0001,
            max_value=0.0100,
            value=0.0020,
            step=0.0001,
            format="%.4f"
        )

        lam = st.number_input(
            "λ - hệ số công bằng vùng",
            min_value=0.10,
            max_value=1.00,
            value=0.70,
            step=0.05,
            format="%.2f"
        )

    if min_region * len(REGIONS) > total_budget:
        st.error(
            f"Bài toán không khả thi: sàn mỗi vùng {min_region:,.0f} × 6 "
            f"= {min_region * len(REGIONS):,.0f} lớn hơn ngân sách tổng {total_budget:,.0f}."
        )
        st.stop()

    if max_region * len(REGIONS) < total_budget:
        st.warning(
            "Tổng trần vùng nhỏ hơn ngân sách tổng, nên mô hình có thể không dùng hết ngân sách."
        )

    if min_human > total_budget:
        st.error("Bài toán không khả thi: sàn nhân lực số H lớn hơn ngân sách tổng.")
        st.stop()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📌 Mô hình",
        "4.4.1 PuLP",
        "4.4.2 CVXPY",
        "4.4.3 Heatmap",
        "4.4.4 Chi phí công bằng",
        "🧠 Thảo luận chính sách"
    ])

    # =====================================================
    # TAB 1 — MÔ HÌNH
    # =====================================================

    with tab1:
        st.header("1. Mô hình toán học")

        st.markdown("### Biến quyết định")

        st.latex(r"x_{j,r} \geq 0,\quad j \in \{I,D,AI,H\},\quad r \in \{1,\ldots,6\}")

        variable_df = pd.DataFrame({
            "Ký hiệu": ["I", "D", "AI", "H"],
            "Ý nghĩa": [
                "Đầu tư hạ tầng số",
                "Đầu tư chuyển đổi số doanh nghiệp",
                "Đầu tư năng lực AI",
                "Đầu tư nhân lực số"
            ]
        })

        st.dataframe(variable_df, use_container_width=True)

        st.markdown("### Hàm mục tiêu")

        st.latex(r"\max Z = \sum_r \sum_j \beta_{j,r} x_{j,r}")

        st.markdown("### Các ràng buộc chính")

        constraints_df = pd.DataFrame({
            "Mã": ["C1", "C2", "C3", "C4", "C5", "C6"],
            "Ràng buộc": [
                r"ΣrΣj xj,r ≤ 50.000",
                r"Σj xj,r ≥ 5.000 với mọi vùng r",
                r"Σj xj,r ≤ 12.000 với mọi vùng r",
                r"Σr xH,r ≥ 12.000",
                r"Dr + γxD,r ≥ λ maxr(Dr + γxD,r)",
                r"xj,r ≥ 0"
            ],
            "Ý nghĩa": [
                "Ngân sách tổng",
                "Sàn ngân sách chống bỏ quên vùng yếu",
                "Trần ngân sách chống tập trung quá mức",
                "Sàn đầu tư nhân lực số",
                "Công bằng vùng miền theo Digital Index sau đầu tư",
                "Không âm"
            ]
        })

        st.dataframe(constraints_df, use_container_width=True)

        st.markdown("### Ma trận hệ số tác động biên β")

        st.dataframe(beta_matrix_df(), use_container_width=True)

        st.markdown("### Digital Index ban đầu Dᵣ")

        st.dataframe(d0_df(), use_container_width=True)

    # =====================================================
    # TAB 2 — PULP
    # =====================================================

    with tab2:
        st.header("Câu 4.4.1 — Cài đặt và giải bằng PuLP")

        result_pulp = solve_with_pulp(
            total_budget=total_budget,
            min_region=min_region,
            max_region=max_region,
            min_human=min_human,
            gamma=gamma,
            lam=lam,
            use_fairness=True,
        )

        if not result_pulp["success"]:
            st.error(result_pulp["message"])
        else:
            allocation = result_pulp["allocation"]
            allocation_named = result_pulp["allocation_named"]
            objective = result_pulp["objective"]

            region_total, item_total, summary = summarize_allocation(allocation, objective)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Trạng thái", result_pulp["status"])
            c2.metric("Z* GDP gain", f"{objective:,.2f}")
            c3.metric("Tổng ngân sách dùng", f"{summary['total_budget_used']:,.0f}")
            c4.metric("Ngân sách H", f"{summary['human_budget']:,.0f}")

            st.markdown("### Ma trận phân bổ tối ưu 6×4")
            st.dataframe(allocation_named.round(3), use_container_width=True)

            st.markdown("### Tổng ngân sách theo vùng")
            st.dataframe(region_total.round(3), use_container_width=True)

            st.markdown("### Tổng ngân sách theo hạng mục")
            st.dataframe(item_total.round(3), use_container_width=True)

            st.markdown("### Digital Index sau đầu tư và kiểm tra C5")
            st.dataframe(result_pulp["digital_after"].round(4), use_container_width=True)

            st.markdown("### Shadow price / Dual values từ PuLP")
            st.dataframe(result_pulp["duals"].round(6), use_container_width=True)

            st.success(
                f"Vùng nhận ngân sách lớn nhất là {summary['top_region_name']}. "
                f"Hạng mục được ưu tiên nhiều nhất toàn quốc là {summary['top_item_name']}."
            )

    # =====================================================
    # TAB 3 — CVXPY
    # =====================================================

    with tab3:
        st.header("Câu 4.4.2 — Cài đặt lại bằng CVXPY và so sánh với PuLP")

        result_pulp = solve_with_pulp(
            total_budget=total_budget,
            min_region=min_region,
            max_region=max_region,
            min_human=min_human,
            gamma=gamma,
            lam=lam,
            use_fairness=True,
        )

        result_cvx = solve_with_cvxpy(
            total_budget=total_budget,
            min_region=min_region,
            max_region=max_region,
            min_human=min_human,
            gamma=gamma,
            lam=lam,
            use_fairness=True,
        )

        if not result_cvx["success"]:
            st.warning(result_cvx["message"])
            st.info("Nếu muốn chạy CVXPY trên Streamlit Cloud, hãy thêm `cvxpy` vào requirements.txt.")
        elif not result_pulp["success"]:
            st.error(result_pulp["message"])
        else:
            c1, c2, c3 = st.columns(3)

            c1.metric("Z* PuLP", f"{result_pulp['objective']:,.4f}")
            c2.metric("Z* CVXPY", f"{result_cvx['objective']:,.4f}")
            c3.metric(
                "Chênh lệch tuyệt đối",
                f"{abs(result_pulp['objective'] - result_cvx['objective']):,.6f}"
            )

            st.markdown("### Phân bổ tối ưu bằng CVXPY")
            st.dataframe(result_cvx["allocation_named"].round(3), use_container_width=True)

            diff = result_cvx["allocation"] - result_pulp["allocation"]

            diff_named = diff.copy()
            diff_named.index = [REGION_NAMES[r] for r in diff_named.index]
            diff_named.columns = [f"{j} - {ITEM_NAMES[j]}" for j in diff_named.columns]

            st.markdown("### Chênh lệch phân bổ CVXPY - PuLP")
            st.dataframe(diff_named.round(6), use_container_width=True)

            if abs(result_pulp["objective"] - result_cvx["objective"]) < 1e-2:
                st.success(
                    "PuLP và CVXPY cho giá trị mục tiêu gần như giống nhau. "
                    "Khác biệt nhỏ nếu có thường do sai số số học hoặc nghiệm tối ưu không duy nhất."
                )
            else:
                st.warning(
                    "PuLP và CVXPY cho kết quả khác đáng kể. Cần kiểm tra solver, ràng buộc và trạng thái nghiệm."
                )

    # =====================================================
    # TAB 4 — HEATMAP
    # =====================================================

    with tab4:
        st.header("Câu 4.4.3 — Vẽ heatmap phân bổ tối ưu")

        result_pulp = solve_with_pulp(
            total_budget=total_budget,
            min_region=min_region,
            max_region=max_region,
            min_human=min_human,
            gamma=gamma,
            lam=lam,
            use_fairness=True,
        )

        if not result_pulp["success"]:
            st.error(result_pulp["message"])
        else:
            allocation = result_pulp["allocation"]
            allocation_named = result_pulp["allocation_named"]

            fig_heat = px.imshow(
                allocation_named,
                text_auto=".0f",
                aspect="auto",
                title="Heatmap phân bổ ngân sách tối ưu 6 vùng × 4 hạng mục",
                labels=dict(x="Hạng mục", y="Vùng", color="Tỷ VND")
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            long_df = allocation_long_df(allocation)

            fig_stack = px.bar(
                long_df,
                x="Vùng",
                y="Ngân sách",
                color="Tên hạng mục",
                title="Cơ cấu ngân sách theo vùng",
                barmode="stack"
            )
            fig_stack.update_layout(xaxis_tickangle=-25)
            st.plotly_chart(fig_stack, use_container_width=True)

            region_total, item_total, summary = summarize_allocation(
                allocation,
                result_pulp["objective"]
            )

            c1, c2 = st.columns(2)
            c1.metric("Vùng nhận nhiều nhất", summary["top_region_name"])
            c2.metric("Hạng mục toàn quốc lớn nhất", summary["top_item_name"])

            st.markdown("### Hạng mục ưu tiên cao nhất ở từng vùng")

            best_item_rows = []

            for r in REGIONS:
                best_j = allocation.loc[r].idxmax()
                best_item_rows.append({
                    "Vùng": REGION_NAMES[r],
                    "Hạng mục ưu tiên lớn nhất": f"{best_j} - {ITEM_NAMES[best_j]}",
                    "Ngân sách": allocation.loc[r, best_j],
                    "Hệ số β": BETA[(r, best_j)]
                })

            best_item_df = pd.DataFrame(best_item_rows)

            st.dataframe(best_item_df.round(3), use_container_width=True)

    # =====================================================
    # TAB 5 — CHI PHÍ CÔNG BẰNG
    # =====================================================

    with tab5:
        st.header("Câu 4.4.4 — So sánh với mô hình không có ràng buộc công bằng C5")

        fair, nofair, compare_result = compare_fairness_cost(
            total_budget=total_budget,
            min_region=min_region,
            max_region=max_region,
            min_human=min_human,
            gamma=gamma,
            lam=lam,
        )

        if compare_result is None:
            st.error("Không so sánh được vì ít nhất một mô hình không có nghiệm tối ưu.")
            if not fair["success"]:
                st.write("Mô hình có C5:", fair["message"])
            if not nofair["success"]:
                st.write("Mô hình không C5:", nofair["message"])
        else:
            compare_df = compare_result["compare_df"]

            st.markdown("### Bảng so sánh Z*")
            st.dataframe(compare_df.round(3), use_container_width=True)

            c1, c2, c3 = st.columns(3)

            c1.metric("Z* có C5", f"{fair['objective']:,.2f}")
            c2.metric("Z* không C5", f"{nofair['objective']:,.2f}")
            c3.metric(
                "Chi phí công bằng",
                f"{compare_result['cost']:,.2f}",
                f"{compare_result['cost_pct']:.4f}%"
            )

            fig_compare = px.bar(
                compare_df,
                x="Mô hình",
                y="Z* GDP gain",
                text=compare_df["Z* GDP gain"].round(2),
                title="So sánh GDP gain giữa mô hình có và không có công bằng C5"
            )
            st.plotly_chart(fig_compare, use_container_width=True)

            fair_long = allocation_long_df(fair["allocation"])
            fair_long["Mô hình"] = "Có C5"

            nofair_long = allocation_long_df(nofair["allocation"])
            nofair_long["Mô hình"] = "Không C5"

            both_long = pd.concat([fair_long, nofair_long], ignore_index=True)

            fig_alloc_compare = px.bar(
                both_long,
                x="Vùng",
                y="Ngân sách",
                color="Tên hạng mục",
                facet_col="Mô hình",
                title="So sánh cơ cấu phân bổ có C5 và không C5",
                barmode="stack"
            )
            fig_alloc_compare.update_layout(xaxis_tickangle=-25)
            st.plotly_chart(fig_alloc_compare, use_container_width=True)

            if compare_result["cost"] > 0:
                st.warning(
                    f"Chi phí kinh tế của ràng buộc công bằng vùng miền là "
                    f"{compare_result['cost']:,.2f} đơn vị GDP gain, tương đương "
                    f"{compare_result['cost_pct']:.4f}% so với mô hình không C5. "
                    "Đây là mức đánh đổi giữa hiệu quả kinh tế thuần túy và mục tiêu thu hẹp khoảng cách vùng miền."
                )
            else:
                st.success(
                    "Trong cấu hình hiện tại, ràng buộc công bằng không làm giảm Z*. "
                    "Điều này có thể xảy ra khi nghiệm tối ưu vốn đã thỏa mãn yêu cầu công bằng."
                )

    # =====================================================
    # TAB 6 — THẢO LUẬN CHÍNH SÁCH
    # =====================================================

    with tab6:
        st.header("4.5 — Câu hỏi thảo luận chính sách")

        fair, nofair, compare_result = compare_fairness_cost(
            total_budget=total_budget,
            min_region=min_region,
            max_region=max_region,
            min_human=min_human,
            gamma=gamma,
            lam=lam,
        )

        st.markdown("### a) Nếu bỏ ràng buộc công bằng, vốn sẽ chảy về vùng nào? Vì sao?")

        if nofair["success"]:
            region_total_nofair, item_total_nofair, summary_nofair = summarize_allocation(
                nofair["allocation"],
                nofair["objective"]
            )

            st.info(
                f"Khi bỏ C5, vùng nhận nhiều ngân sách nhất là {summary_nofair['top_region_name']}. "
                "Vốn có xu hướng chảy về nơi có hệ số tác động β cao, khả năng hấp thụ tốt hoặc nơi không bị yêu cầu nâng mức số hóa cho vùng yếu. "
                "Về dài hạn, điều này có thể làm gia tăng khoảng cách số giữa các vùng."
            )

            st.dataframe(region_total_nofair.round(3), use_container_width=True)

        st.markdown("### b) Ràng buộc trần ngân sách mỗi vùng C3 có thể coi như chính sách phân quyền không?")

        st.write(
            "Có. Trần ngân sách mỗi vùng ngăn mô hình tập trung toàn bộ nguồn lực vào một vài vùng có hiệu quả biên cao nhất. "
            "Về bản chất, đây là cơ chế phân quyền và chống tập trung quá mức. Nó có thể làm giảm Z* trong ngắn hạn, "
            "nhưng giúp duy trì cân bằng lãnh thổ, giảm bất bình đẳng vùng và tạo nền tảng tăng trưởng bao trùm hơn."
        )

        st.markdown("### c) Tây Nguyên có hệ số AI thấp, nên đầu tư AI hay H/I trước?")

        st.write(
            "Theo ma trận β, Tây Nguyên có hệ số AI = 0,45 thấp hơn đáng kể so với H = 1,35 và I = 1,20. "
            "Vì vậy, mô hình thường khuyến nghị ưu tiên nhân lực số H và hạ tầng số I trước khi mở rộng đầu tư AI. "
            "Điều này hợp lý về chính sách: vùng có nền tảng số còn yếu cần nâng năng lực hấp thụ trước, "
            "nếu đầu tư AI quá sớm có thể hiệu quả thấp."
        )

        if compare_result is not None:
            st.markdown("### Kết luận ngắn")

            st.success(
                f"Ràng buộc công bằng C5 tạo ra chi phí đánh đổi khoảng "
                f"{compare_result['cost']:,.2f} GDP gain, tương đương "
                f"{compare_result['cost_pct']:.4f}% so với mô hình không C5. "
                "Nếu mức chi phí này nhỏ, chính sách công bằng vùng miền có thể được xem là đáng chấp nhận."
            )


def run():
    render()

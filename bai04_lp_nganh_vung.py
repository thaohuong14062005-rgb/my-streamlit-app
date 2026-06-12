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
BRAND = "#053151"

REGIONS = ["NMM", "RRD", "NCC", "CH", "SE", "MD"]

REGION_NAMES = {
    "NMM": "Trung du miền núi phía Bắc",
REGION_LABELS = {
    "NMM": "Trung du & miền núi Bắc Bộ",
"RRD": "Đồng bằng sông Hồng",
    "NCC": "Bắc Trung Bộ + DH Trung Bộ",
    "NCC": "Bắc Trung Bộ & DHMT",
"CH": "Tây Nguyên",
"SE": "Đông Nam Bộ",
"MD": "Đồng bằng sông Cửu Long",
}

ITEMS = ["I", "D", "AI", "H"]

ITEM_NAMES = {
ITEM_LABELS = {
"I": "Hạ tầng số",
    "D": "Chuyển đổi số DN",
    "AI": "Năng lực AI",
    "D": "Dữ liệu/nền tảng",
    "AI": "AI",
"H": "Nhân lực số",
}

MULTI_COLORS = {
    "Hạ tầng số": "#053151",
    "Dữ liệu/nền tảng": "#E76F51",
    "AI": "#2A9D8F",
    "Nhân lực số": "#F4A261",
    "PuLP": "#053151",
    "CVXPY": "#E76F51",
    "Có công bằng": "#053151",
    "Không công bằng": "#E76F51",
}


BETA = {
("NMM", "I"): 1.15, ("NMM", "D"): 0.85, ("NMM", "AI"): 0.55, ("NMM", "H"): 1.30,
("RRD", "I"): 0.95, ("RRD", "D"): 1.25, ("RRD", "AI"): 1.40, ("RRD", "H"): 1.05,
@@ -66,864 +68,824 @@
}


def beta_matrix_df():
def beta_table():
rows = []

for r in REGIONS:
        row = {
            "Mã vùng": r,
            "Vùng": REGION_NAMES[r],
        }

        row = {"Vùng": r, "Tên vùng": REGION_LABELS[r]}
for j in ITEMS:
            row[f"{j} - {ITEM_NAMES[j]}"] = BETA[(r, j)]

            row[ITEM_LABELS[j]] = BETA[(r, j)]
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
def solve_pulp(
    budget=50000.0,
    floor_region=5000.0,
    cap_region=12000.0,
    h_min=12000.0,
gamma=0.002,
    lam=0.7,
    use_fairness=True,
    lam=0.68,
    fairness=True,
    use_cap=True,
):
if not PULP_AVAILABLE:
return {
"success": False,
            "message": "PuLP chưa được cài đặt. Hãy thêm `pulp` vào requirements.txt.",
            "allocation": None,
            "objective": None,
            "digital_after": None,
            "status": None,
            "status": "PuLP chưa được cài",
            "x": None,
            "objective": np.nan,
            "M": np.nan,
}

    model = pulp.LpProblem("VN_Digital_Budget_Region", pulp.LpMaximize)

    x = pulp.LpVariable.dicts(
        "x",
        (REGIONS, ITEMS),
        lowBound=0,
        cat="Continuous"
    )
    model = pulp.LpProblem("VN_Digital_Budget", pulp.LpMaximize)

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
    x = pulp.LpVariable.dicts("x", (REGIONS, ITEMS), lowBound=0)

    # C4. Sàn nhân lực số
    model += pulp.lpSum(x[r]["H"] for r in REGIONS) >= min_human, "C4_San_nhan_luc_so"
    model += pulp.lpSum(BETA[(r, j)] * x[r][j] for r in REGIONS for j in ITEMS), "Z"

    # C5. Công bằng vùng miền
    # D_r + gamma*x_D,r >= lambda * max_r(D_r + gamma*x_D,r)
    # Linear hóa bằng biến M.
    M = None
    model += pulp.lpSum(x[r][j] for r in REGIONS for j in ITEMS) <= budget, "C1_Ngan_sach_tong"

    if use_fairness:
        M = pulp.LpVariable("Dmax", lowBound=0, cat="Continuous")
    for r in REGIONS:
        model += pulp.lpSum(x[r][j] for j in ITEMS) >= floor_region, f"C2_San_{r}"

    if use_cap:
for r in REGIONS:
            model += D0[r] + gamma * x[r]["D"] <= M, f"C5a_Dmax_upper_{r}"
            model += pulp.lpSum(x[r][j] for j in ITEMS) <= cap_region, f"C3_Tran_{r}"

    model += pulp.lpSum(x[r]["H"] for r in REGIONS) >= h_min, "C4_Nhan_luc_so"

    M = None
    if fairness:
        M = pulp.LpVariable("Dmax", lowBound=0)
for r in REGIONS:
            model += D0[r] + gamma * x[r]["D"] >= lam * M, f"C5b_Fairness_lower_{r}"
            model += D0[r] + gamma * x[r]["D"] <= M, f"C5_Max_{r}"
        for r in REGIONS:
            model += D0[r] + gamma * x[r]["D"] >= lam * M, f"C5_Min_{r}"

solver = pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)

    status = pulp.LpStatus[model.status]
    status_code = model.solve(solver)
    status = pulp.LpStatus[status_code]

if status != "Optimal":
return {
"success": False,
            "message": f"PuLP không tìm được nghiệm tối ưu. Trạng thái: {status}",
            "allocation": None,
            "objective": None,
            "digital_after": None,
"status": status,
            "x": None,
            "objective": np.nan,
            "M": np.nan,
            "model": model,
}

    allocation = pd.DataFrame(index=REGIONS, columns=ITEMS, dtype=float)
    x_matrix = pd.DataFrame(index=REGIONS, columns=ITEMS, dtype=float)

for r in REGIONS:
for j in ITEMS:
            allocation.loc[r, j] = pulp.value(x[r][j])

    allocation.index.name = "Mã vùng"
    allocation_named = allocation.copy()
    allocation_named.index = [REGION_NAMES[r] for r in allocation.index]
    allocation_named.columns = [f"{j} - {ITEM_NAMES[j]}" for j in allocation.columns]
            x_matrix.loc[r, j] = float(pulp.value(x[r][j]))

    objective = pulp.value(model.objective)
    obj = float(pulp.value(model.objective))

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
    if fairness and M is not None:
        m_value = float(pulp.value(M))
    else:
        m_value = np.nan

return {
"success": True,
        "message": "Optimal",
        "allocation": allocation,
        "allocation_named": allocation_named,
        "objective": objective,
        "digital_after": digital_after,
        "duals": dual_df,
"status": status,
        "x": x_matrix,
        "objective": obj,
        "M": m_value,
        "model": model,
}


# =====================================================
# 3. GIẢI BẰNG CVXPY
# =====================================================

def solve_with_cvxpy(
    total_budget=50000.0,
    min_region=5000.0,
    max_region=12000.0,
    min_human=12000.0,
def solve_cvxpy(
    budget=50000.0,
    floor_region=5000.0,
    cap_region=12000.0,
    h_min=12000.0,
gamma=0.002,
    lam=0.7,
    use_fairness=True,
    lam=0.68,
    fairness=True,
    use_cap=True,
):
if not CVXPY_AVAILABLE:
return {
"success": False,
            "message": "CVXPY chưa được cài đặt. Hãy thêm `cvxpy` vào requirements.txt.",
            "allocation": None,
            "objective": None,
            "status": None,
            "status": "CVXPY chưa được cài",
            "x": None,
            "objective": np.nan,
            "M": np.nan,
}

    beta = np.array([[BETA[(r, j)] for j in ITEMS] for r in REGIONS], dtype=float)
    d0 = np.array([D0[r] for r in REGIONS], dtype=float)

    x = cp.Variable((len(REGIONS), len(ITEMS)), nonneg=True)
    n_r = len(REGIONS)
    n_j = len(ITEMS)

    constraints = []
    beta_matrix = np.array([[BETA[(r, j)] for j in ITEMS] for r in REGIONS], dtype=float)

    # C1
    constraints.append(cp.sum(x) <= total_budget)
    X = cp.Variable((n_r, n_j), nonneg=True)

    # C2, C3
    region_sum = cp.sum(x, axis=1)
    constraints.append(region_sum >= min_region)
    constraints.append(region_sum <= max_region)
    constraints = [
        cp.sum(X) <= budget,
        cp.sum(X, axis=1) >= floor_region,
        cp.sum(X[:, ITEMS.index("H")]) >= h_min,
    ]

    # C4
    h_idx = ITEMS.index("H")
    constraints.append(cp.sum(x[:, h_idx]) >= min_human)
    if use_cap:
        constraints.append(cp.sum(X, axis=1) <= cap_region)

    # C5
    if use_fairness:
        d_idx = ITEMS.index("D")
    M = None
    if fairness:
M = cp.Variable(nonneg=True)
        d_scores = np.array([D0[r] for r in REGIONS], dtype=float) + gamma * X[:, ITEMS.index("D")]
        constraints += [
            d_scores <= M,
            d_scores >= lam * M,
        ]

        digital_after = d0 + gamma * x[:, d_idx]
    objective = cp.Maximize(cp.sum(cp.multiply(beta_matrix, X)))
    problem = cp.Problem(objective, constraints)

        constraints.append(digital_after <= M)
        constraints.append(digital_after >= lam * M)
    installed = cp.installed_solvers()
    candidate_solvers = ["CLARABEL", "SCIPY", "ECOS", "SCS"]
    used_solver = None

    objective = cp.Maximize(cp.sum(cp.multiply(beta, x)))
    problem = cp.Problem(objective, constraints)
    for solver_name in candidate_solvers:
        if solver_name in installed:
            try:
                problem.solve(solver=solver_name, verbose=False)
                used_solver = solver_name
                break
            except Exception:
                continue

    try:
        problem.solve(solver=cp.CLARABEL)
    except Exception:
    if used_solver is None:
try:
            problem.solve(solver=cp.SCS)
        except Exception as e:
            problem.solve(verbose=False)
            used_solver = "default"
        except Exception as exc:
return {
"success": False,
                "message": f"CVXPY không giải được bài toán. Lỗi: {e}",
                "allocation": None,
                "objective": None,
                "status": None,
                "status": f"Lỗi solver: {exc}",
                "x": None,
                "objective": np.nan,
                "M": np.nan,
}

if problem.status not in ["optimal", "optimal_inaccurate"]:
return {
"success": False,
            "message": f"CVXPY không tìm được nghiệm tối ưu. Trạng thái: {problem.status}",
            "allocation": None,
            "objective": None,
"status": problem.status,
            "x": None,
            "objective": np.nan,
            "M": np.nan,
            "solver": used_solver,
}

    allocation = pd.DataFrame(x.value, index=REGIONS, columns=ITEMS)
    allocation.index.name = "Mã vùng"

    allocation_named = allocation.copy()
    allocation_named.index = [REGION_NAMES[r] for r in allocation.index]
    allocation_named.columns = [f"{j} - {ITEM_NAMES[j]}" for j in allocation.columns]
    x_matrix = pd.DataFrame(X.value, index=REGIONS, columns=ITEMS)

    digital_after_df = pd.DataFrame({
        "Mã vùng": REGIONS,
        "Vùng": [REGION_NAMES[r] for r in REGIONS],
        "D ban đầu": [D0[r] for r in REGIONS],
        "Đầu tư D": [allocation.loc[r, "D"] for r in REGIONS],
        "D sau đầu tư": [D0[r] + gamma * allocation.loc[r, "D"] for r in REGIONS],
    })
    m_value = float(M.value) if fairness and M is not None and M.value is not None else np.nan

return {
"success": True,
        "message": "Optimal",
        "allocation": allocation,
        "allocation_named": allocation_named,
        "objective": float(problem.value),
        "digital_after": digital_after_df,
"status": problem.status,
        "x": x_matrix,
        "objective": float(problem.value),
        "M": m_value,
        "solver": used_solver,
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

def matrix_display(x_matrix):
    out = x_matrix.copy()
    out.index = [REGION_LABELS[r] for r in out.index]
    out = out.rename(columns=ITEM_LABELS)
    out.insert(0, "Vùng", out.index)
    out = out.reset_index(drop=True)
    return out

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
def long_allocation(x_matrix):
    rows = []

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
    for r in REGIONS:
        for j in ITEMS:
            rows.append(
                {
                    "Vùng": REGION_LABELS[r],
                    "Mã vùng": r,
                    "Hạng mục": ITEM_LABELS[j],
                    "Mã hạng mục": j,
                    "Ngân sách": float(x_matrix.loc[r, j]),
                    "Beta": BETA[(r, j)],
                    "Đóng góp Z": float(x_matrix.loc[r, j]) * BETA[(r, j)],
                }
            )

    return fair, nofair, result
    return pd.DataFrame(rows)


# =====================================================
# 5. GIAO DIỆN STREAMLIT
# =====================================================
def region_summary(x_matrix):
    long_df = long_allocation(x_matrix)

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
    region_df = (
        long_df.groupby(["Mã vùng", "Vùng"], as_index=False)
        .agg(
            **{
                "Tổng ngân sách": ("Ngân sách", "sum"),
                "Tổng đóng góp Z": ("Đóng góp Z", "sum"),
            }
        )
        .sort_values("Tổng ngân sách", ascending=False)
        .reset_index(drop=True)
)

    st.info(
        "Đơn vị ngân sách trong module là tỷ VND. "
        "Hàm mục tiêu Z đo GDP gain kỳ vọng theo hệ số tác động biên β của từng vùng và hạng mục."
    top_item = (
        long_df.sort_values(["Mã vùng", "Ngân sách"], ascending=[True, False])
        .groupby("Mã vùng")
        .head(1)[["Mã vùng", "Hạng mục", "Ngân sách"]]
        .rename(columns={"Hạng mục": "Hạng mục ưu tiên", "Ngân sách": "Ngân sách hạng mục ưu tiên"})
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
    region_df = region_df.merge(top_item, on="Mã vùng", how="left")

        max_region = st.number_input(
            "Trần ngân sách mỗi vùng",
            min_value=5000.0,
            max_value=30000.0,
            value=12000.0,
            step=500.0
        )
    return region_df

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
def digital_fairness_table(x_matrix, gamma=0.002):
    rows = []

        lam = st.number_input(
            "λ - hệ số công bằng vùng",
            min_value=0.10,
            max_value=1.00,
            value=0.70,
            step=0.05,
            format="%.2f"
    for r in REGIONS:
        d_invest = float(x_matrix.loc[r, "D"])
        d_after = D0[r] + gamma * d_invest

        rows.append(
            {
                "Vùng": REGION_LABELS[r],
                "D0": D0[r],
                "Đầu tư D": d_invest,
                "D sau đầu tư": d_after,
            }
)

    if min_region * len(REGIONS) > total_budget:
        st.error(
            f"Bài toán không khả thi: sàn mỗi vùng {min_region:,.0f} × 6 "
            f"= {min_region * len(REGIONS):,.0f} lớn hơn ngân sách tổng {total_budget:,.0f}."
        )
        st.stop()
    return pd.DataFrame(rows).sort_values("D sau đầu tư", ascending=False).reset_index(drop=True)

    if max_region * len(REGIONS) < total_budget:
        st.warning(
            "Tổng trần vùng nhỏ hơn ngân sách tổng, nên mô hình có thể không dùng hết ngân sách."
        )

    if min_human > total_budget:
        st.error("Bài toán không khả thi: sàn nhân lực số H lớn hơn ngân sách tổng.")
        st.stop()
def make_styled_table(df, decimals=3):
    show_df = df.copy()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📌 Mô hình",
        "4.4.1 PuLP",
        "4.4.2 CVXPY",
        "4.4.3 Heatmap",
        "4.4.4 Chi phí công bằng",
        "🧠 Thảo luận chính sách"
    ])
    format_dict = {}
    for col in show_df.columns:
        if pd.api.types.is_numeric_dtype(show_df[col]):
            format_dict[col] = "{:." + str(decimals) + "f}"

    # =====================================================
    # TAB 1 — MÔ HÌNH
    # =====================================================
    styler = show_df.style.format(format_dict)

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

        st.markdown("### Hàm mục tiêu")
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

        st.latex(r"\max Z = \sum_r \sum_j \beta_{j,r} x_{j,r}")
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
def render():
    st.title("Bài 4. Phân bổ ngân sách số theo vùng")
    st.caption("PuLP, CVXPY, heatmap phân bổ, ràng buộc công bằng vùng miền và chi phí công bằng")

        st.dataframe(constraints_df, use_container_width=True)
    if not PULP_AVAILABLE:
        st.error("Chưa cài PuLP. Hãy thêm `pulp` vào requirements.txt.")
        return

    budget = 50000.0
    floor_region = 5000.0
    cap_region = 12000.0
    h_min = 12000.0
    gamma = 0.002
    lam = 0.68

    pulp_res = solve_pulp(
        budget=budget,
        floor_region=floor_region,
        cap_region=cap_region,
        h_min=h_min,
        gamma=gamma,
        lam=lam,
        fairness=True,
        use_cap=True,
    )

        st.markdown("### Ma trận hệ số tác động biên β")
    pulp_no_fair = solve_pulp(
        budget=budget,
        floor_region=floor_region,
        cap_region=cap_region,
        h_min=h_min,
        gamma=gamma,
        lam=lam,
        fairness=False,
        use_cap=True,
    )

        st.dataframe(beta_matrix_df(), use_container_width=True)
    pulp_no_cap = solve_pulp(
        budget=budget,
        floor_region=floor_region,
        cap_region=cap_region,
        h_min=h_min,
        gamma=gamma,
        lam=lam,
        fairness=True,
        use_cap=False,
    )

        st.markdown("### Digital Index ban đầu Dᵣ")
    cvx_res = solve_cvxpy(
        budget=budget,
        floor_region=floor_region,
        cap_region=cap_region,
        h_min=h_min,
        gamma=gamma,
        lam=lam,
        fairness=True,
        use_cap=True,
    )

        st.dataframe(d0_df(), use_container_width=True)
    tabs = st.tabs(
        [
            "4.4.1 PuLP",
            "4.4.2 CVXPY",
            "4.4.3 Heatmap",
            "4.4.4 Chi phí công bằng",
            "4.5 Chính sách",
        ]
    )

# =====================================================
    # TAB 2 — PULP
    # 4.4.1
# =====================================================
    with tabs[0]:
        st.header("4.4.1. Mô hình PuLP và nghiệm tối ưu")

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
        show_table(beta_table(), decimals=3)

        if not result_pulp["success"]:
            st.error(result_pulp["message"])
        if not pulp_res["success"]:
            st.error(f"Mô hình PuLP không khả thi hoặc không tối ưu. Trạng thái: {pulp_res['status']}")
            st.info(
                "Với λ = 0,70 theo đề và trần 12.000 tỷ/vùng, mô hình có thể không khả thi. "
                "Bản app đang dùng λ = 0,68 để minh họa nghiệm khả thi."
            )
else:
            allocation = result_pulp["allocation"]
            allocation_named = result_pulp["allocation_named"]
            objective = result_pulp["objective"]
            x_matrix = pulp_res["x"]
            region_df = region_summary(x_matrix)

            region_total, item_total, summary = summarize_allocation(allocation, objective)
            top_region = region_df.iloc[0]["Vùng"]
            top_budget = region_df.iloc[0]["Tổng ngân sách"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Trạng thái", result_pulp["status"])
            c2.metric("Z* GDP gain", f"{objective:,.2f}")
            c3.metric("Tổng ngân sách dùng", f"{summary['total_budget_used']:,.0f}")
            c4.metric("Ngân sách H", f"{summary['human_budget']:,.0f}")

            st.markdown("### Ma trận phân bổ tối ưu 6×4")
            st.dataframe(allocation_named.round(3), use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Trạng thái", "Optimal")
            c2.metric("Z*", f"{pulp_res['objective']:,.1f}")
            c3.metric("Vùng nhận nhiều nhất", top_region)

            st.markdown("### Tổng ngân sách theo vùng")
            st.dataframe(region_total.round(3), use_container_width=True)
            st.subheader("Ma trận phân bổ tối ưu 6×4")
            show_table(matrix_display(x_matrix), decimals=2)

            st.markdown("### Tổng ngân sách theo hạng mục")
            st.dataframe(item_total.round(3), use_container_width=True)
            st.subheader("Tổng hợp theo vùng")
            show_table(region_df, decimals=2)

            st.markdown("### Digital Index sau đầu tư và kiểm tra C5")
            st.dataframe(result_pulp["digital_after"].round(4), use_container_width=True)
            long_df = long_allocation(x_matrix)

            st.markdown("### Shadow price / Dual values từ PuLP")
            st.dataframe(result_pulp["duals"].round(6), use_container_width=True)
            fig = px.bar(
                long_df,
                x="Vùng",
                y="Ngân sách",
                color="Hạng mục",
                title="Phân bổ ngân sách theo vùng và hạng mục",
                color_discrete_map=MULTI_COLORS,
            )
            fig.update_traces(marker_line_color="white", marker_line_width=1)
            fig.update_layout(
                barmode="stack",
                xaxis_title="Vùng",
                yaxis_title="Tỷ VND",
            )
            style_base_fig(fig, height=470)
            st.plotly_chart(fig, use_container_width=True)

st.success(
                f"Vùng nhận ngân sách lớn nhất là {summary['top_region_name']}. "
                f"Hạng mục được ưu tiên nhiều nhất toàn quốc là {summary['top_item_name']}."
                f"PuLP tìm được nghiệm tối ưu với Z* = {pulp_res['objective']:,.1f}. "
                f"Vùng nhận ngân sách lớn nhất là {top_region}, khoảng {top_budget:,.1f} tỷ VND."
)

# =====================================================
    # TAB 3 — CVXPY
    # 4.4.2
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
    with tabs[1]:
        st.header("4.4.2. So sánh PuLP và CVXPY")

        if not CVXPY_AVAILABLE:
            st.warning("Chưa cài CVXPY. Hãy thêm `cvxpy` vào requirements.txt để chạy phần này.")
        elif not cvx_res["success"]:
            st.warning(f"CVXPY chưa tìm được nghiệm tối ưu. Trạng thái: {cvx_res['status']}")
        elif not pulp_res["success"]:
            st.warning("PuLP chưa có nghiệm tối ưu nên chưa thể so sánh.")
else:
            c1, c2, c3 = st.columns(3)

            c1.metric("Z* PuLP", f"{result_pulp['objective']:,.4f}")
            c2.metric("Z* CVXPY", f"{result_cvx['objective']:,.4f}")
            c3.metric(
                "Chênh lệch tuyệt đối",
                f"{abs(result_pulp['objective'] - result_cvx['objective']):,.6f}"
            z_pulp = pulp_res["objective"]
            z_cvx = cvx_res["objective"]
            max_diff = np.max(np.abs(pulp_res["x"].values - cvx_res["x"].values))

            compare_summary = pd.DataFrame(
                {
                    "Phương pháp": ["PuLP/CBC", f"CVXPY/{cvx_res.get('solver', 'solver')}"],
                    "Trạng thái": [pulp_res["status"], cvx_res["status"]],
                    "Z*": [z_pulp, z_cvx],
                    "Sai khác so với PuLP": [0.0, abs(z_cvx - z_pulp)],
                }
)

            st.markdown("### Phân bổ tối ưu bằng CVXPY")
            st.dataframe(result_cvx["allocation_named"].round(3), use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Z* PuLP", f"{z_pulp:,.1f}")
            c2.metric("Z* CVXPY", f"{z_cvx:,.1f}")
            c3.metric("Sai khác max x", f"{max_diff:,.4f}")

            diff = result_cvx["allocation"] - result_pulp["allocation"]
            show_table(compare_summary, decimals=4)

            diff_named = diff.copy()
            diff_named.index = [REGION_NAMES[r] for r in diff_named.index]
            diff_named.columns = [f"{j} - {ITEM_NAMES[j]}" for j in diff_named.columns]
            st.subheader("Ma trận nghiệm CVXPY")
            show_table(matrix_display(cvx_res["x"]), decimals=2)

            st.markdown("### Chênh lệch phân bổ CVXPY - PuLP")
            st.dataframe(diff_named.round(6), use_container_width=True)
            plot_compare = pd.DataFrame(
                {
                    "Phương pháp": ["PuLP", "CVXPY"],
                    "Z*": [z_pulp, z_cvx],
                }
            )

            if abs(result_pulp["objective"] - result_cvx["objective"]) < 1e-2:
                st.success(
                    "PuLP và CVXPY cho giá trị mục tiêu gần như giống nhau. "
                    "Khác biệt nhỏ nếu có thường do sai số số học hoặc nghiệm tối ưu không duy nhất."
                )
            fig = px.bar(
                plot_compare,
                x="Phương pháp",
                y="Z*",
                color="Phương pháp",
                text=plot_compare["Z*"].round(1),
                title="So sánh giá trị tối ưu Z*",
                color_discrete_map=MULTI_COLORS,
            )
            fig.update_traces(textposition="outside", marker_line_color="white", marker_line_width=1)
            fig.update_layout(xaxis_title="Phương pháp", yaxis_title="Z*")
            style_base_fig(fig, height=420)
            st.plotly_chart(fig, use_container_width=True)

            if abs(z_cvx - z_pulp) < 1e-3:
                st.success("Hai phương pháp cho giá trị tối ưu gần như giống nhau.")
else:
                st.warning(
                    "PuLP và CVXPY cho kết quả khác đáng kể. Cần kiểm tra solver, ràng buộc và trạng thái nghiệm."
                st.info(
                    "Hai phương pháp có thể cho phân bổ hơi khác nhau do sai số solver hoặc nhiều nghiệm tối ưu, "
                    "nhưng giá trị mục tiêu cần được so sánh theo Z*."
)

# =====================================================
    # TAB 4 — HEATMAP
    # 4.4.3
# =====================================================
    with tabs[2]:
        st.header("4.4.3. Heatmap phân bổ tối ưu")

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
        if not pulp_res["success"]:
            st.warning("Chưa có nghiệm tối ưu để vẽ heatmap.")
else:
            allocation = result_pulp["allocation"]
            allocation_named = result_pulp["allocation_named"]
            x_matrix = pulp_res["x"]
            matrix_named = x_matrix.copy()
            matrix_named.index = [REGION_LABELS[r] for r in matrix_named.index]
            matrix_named = matrix_named.rename(columns=ITEM_LABELS)

            region_df = region_summary(x_matrix)
            top_region = region_df.iloc[0]["Vùng"]
            top_item_text = region_df.iloc[0]["Hạng mục ưu tiên"]

            c1, c2, c3 = st.columns(3)
            c1.metric("Vùng nhận nhiều nhất", top_region)
            c2.metric("Hạng mục nổi bật", top_item_text)
            c3.metric("Tổng ngân sách", f"{x_matrix.values.sum():,.0f}")

fig_heat = px.imshow(
                allocation_named,
                text_auto=".0f",
                matrix_named,
aspect="auto",
                title="Heatmap phân bổ ngân sách tối ưu 6 vùng × 4 hạng mục",
                labels=dict(x="Hạng mục", y="Vùng", color="Tỷ VND")
                color_continuous_scale="YlGnBu",
                title="Heatmap phân bổ ngân sách tối ưu 6×4",
                labels=dict(x="Hạng mục", y="Vùng", color="Tỷ VND"),
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
            fig_heat.update_traces(
                text=matrix_named.values,
                texttemplate="%{text:.0f}",
                hovertemplate="<b>%{y}</b><br>%{x}: %{z:,.1f} tỷ VND<extra></extra>",
)
            fig_stack.update_layout(xaxis_tickangle=-25)
            st.plotly_chart(fig_stack, use_container_width=True)

            region_total, item_total, summary = summarize_allocation(
                allocation,
                result_pulp["objective"]
            fig_heat.update_layout(
                height=560,
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(color=BRAND),
                title_font=dict(color=BRAND, size=20),
)
            st.plotly_chart(fig_heat, use_container_width=True)

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
            st.subheader("Hạng mục ưu tiên ở từng vùng")
            show_table(
                region_df[["Vùng", "Tổng ngân sách", "Hạng mục ưu tiên", "Ngân sách hạng mục ưu tiên"]],
                decimals=2,
            )

            best_item_df = pd.DataFrame(best_item_rows)
            st.subheader("Chỉ số số hóa sau đầu tư D")
            fair_df = digital_fairness_table(x_matrix, gamma=gamma)
            show_table(fair_df, decimals=3)

            st.dataframe(best_item_df.round(3), use_container_width=True)
            st.success(
                f"Vùng nhận nhiều ngân sách nhất là {top_region}. "
                f"Hạng mục chiếm tỷ trọng lớn nhất tại vùng này là {top_item_text}."
            )

# =====================================================
    # TAB 5 — CHI PHÍ CÔNG BẰNG
    # 4.4.4
# =====================================================
    with tabs[3]:
        st.header("4.4.4. Chi phí kinh tế của công bằng vùng miền")

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
        if not pulp_res["success"] or not pulp_no_fair["success"]:
            st.warning("Không đủ nghiệm để so sánh có/không có ràng buộc công bằng.")
else:
            compare_df = compare_result["compare_df"]

            st.markdown("### Bảng so sánh Z*")
            st.dataframe(compare_df.round(3), use_container_width=True)
            z_fair = pulp_res["objective"]
            z_no_fair = pulp_no_fair["objective"]
            cost_fair = z_no_fair - z_fair
            cost_pct = cost_fair / z_no_fair * 100 if abs(z_no_fair) > 1e-12 else np.nan

            if pulp_no_cap["success"]:
                z_no_cap = pulp_no_cap["objective"]
                cost_cap = z_no_cap - z_fair
                cost_cap_pct = cost_cap / z_no_cap * 100 if abs(z_no_cap) > 1e-12 else np.nan
            else:
                z_no_cap = np.nan
                cost_cap = np.nan
                cost_cap_pct = np.nan

c1, c2, c3 = st.columns(3)

            c1.metric("Z* có C5", f"{fair['objective']:,.2f}")
            c2.metric("Z* không C5", f"{nofair['objective']:,.2f}")
            c3.metric(
                "Chi phí công bằng",
                f"{compare_result['cost']:,.2f}",
                f"{compare_result['cost_pct']:.4f}%"
            c1.metric("Z* có C5", f"{z_fair:,.1f}")
            c2.metric("Z* bỏ C5", f"{z_no_fair:,.1f}")
            c3.metric("Chi phí công bằng", f"{cost_fair:,.1f}")

            compare_df = pd.DataFrame(
                {
                    "Mô hình": ["Có công bằng", "Không công bằng", "Bỏ trần C3"],
                    "Z*": [z_fair, z_no_fair, z_no_cap],
                    "Chênh lệch so với mô hình gốc": [0.0, z_no_fair - z_fair, z_no_cap - z_fair],
                    "Tỷ lệ chênh lệch (%)": [0.0, cost_pct, cost_cap_pct],
                }
)

            fig_compare = px.bar(
            show_table(compare_df, decimals=3)

            fig = px.bar(
compare_df,
x="Mô hình",
                y="Z* GDP gain",
                text=compare_df["Z* GDP gain"].round(2),
                title="So sánh GDP gain giữa mô hình có và không có công bằng C5"
                y="Z*",
                text=compare_df["Z*"].round(1),
                title="So sánh Z* giữa các mô hình",
            )
            fig.update_traces(
                marker_color=BRAND,
                textposition="outside",
                textfont=dict(color=BRAND),
)
            st.plotly_chart(fig_compare, use_container_width=True)
            fig.update_layout(xaxis_title="Mô hình", yaxis_title="Z*")
            style_base_fig(fig, height=420)
            st.plotly_chart(fig, use_container_width=True)

            fair_long = allocation_long_df(fair["allocation"])
            fair_long["Mô hình"] = "Có C5"
            st.subheader("So sánh phân bổ có và không có công bằng")

            nofair_long = allocation_long_df(nofair["allocation"])
            nofair_long["Mô hình"] = "Không C5"
            long_fair = long_allocation(pulp_res["x"])
            long_fair["Mô hình"] = "Có công bằng"

            both_long = pd.concat([fair_long, nofair_long], ignore_index=True)
            long_no_fair = long_allocation(pulp_no_fair["x"])
            long_no_fair["Mô hình"] = "Không công bằng"

            fig_alloc_compare = px.bar(
                both_long,
            compare_alloc = pd.concat([long_fair, long_no_fair], ignore_index=True)

            region_compare = (
                compare_alloc.groupby(["Mô hình", "Vùng"], as_index=False)["Ngân sách"]
                .sum()
            )

            fig_region = px.bar(
                region_compare,
x="Vùng",
y="Ngân sách",
                color="Tên hạng mục",
                facet_col="Mô hình",
                title="So sánh cơ cấu phân bổ có C5 và không C5",
                barmode="stack"
                color="Mô hình",
                barmode="group",
                title="Tổng ngân sách theo vùng: có và không có công bằng",
                color_discrete_map=MULTI_COLORS,
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
            fig_region.update_layout(xaxis_title="Vùng", yaxis_title="Tỷ VND")
            style_base_fig(fig_region, height=470)
            st.plotly_chart(fig_region, use_container_width=True)

            st.markdown("### Kiểm tra nhanh mức công bằng λ")

            test_lam = st.slider(
                "λ - mức công bằng vùng miền",
                0.50,
                0.72,
                0.68,
                0.01,
            )

            test_res = solve_pulp(
                budget=budget,
                floor_region=floor_region,
                cap_region=cap_region,
                h_min=h_min,
                gamma=gamma,
                lam=test_lam,
                fairness=True,
                use_cap=True,
            )

            if test_res["success"]:
                st.success(f"Với λ = {test_lam:.2f}, mô hình khả thi. Z* = {test_res['objective']:,.1f}.")
else:
                st.success(
                    "Trong cấu hình hiện tại, ràng buộc công bằng không làm giảm Z*. "
                    "Điều này có thể xảy ra khi nghiệm tối ưu vốn đã thỏa mãn yêu cầu công bằng."
                )
                st.error(f"Với λ = {test_lam:.2f}, mô hình không khả thi. Trạng thái: {test_res['status']}.")

            st.success(
                f"Chi phí kinh tế của công bằng vùng miền là khoảng {cost_fair:,.1f} tỷ VND GDP gain, "
                f"tương đương {cost_pct:.2f}% so với mô hình không có C5."
            )

# =====================================================
    # TAB 6 — THẢO LUẬN CHÍNH SÁCH
    # 4.5
# =====================================================
    with tabs[4]:
        st.header("4.5. Thảo luận chính sách")

        if pulp_res["success"] and pulp_no_fair["success"]:
            z_fair = pulp_res["objective"]
            z_no_fair = pulp_no_fair["objective"]
            cost_fair = z_no_fair - z_fair
            cost_pct = cost_fair / z_no_fair * 100 if abs(z_no_fair) > 1e-12 else np.nan
        else:
            z_fair = np.nan
            z_no_fair = np.nan
            cost_fair = np.nan
            cost_pct = np.nan

        if pulp_res["success"] and pulp_no_cap["success"]:
            z_no_cap = pulp_no_cap["objective"]
            cost_cap = z_no_cap - pulp_res["objective"]
            cost_cap_pct = cost_cap / z_no_cap * 100 if abs(z_no_cap) > 1e-12 else np.nan
        else:
            z_no_cap = np.nan
            cost_cap = np.nan
            cost_cap_pct = np.nan

        st.markdown("### a) Nếu bỏ ràng buộc công bằng, vốn sẽ chảy về vùng nào?")

        st.markdown(
            """
            Nếu bỏ ràng buộc công bằng vùng miền, vốn có xu hướng chảy về các vùng có hệ số tác động biên cao,
            đặc biệt là những vùng có lợi thế rõ trong hạng mục AI, dữ liệu/nền tảng hoặc nhân lực số.
            Trong bộ hệ số của bài, Đông Nam Bộ và Đồng bằng sông Hồng có nhiều hệ số cao ở D và AI,
            nên mô hình không công bằng thường ưu tiên các vùng này hơn.

            Lý do kinh tế là mô hình tối đa hóa GDP gain, nên mỗi đồng ngân sách sẽ được đẩy vào nơi tạo ra
            mức tăng Z lớn nhất. Đây là logic hiệu quả thuần túy.
            """
        )

    with tab6:
        st.header("4.5 — Câu hỏi thảo luận chính sách")

        fair, nofair, compare_result = compare_fairness_cost(
            total_budget=total_budget,
            min_region=min_region,
            max_region=max_region,
            min_human=min_human,
            gamma=gamma,
            lam=lam,
        st.success(
            f"Chi phí công bằng ước tính: {cost_fair:,.1f} tỷ VND GDP gain, khoảng {cost_pct:.2f}% so với mô hình bỏ C5."
)

        st.markdown("### a) Nếu bỏ ràng buộc công bằng, vốn sẽ chảy về vùng nào? Vì sao?")
        st.markdown(
            """
            Hậu quả xã hội dài hạn nếu bỏ công bằng là khoảng cách số giữa các vùng có thể nới rộng.
            Vùng đã phát triển sẽ tiếp tục nhận nhiều vốn hơn, có hạ tầng tốt hơn, doanh nghiệp số mạnh hơn
            và nhân lực chất lượng cao hơn. Ngược lại, vùng có nền số hóa thấp có thể bị mắc kẹt trong vòng lặp
            thiếu hạ tầng, thiếu nhân lực, thiếu doanh nghiệp công nghệ và năng suất thấp.

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
            Vì vậy, ràng buộc công bằng làm giảm một phần hiệu quả kinh tế ngắn hạn,
            nhưng giúp giảm rủi ro phân hóa vùng miền trong dài hạn.
            """
        )

            st.dataframe(region_total_nofair.round(3), use_container_width=True)
        st.markdown("### b) Trần ngân sách mỗi vùng C3 có thể coi là chính sách phân quyền không?")

        st.markdown("### b) Ràng buộc trần ngân sách mỗi vùng C3 có thể coi như chính sách phân quyền không?")
        st.success(
            f"Khi bỏ trần C3, Z* có thể tăng thêm khoảng {cost_cap:,.1f}, tương đương {cost_cap_pct:.2f}%."
        )

        st.write(
            "Có. Trần ngân sách mỗi vùng ngăn mô hình tập trung toàn bộ nguồn lực vào một vài vùng có hiệu quả biên cao nhất. "
            "Về bản chất, đây là cơ chế phân quyền và chống tập trung quá mức. Nó có thể làm giảm Z* trong ngắn hạn, "
            "nhưng giúp duy trì cân bằng lãnh thổ, giảm bất bình đẳng vùng và tạo nền tảng tăng trưởng bao trùm hơn."
        st.markdown(
            """
            Ràng buộc trần ngân sách mỗi vùng có thể được hiểu như một cơ chế phân quyền và chống tập trung quá mức.
            Nếu không có trần, mô hình sẽ dồn vốn vào một số vùng có hiệu suất cao nhất,
            trong khi các vùng khác chỉ nhận mức sàn tối thiểu. Điều đó có thể tối đa hóa GDP gain,
            nhưng làm suy yếu mục tiêu cân bằng không gian phát triển.

            Mức giảm Z* do C3 tạo ra chính là chi phí kinh tế của việc tránh tập trung nguồn lực.
            Mức giảm này có thể chấp nhận được nếu mục tiêu chính sách không chỉ là tăng trưởng ngắn hạn,
            mà còn bao gồm ổn định xã hội, nâng cấp năng lực vùng yếu và bảo đảm quyền tiếp cận hạ tầng số cơ bản.
            """
)

        st.markdown("### c) Tây Nguyên có hệ số AI thấp, nên đầu tư AI hay H/I trước?")
        st.markdown("### c) Tây Nguyên nên đầu tư AI hay tập trung H và I trước?")

        st.write(
            "Theo ma trận β, Tây Nguyên có hệ số AI = 0,45 thấp hơn đáng kể so với H = 1,35 và I = 1,20. "
            "Vì vậy, mô hình thường khuyến nghị ưu tiên nhân lực số H và hạ tầng số I trước khi mở rộng đầu tư AI. "
            "Điều này hợp lý về chính sách: vùng có nền tảng số còn yếu cần nâng năng lực hấp thụ trước, "
            "nếu đầu tư AI quá sớm có thể hiệu quả thấp."
        )
        st.markdown(
            """
            Trong dữ liệu của bài, Tây Nguyên có hệ số AI thấp nhất, chỉ khoảng 0,45.
            Trong khi đó, hệ số của H và I tại Tây Nguyên lại cao hơn đáng kể.
            Điều này cho thấy nếu xét theo hiệu quả biên trực tiếp, đầu tư vào AI tại Tây Nguyên chưa phải lựa chọn tốt nhất.

        if compare_result is not None:
            st.markdown("### Kết luận ngắn")
            Mô hình hàm ý rằng Tây Nguyên nên ưu tiên **nhân lực số H** và **hạ tầng số I** trước.
            Đây là hai nền tảng giúp vùng cải thiện năng lực hấp thụ công nghệ.
            Khi nhân lực, kết nối, hạ tầng dữ liệu và nền tảng vận hành được cải thiện,
            đầu tư AI sau đó mới có khả năng tạo hiệu quả thực chất hơn.

            st.success(
                f"Ràng buộc công bằng C5 tạo ra chi phí đánh đổi khoảng "
                f"{compare_result['cost']:,.2f} GDP gain, tương đương "
                f"{compare_result['cost_pct']:.4f}% so với mô hình không C5. "
                "Nếu mức chi phí này nhỏ, chính sách công bằng vùng miền có thể được xem là đáng chấp nhận."
            )
            Nói cách khác, không nên áp dụng cùng một công thức AI cho mọi vùng.
            Vùng có nền tảng số thấp cần đi theo lộ trình: **hạ tầng số → nhân lực số → dữ liệu/nền tảng → AI ứng dụng**.
            """
        )

        st.info(
            "Kết luận: công bằng vùng miền có chi phí kinh tế ngắn hạn, nhưng là điều kiện quan trọng để tránh phân hóa số dài hạn."
        )


def run():

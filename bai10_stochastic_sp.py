# bai10_stochastic_sp.py

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

try:
    import pyomo.environ as pyo
    PYOMO_AVAILABLE = True
except Exception:
    PYOMO_AVAILABLE = False

try:
    from scipy.optimize import linprog
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


BRAND = "#053151"

MULTI_COLORS = {
    "I": "#053151",
    "D": "#E76F51",
    "AI": "#2A9D8F",
    "H": "#F4A261",
    "SP": "#053151",
    "EV": "#E76F51",
    "Robust": "#2A9D8F",
    "Wait-and-see": "#F4A261",
}


# =========================================================
# BÀI 10 — TWO-STAGE STOCHASTIC PROGRAMMING
# =========================================================


J = ["I", "D", "AI", "H"]
J_LABELS = {
    "I": "Hạ tầng số",
    "D": "Dữ liệu/nền tảng",
    "AI": "AI",
    "H": "Nhân lực số",
}

S = ["s1", "s2", "s3", "s4"]
S_LABELS = {
    "s1": "Tăng trưởng số cao",
    "s2": "Cơ sở",
    "s3": "Suy giảm nhẹ",
    "s4": "Cú sốc xấu",
}

P = {
    "s1": 0.30,
    "s2": 0.45,
    "s3": 0.20,
    "s4": 0.05,
}

BETA = {
    "I": 1.00,
    "D": 1.10,
    "AI": 1.25,
    "H": 0.95,
}

BETA_S = {
    ("s1", "I"): 1.25,
    ("s1", "D"): 1.35,
    ("s1", "AI"): 1.55,
    ("s1", "H"): 1.05,
    ("s2", "I"): 1.00,
    ("s2", "D"): 1.10,
    ("s2", "AI"): 1.25,
    ("s2", "H"): 0.95,
    ("s3", "I"): 0.75,
    ("s3", "D"): 0.85,
    ("s3", "AI"): 0.90,
    ("s3", "H"): 1.00,
    ("s4", "I"): 0.40,
    ("s4", "D"): 0.50,
    ("s4", "AI"): 0.55,
    ("s4", "H"): 1.10,
}

BUDGET_1 = 65000.0
BUDGET_2 = 15000.0


def scenario_table():
    rows = []

    for s in S:
        row = {
            "Kịch bản": s,
            "Tên kịch bản": S_LABELS[s],
            "Xác suất": P[s],
        }

        for j in J:
            row[J_LABELS[j]] = BETA_S[(s, j)]

        rows.append(row)

    return pd.DataFrame(rows)


def beta_table():
    return pd.DataFrame(
        {
            "Hạng mục": [J_LABELS[j] for j in J],
            "Mã": J,
            "Beta first-stage": [BETA[j] for j in J],
        }
    )


def expected_coefficients():
    coeff = {}

    for j in J:
        coeff[j] = sum(P[s] * BETA_S[(s, j)] for s in S)

    return coeff


def pyomo_solver_name():
    if not PYOMO_AVAILABLE:
        return None

    candidates = ["appsi_highs", "highs", "cbc", "glpk"]

    for name in candidates:
        try:
            solver = pyo.SolverFactory(name)
            if solver is not None and solver.available(exception_flag=False):
                return name
        except Exception:
            pass

    return None


def solve_sp_pyomo():
    if not PYOMO_AVAILABLE:
        return {
            "success": False,
            "status": "Pyomo chưa được cài",
            "solver": "None",
        }

    solver_name = pyomo_solver_name()

    if solver_name is None:
        return {
            "success": False,
            "status": "Không tìm thấy solver Pyomo khả dụng",
            "solver": "None",
        }

    try:
        m = pyo.ConcreteModel()

        m.J = pyo.Set(initialize=J)
        m.S = pyo.Set(initialize=S)

        m.p = pyo.Param(m.S, initialize=P)
        m.beta = pyo.Param(m.J, initialize=BETA)
        m.beta_s = pyo.Param(m.S, m.J, initialize=BETA_S)

        # First-stage decision
        m.x = pyo.Var(m.J, within=pyo.NonNegativeReals)

        # Second-stage recourse decision, phụ thuộc kịch bản
        m.y = pyo.Var(m.S, m.J, within=pyo.NonNegativeReals)

        m.budget1 = pyo.Constraint(
            expr=sum(m.x[j] for j in m.J) <= BUDGET_1
        )

        def budget2_rule(m, s):
            return sum(m.y[s, j] for j in m.J) <= BUDGET_2

        m.budget2 = pyo.Constraint(m.S, rule=budget2_rule)

        def obj_rule(m):
            first = sum(m.beta[j] * m.x[j] for j in m.J)
            second = sum(
                m.p[s] * sum(m.beta_s[s, j] * m.y[s, j] for j in m.J)
                for s in m.S
            )
            return first + second

        m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

        solver = pyo.SolverFactory(solver_name)
        results = solver.solve(m, tee=False)

        termination = str(results.solver.termination_condition).lower()

        if "optimal" not in termination:
            return {
                "success": False,
                "status": str(results.solver.termination_condition),
                "solver": solver_name,
            }

        x = {j: float(pyo.value(m.x[j])) for j in J}
        y = {
            s: {j: float(pyo.value(m.y[s, j])) for j in J}
            for s in S
        }

        return {
            "success": True,
            "status": "Optimal",
            "solver": solver_name,
            "objective": float(pyo.value(m.obj)),
            "x": x,
            "y": y,
            "source": f"Pyomo/{solver_name}",
        }

    except Exception as exc:
        return {
            "success": False,
            "status": str(exc),
            "solver": solver_name,
        }


def solve_sp_scipy():
    if not SCIPY_AVAILABLE:
        return {
            "success": False,
            "status": "SciPy chưa được cài",
            "solver": "None",
        }

    # Variable order:
    # x_I, x_D, x_AI, x_H,
    # y_s1_I,...,y_s1_H, y_s2_I,...,y_s4_H
    n_x = len(J)
    n_y = len(S) * len(J)
    n = n_x + n_y

    c = np.zeros(n)

    for j_idx, j in enumerate(J):
        c[j_idx] = -BETA[j]

    offset = n_x

    for s_idx, s in enumerate(S):
        for j_idx, j in enumerate(J):
            idx = offset + s_idx * len(J) + j_idx
            c[idx] = -P[s] * BETA_S[(s, j)]

    A_ub = []
    b_ub = []

    row = np.zeros(n)
    row[:n_x] = 1
    A_ub.append(row)
    b_ub.append(BUDGET_1)

    for s_idx, s in enumerate(S):
        row = np.zeros(n)
        start = offset + s_idx * len(J)
        row[start:start + len(J)] = 1
        A_ub.append(row)
        b_ub.append(BUDGET_2)

    bounds = [(0, None)] * n

    res = linprog(
        c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        return {
            "success": False,
            "status": res.message,
            "solver": "SciPy/HiGHS",
        }

    sol = res.x

    x = {j: float(sol[j_idx]) for j_idx, j in enumerate(J)}

    y = {}
    for s_idx, s in enumerate(S):
        y[s] = {}
        for j_idx, j in enumerate(J):
            idx = offset + s_idx * len(J) + j_idx
            y[s][j] = float(sol[idx])

    return {
        "success": True,
        "status": "Optimal",
        "solver": "SciPy/HiGHS fallback",
        "objective": float(-res.fun),
        "x": x,
        "y": y,
        "source": "SciPy/HiGHS fallback",
    }


def solve_sp():
    pyomo_res = solve_sp_pyomo()

    if pyomo_res.get("success", False):
        return pyomo_res

    scipy_res = solve_sp_scipy()

    if scipy_res.get("success", False):
        scipy_res["pyomo_status"] = pyomo_res.get("status", "")
        return scipy_res

    return {
        "success": False,
        "status": f"Pyomo: {pyomo_res.get('status', '')}; SciPy: {scipy_res.get('status', '')}",
        "solver": "None",
    }


def solve_deterministic(coeff, label="Deterministic"):
    if not SCIPY_AVAILABLE:
        return {
            "success": False,
            "status": "SciPy chưa được cài",
            "label": label,
        }

    # Variables: x[4], y[4]
    n = 8
    c = np.zeros(n)

    for idx, j in enumerate(J):
        c[idx] = -coeff[j]
        c[4 + idx] = -coeff[j]

    A_ub = []
    b_ub = []

    row = np.zeros(n)
    row[:4] = 1
    A_ub.append(row)
    b_ub.append(BUDGET_1)

    row = np.zeros(n)
    row[4:] = 1
    A_ub.append(row)
    b_ub.append(BUDGET_2)

    res = linprog(
        c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        bounds=[(0, None)] * n,
        method="highs",
    )

    if not res.success:
        return {
            "success": False,
            "status": res.message,
            "label": label,
        }

    sol = res.x

    x = {j: float(sol[idx]) for idx, j in enumerate(J)}
    y = {j: float(sol[4 + idx]) for idx, j in enumerate(J)}

    return {
        "success": True,
        "status": "Optimal",
        "label": label,
        "objective": float(-res.fun),
        "x": x,
        "y": y,
        "coeff": coeff,
    }


def solve_all_deterministic_scenarios():
    results = {}

    for s in S:
        coeff = {j: BETA_S[(s, j)] for j in J}
        results[s] = solve_deterministic(coeff, label=s)

    return results


def evaluate_fixed_x_in_sp(x_fixed):
    # First-stage dùng beta nền theo cấu trúc SP gốc.
    first_value = sum(BETA[j] * x_fixed.get(j, 0.0) for j in J)

    # Khi x đã cố định, second-stage vẫn tối ưu theo từng kịch bản.
    second_value = 0.0
    y = {}

    for s in S:
        best_j = max(J, key=lambda j: BETA_S[(s, j)])
        y[s] = {j: 0.0 for j in J}
        y[s][best_j] = BUDGET_2

        second_value += P[s] * BETA_S[(s, best_j)] * BUDGET_2

    return {
        "objective": first_value + second_value,
        "x": dict(x_fixed),
        "y": y,
    }


def wait_and_see_value(det_results):
    total = 0.0

    for s in S:
        if det_results[s]["success"]:
            total += P[s] * det_results[s]["objective"]

    return total


def solve_robust_regret():
    if not SCIPY_AVAILABLE:
        return {
            "success": False,
            "status": "SciPy chưa được cài",
        }

    det_results = solve_all_deterministic_scenarios()

    w_star = {
        s: det_results[s]["objective"]
        for s in S
        if det_results[s]["success"]
    }

    if len(w_star) != len(S):
        return {
            "success": False,
            "status": "Không giải được đủ deterministic scenario để tính regret",
        }

    # Variables:
    # x[4], y_sj[16], R
    n_x = len(J)
    n_y = len(S) * len(J)
    idx_R = n_x + n_y
    n = idx_R + 1

    c = np.zeros(n)
    c[idx_R] = 1.0

    A_ub = []
    b_ub = []

    # first-stage budget
    row = np.zeros(n)
    row[:n_x] = 1
    A_ub.append(row)
    b_ub.append(BUDGET_1)

    # second-stage budget per scenario
    offset = n_x
    for s_idx, s in enumerate(S):
        row = np.zeros(n)
        start = offset + s_idx * len(J)
        row[start:start + len(J)] = 1
        A_ub.append(row)
        b_ub.append(BUDGET_2)

    # regret_s = W*_s - value_s(x,y_s) <= R
    # - value_s(x,y_s) - R <= - W*_s
    for s_idx, s in enumerate(S):
        row = np.zeros(n)

        for j_idx, j in enumerate(J):
            row[j_idx] = -BETA_S[(s, j)]

            y_idx = offset + s_idx * len(J) + j_idx
            row[y_idx] = -BETA_S[(s, j)]

        row[idx_R] = -1.0

        A_ub.append(row)
        b_ub.append(-w_star[s])

    bounds = [(0, None)] * n

    res = linprog(
        c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        return {
            "success": False,
            "status": res.message,
        }

    sol = res.x

    x = {j: float(sol[j_idx]) for j_idx, j in enumerate(J)}

    y = {}
    for s_idx, s in enumerate(S):
        y[s] = {}
        for j_idx, j in enumerate(J):
            y_idx = offset + s_idx * len(J) + j_idx
            y[s][j] = float(sol[y_idx])

    regret_rows = []

    for s in S:
        value_s = sum(BETA_S[(s, j)] * x[j] for j in J)
        value_s += sum(BETA_S[(s, j)] * y[s][j] for j in J)

        regret_rows.append(
            {
                "Kịch bản": s,
                "Tên kịch bản": S_LABELS[s],
                "Best deterministic": w_star[s],
                "Robust value": value_s,
                "Regret": w_star[s] - value_s,
            }
        )

    expected_value = 0.0

    for s in S:
        scenario_value = sum(BETA_S[(s, j)] * x[j] for j in J)
        scenario_value += sum(BETA_S[(s, j)] * y[s][j] for j in J)
        expected_value += P[s] * scenario_value

    return {
        "success": True,
        "status": "Optimal",
        "x": x,
        "y": y,
        "max_regret": float(sol[idx_R]),
        "expected_value": float(expected_value),
        "regret_table": pd.DataFrame(regret_rows),
    }


def first_stage_table(x_dict, label):
    return pd.DataFrame(
        {
            "Hạng mục": [J_LABELS[j] for j in J],
            "Mã": J,
            label: [x_dict.get(j, 0.0) for j in J],
        }
    )


def second_stage_table(y_dict):
    rows = []

    for s in S:
        for j in J:
            rows.append(
                {
                    "Kịch bản": s,
                    "Tên kịch bản": S_LABELS[s],
                    "Hạng mục": J_LABELS[j],
                    "Mã": j,
                    "y_sj": y_dict[s].get(j, 0.0),
                    "Beta_s": BETA_S[(s, j)],
                }
            )

    return pd.DataFrame(rows)


def deterministic_summary_table(det_results):
    rows = []

    for s in S:
        res = det_results[s]

        if res["success"]:
            best_x = max(res["x"], key=res["x"].get)
            best_y = max(res["y"], key=res["y"].get)

            rows.append(
                {
                    "Kịch bản": s,
                    "Tên kịch bản": S_LABELS[s],
                    "Trạng thái": res["status"],
                    "Objective": res["objective"],
                    "First-stage ưu tiên": J_LABELS[best_x],
                    "Second-stage ưu tiên": J_LABELS[best_y],
                    "x_I": res["x"]["I"],
                    "x_D": res["x"]["D"],
                    "x_AI": res["x"]["AI"],
                    "x_H": res["x"]["H"],
                }
            )
        else:
            rows.append(
                {
                    "Kịch bản": s,
                    "Tên kịch bản": S_LABELS[s],
                    "Trạng thái": res["status"],
                    "Objective": np.nan,
                    "First-stage ưu tiên": "",
                    "Second-stage ưu tiên": "",
                    "x_I": np.nan,
                    "x_D": np.nan,
                    "x_AI": np.nan,
                    "x_H": np.nan,
                }
            )

    return pd.DataFrame(rows)


def compare_first_stage_table(sp_res, ev_res, robust_res):
    rows = []

    for j in J:
        rows.append(
            {
                "Hạng mục": J_LABELS[j],
                "Mã": j,
                "SP": sp_res["x"].get(j, 0.0) if sp_res["success"] else np.nan,
                "EV": ev_res["x"].get(j, 0.0) if ev_res["success"] else np.nan,
                "Robust": robust_res["x"].get(j, 0.0) if robust_res["success"] else np.nan,
            }
        )

    return pd.DataFrame(rows)


def value_table(sp_res, ev_eval, wait_see, robust_res):
    sp_value = sp_res["objective"] if sp_res["success"] else np.nan
    ev_value = ev_eval["objective"]
    robust_value = robust_res["expected_value"] if robust_res["success"] else np.nan

    vss = sp_value - ev_value
    evpi = wait_see - sp_value

    return pd.DataFrame(
        {
            "Chỉ tiêu": [
                "SP - Stochastic solution",
                "EEV - EV solution evaluated in SP",
                "Wait-and-see",
                "Robust expected value",
                "VSS = SP - EEV",
                "EVPI = WS - SP",
            ],
            "Giá trị": [
                sp_value,
                ev_value,
                wait_see,
                robust_value,
                vss,
                evpi,
            ],
        }
    )


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
    st.title("Bài 10. Two-stage Stochastic Programming")
    st.caption("Pyomo first-stage/second-stage, EV, SP, VSS, EVPI và robust regret")

    if not PYOMO_AVAILABLE and not SCIPY_AVAILABLE:
        st.error("Cần cài `pyomo` hoặc `scipy` để chạy bài này.")
        return

    sp_res = solve_sp()
    det_results = solve_all_deterministic_scenarios()

    mean_coeff = expected_coefficients()
    ev_res = solve_deterministic(mean_coeff, label="EV")
    ev_eval = evaluate_fixed_x_in_sp(ev_res["x"]) if ev_res["success"] else {"objective": np.nan, "x": {}, "y": {}}

    ws_value = wait_and_see_value(det_results)
    robust_res = solve_robust_regret()

    tabs = st.tabs(
        [
            "Mô hình two-stage bằng Pyomo",
            " EV & Deterministic",
            " VSS & EVPI",
            "Robust optimization: cực tiểu hóa regret xấu nhất",
            "Thảo luận Chính sách",
        ]
    )

    # =====================================================
    # 10.5.1
    # =====================================================
    with tabs[0]:


        c1, c2, c3 = st.columns(3)
        c1.metric("Pyomo", "Có" if PYOMO_AVAILABLE else "Không")
        c2.metric("Solver dùng", sp_res.get("source", sp_res.get("solver", "None")))
        c3.metric("Trạng thái", sp_res.get("status", ""))

        if "pyomo_status" in sp_res:
            st.info(
                f"Pyomo chưa giải được do: {sp_res['pyomo_status']}. "
                "App đang dùng SciPy/HiGHS fallback để tránh lỗi deploy."
            )

        st.subheader("Hệ số first-stage")
        show_table(beta_table(), decimals=3)

        st.subheader("Hệ số second-stage theo kịch bản")
        show_table(scenario_table(), decimals=3)

        if not sp_res["success"]:
            st.error(sp_res["status"])
        else:
            st.subheader("Quyết định first-stage tối ưu")
            show_table(first_stage_table(sp_res["x"], "x_j - SP"), decimals=2)

            st.subheader("Quyết định second-stage theo kịch bản")
            show_table(second_stage_table(sp_res["y"]), decimals=2)

            c4, c5, c6 = st.columns(3)
            c4.metric("Objective SP", f"{sp_res['objective']:,.1f}")
            c5.metric("Budget first-stage", f"{sum(sp_res['x'].values()):,.0f}")
            c6.metric("Budget second mỗi kịch bản", f"{BUDGET_2:,.0f}")

            x_plot = first_stage_table(sp_res["x"], "x_j - SP")

            fig = px.bar(
                x_plot,
                x="Hạng mục",
                y="x_j - SP",
                color="Mã",
                text=x_plot["x_j - SP"].round(0),
                title="First-stage decision của lời giải stochastic SP",
                color_discrete_map=MULTI_COLORS,
            )
            fig.update_traces(textposition="outside", marker_line_color="white", marker_line_width=1)
            fig.update_layout(xaxis_title="Hạng mục", yaxis_title="Ngân sách")
            style_base_fig(fig, height=430)
            st.plotly_chart(fig, use_container_width=True)

            st.success(
                "Mô hình đã tách rõ first-stage x_j và second-stage y_sj. "
                "Second-stage được điều chỉnh theo từng kịch bản sau khi bất định được quan sát."
            )

    # =====================================================
    # 10.5.2
    # =====================================================
    with tabs[1]:
     

        st.subheader("Bài toán xác định theo từng kịch bản")
        det_summary = deterministic_summary_table(det_results)
        show_table(det_summary, decimals=2)

        st.subheader("Hệ số kỳ vọng EV")
        ev_coeff_df = pd.DataFrame(
            {
                "Hạng mục": [J_LABELS[j] for j in J],
                "Mã": J,
                "Beta kỳ vọng": [mean_coeff[j] for j in J],
            }
        )
        show_table(ev_coeff_df, decimals=4)

        if ev_res["success"]:
            st.subheader("First-stage của lời giải EV")
            show_table(first_stage_table(ev_res["x"], "x_j - EV"), decimals=2)

        if sp_res["success"] and ev_res["success"] and robust_res["success"]:
            compare_x = compare_first_stage_table(sp_res, ev_res, robust_res)
            st.subheader("So sánh first-stage: SP, EV, Robust")
            show_table(compare_x, decimals=2)

            compare_long = compare_x.melt(
                id_vars=["Hạng mục", "Mã"],
                value_vars=["SP", "EV", "Robust"],
                var_name="Phương pháp",
                value_name="Ngân sách",
            )

            fig = px.bar(
                compare_long,
                x="Hạng mục",
                y="Ngân sách",
                color="Phương pháp",
                barmode="group",
                title="So sánh quyết định first-stage",
                color_discrete_map=MULTI_COLORS,
            )
            fig.update_traces(marker_line_color="white", marker_line_width=1)
            fig.update_layout(xaxis_title="Hạng mục", yaxis_title="Ngân sách")
            style_base_fig(fig, height=460)
            st.plotly_chart(fig, use_container_width=True)

        st.success(
            "Lời giải deterministic theo từng kịch bản cho thấy nếu biết trước kịch bản xấu, quyết định first-stage có thể chuyển sang H như một tài sản bảo hiểm."
        )

    # =====================================================
    # 10.5.3
    # =====================================================
    with tabs[2]:
   

        if not sp_res["success"]:
            st.error("Chưa có nghiệm SP để tính VSS và EVPI.")
        else:
            values_df = value_table(sp_res, ev_eval, ws_value, robust_res)
            show_table(values_df, decimals=3)

            value_plot = values_df[
                values_df["Chỉ tiêu"].isin(
                    [
                        "SP - Stochastic solution",
                        "EEV - EV solution evaluated in SP",
                        "Wait-and-see",
                        "Robust expected value",
                    ]
                )
            ].copy()

            fig = px.bar(
                value_plot,
                x="Chỉ tiêu",
                y="Giá trị",
                text=value_plot["Giá trị"].round(1),
                title="So sánh giá trị SP, EEV, Wait-and-see và Robust",
            )
            fig.update_traces(
                marker_color=BRAND,
                textposition="outside",
                textfont=dict(color=BRAND),
            )
            fig.update_layout(xaxis_title="Chỉ tiêu", yaxis_title="Giá trị mục tiêu")
            style_base_fig(fig, height=480)
            st.plotly_chart(fig, use_container_width=True)

            vss = values_df.loc[values_df["Chỉ tiêu"] == "VSS = SP - EEV", "Giá trị"].iloc[0]
            evpi = values_df.loc[values_df["Chỉ tiêu"] == "EVPI = WS - SP", "Giá trị"].iloc[0]

            c1, c2 = st.columns(2)
            c1.metric("VSS", f"{vss:,.3f}")
            c2.metric("EVPI", f"{evpi:,.3f}")

            st.success(
                "VSS đo lợi ích của việc dùng mô hình stochastic thay vì chỉ dùng kịch bản kỳ vọng. "
                "EVPI đo mức sẵn sàng trả tối đa cho thông tin hoàn hảo về kịch bản tương lai."
            )

    # =====================================================
    # 10.5.4
    # =====================================================
    with tabs[3]:
    

        if not robust_res["success"]:
            st.error(robust_res["status"])
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Trạng thái", robust_res["status"])
            c2.metric("Max regret", f"{robust_res['max_regret']:,.1f}")
            c3.metric("Expected value", f"{robust_res['expected_value']:,.1f}")

            st.subheader("First-stage robust")
            show_table(first_stage_table(robust_res["x"], "x_j - Robust"), decimals=2)

            st.subheader("Regret theo kịch bản")
            show_table(robust_res["regret_table"], decimals=3)

            regret_plot = robust_res["regret_table"].copy()

            fig = px.bar(
                regret_plot,
                x="Tên kịch bản",
                y="Regret",
                text=regret_plot["Regret"].round(1),
                title="Regret của nghiệm robust theo từng kịch bản",
            )
            fig.update_traces(
                marker_color=BRAND,
                textposition="outside",
                textfont=dict(color=BRAND),
            )
            fig.update_layout(xaxis_title="Kịch bản", yaxis_title="Regret")
            style_base_fig(fig, height=430)
            st.plotly_chart(fig, use_container_width=True)

            if sp_res["success"]:
                compare_x = compare_first_stage_table(sp_res, ev_res, robust_res)
                show_table(compare_x, decimals=2)

            st.success(
                "Robust regret không tối đa hóa kỳ vọng thuần túy, mà tìm quyết định giảm thiệt hại tương đối trong kịch bản xấu nhất."
            )

    # =====================================================
    # 10.6
    # =====================================================
    with tabs[4]:
 

        st.markdown("### a) SP đầu tư H nhiều hơn hay ít hơn so với lời giải xác định?")

        st.markdown(
            """
            Lời giải SP cân bằng giữa lợi ích kỳ vọng và khả năng điều chỉnh ở second-stage.
            Với cấu trúc dữ liệu của bài, first-stage theo SP thường vẫn ưu tiên hạng mục có beta nền cao nhất.
            Tuy nhiên, khi xét từng kịch bản xác định, các kịch bản xấu có xu hướng làm H trở nên hấp dẫn hơn,
            vì nhân lực số có vai trò như một năng lực chống chịu.

            Do đó, nếu mô hình mở rộng để hệ số first-stage cũng chịu tác động mạnh bởi kịch bản,
            hoặc thêm ràng buộc phục hồi sau cú sốc, SP có thể đầu tư H nhiều hơn so với lời giải kỳ vọng EV.
            """
        )

        st.markdown("### b) VSS dương nói lên điều gì?")

        st.markdown(
            """
            VSS dương nghĩa là việc cân nhắc bất định ngay từ lúc ra quyết định first-stage tạo ra giá trị thực.
            Nói cách khác, lập kế hoạch theo xác suất giúp tránh quyết định quá thiên về kịch bản trung bình.

            Trong hoạch định chính sách Việt Nam, điều này rất quan trọng vì các cú sốc như dịch bệnh,
            thiên tai, gián đoạn chuỗi cung ứng hoặc biến động công nghệ có thể làm thay đổi hiệu quả đầu tư.
            Nếu chỉ dùng một kịch bản trung bình, chính sách dễ đánh giá thấp nhu cầu đầu tư vào năng lực chống chịu,
            như nhân lực số, dữ liệu công, an ninh mạng và năng lực vận hành từ xa.
            """
        )

        st.markdown("### c) Việt Nam có đang dưới đầu tư vào nhân lực số như một hàng hóa bảo hiểm không?")

        st.markdown(
            """
            Đại dịch COVID-19 và bão Yagi cho thấy năng lực số không chỉ tạo tăng trưởng,
            mà còn là năng lực duy trì hoạt động trong khủng hoảng. Nhân lực số giúp doanh nghiệp,
            cơ quan nhà nước và người lao động chuyển sang mô hình trực tuyến, xử lý dữ liệu,
            vận hành hệ thống số và thích ứng nhanh hơn.

            Vì lợi ích của nhân lực số thường xuất hiện rõ nhất khi có cú sốc,
            thị trường và chính sách ngắn hạn có thể đánh giá thấp khoản đầu tư này.
            Đây là lý do nhân lực số có thể được xem như một **hàng hóa bảo hiểm**:
            bình thường có vẻ không tạo lợi ích biên cao nhất, nhưng khi khủng hoảng xảy ra,
            nó làm giảm tổn thất hệ thống.

            Bài học chính sách là Việt Nam không nên chỉ đầu tư vào AI, hạ tầng hay dữ liệu theo logic tăng trưởng,
            mà cần đầu tư đủ vào H để bảo đảm khả năng hấp thụ, phục hồi và chống chịu.
            """
        )

        st.info(
            "Kết luận: SP giúp đưa bất định vào quyết định first-stage; robust regret giúp kiểm soát thiệt hại trong kịch bản xấu nhất."
        )


def run():
    render()

# bai08_dynamic_2026_2035.py

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


BRAND = "#053151"

MULTI_COLORS = {
    "K": "#053151",
    "D": "#E76F51",
    "AI": "#2A9D8F",
    "H": "#F4A261",
    "Y": "#053151",
    "C": "#E76F51",
    "Kế hoạch gốc": "#053151",
    "Có cú sốc": "#E76F51",
    "Tối ưu SLSQP": "#053151",
    "Đầu tư trải đều": "#E76F51",
    "Đầu tư front-load": "#2A9D8F",
}


# =========================================================
# BÀI 8 — MÔ HÌNH ĐỘNG COBB-DOUGLAS 2026-2035
# Cách B: scipy.optimize.minimize SLSQP
# =========================================================


T = 10
YEARS = np.arange(2026, 2036)

# Hệ số Cobb-Douglas
ALPHA = 0.33
BETA_L = 0.42
GAMMA = 0.10
DELTA = 0.08
THETA = 0.07

# Trạng thái ban đầu
K0 = 27500.0
D0 = 20.3
AI0 = 86.0
H0 = 30.0
L0 = 54.0

# Tăng trưởng ngoại sinh
G_L = 0.004
G_A = 0.012

# Khấu hao / suy giảm
DEP_K = 0.05
DEP_D = 0.04
DEP_AI = 0.05
DEP_H = 0.02

# Hiệu quả chuyển đầu tư thành trạng thái
PHI_D = 0.004
PHI_AI = 0.020
PHI_H = 0.012

# Trần trạng thái
D_MAX = 55.0
AI_MAX = 180.0
H_MAX = 100.0

# Ràng buộc mềm
MIN_CONSUME_SHARE = 0.55
MAX_INVEST_SHARE = 0.38

# Hiệu chỉnh A0 để Y 2026 hợp lý
Y0_TARGET = 14000.0

A0 = Y0_TARGET / (
    (K0 ** ALPHA)
    * (L0 ** BETA_L)
    * (D0 ** GAMMA)
    * (AI0 ** DELTA)
    * (H0 ** THETA)
)

TERMINAL_WEIGHT = 8.0


def production(K, D, AI, H, t):
    L_t = L0 * ((1 + G_L) ** t)
    A_t = A0 * ((1 + G_A) ** t)

    return (
        A_t
        * (max(K, 1e-9) ** ALPHA)
        * (max(L_t, 1e-9) ** BETA_L)
        * (max(D, 1e-9) ** GAMMA)
        * (max(AI, 1e-9) ** DELTA)
        * (max(H, 1e-9) ** THETA)
    )


def simulate_path(x_vec, shock_year=None, shock_factor=1.0):
    X = np.asarray(x_vec, dtype=float).reshape(T, 4)

    K = np.zeros(T + 1)
    D = np.zeros(T + 1)
    AI = np.zeros(T + 1)
    H = np.zeros(T + 1)

    Y = np.zeros(T)
    C = np.zeros(T)

    K[0] = K0
    D[0] = D0
    AI[0] = AI0
    H[0] = H0

    for t in range(T):
        y_t = production(K[t], D[t], AI[t], H[t], t)

        if shock_year is not None and YEARS[t] == shock_year:
            y_t = y_t * shock_factor

        Y[t] = y_t
        C[t] = Y[t] - X[t].sum()

        K[t + 1] = (1 - DEP_K) * K[t] + X[t, 0]
        D[t + 1] = (1 - DEP_D) * D[t] + PHI_D * X[t, 1]
        AI[t + 1] = (1 - DEP_AI) * AI[t] + PHI_AI * X[t, 2]
        H[t + 1] = (1 - DEP_H) * H[t] + PHI_H * X[t, 3]

    terminal_y = production(K[T], D[T], AI[T], H[T], T)

    return {
        "X": X,
        "K": K,
        "D": D,
        "AI": AI,
        "H": H,
        "Y": Y,
        "C": C,
        "terminal_y": terminal_y,
    }


def welfare_from_sim(sim, rho=0.97):
    C = sim["C"]

    if np.any(C <= 1):
        return -1e9

    flow_welfare = sum((rho ** t) * np.log(C[t]) for t in range(T))
    terminal_value = TERMINAL_WEIGHT * (rho ** T) * np.log(max(sim["terminal_y"], 1e-9))

    return flow_welfare + terminal_value


def penalty_from_sim(sim):
    X = sim["X"]

    violations = []

    violations.extend(MIN_CONSUME_SHARE * sim["Y"] - sim["C"])
    violations.extend(X.sum(axis=1) - MAX_INVEST_SHARE * sim["Y"])
    violations.extend(sim["D"][1:] - D_MAX)
    violations.extend(sim["AI"][1:] - AI_MAX)
    violations.extend(sim["H"][1:] - H_MAX)
    violations.extend(1 - sim["C"])

    v = np.maximum(0, np.array(violations, dtype=float))

    return 1e-5 * np.sum(v ** 2) + 1000 * np.sum(v > 0)


def objective_from_x(x_vec, rho=0.97, shock_year=None, shock_factor=1.0):
    sim = simulate_path(x_vec, shock_year=shock_year, shock_factor=shock_factor)
    welfare = welfare_from_sim(sim, rho=rho)
    penalty = penalty_from_sim(sim)

    return -welfare + penalty


# Dùng cache_resource thay vì cache_data để tránh lỗi pickle dict/numpy/scipy object
@st.cache_resource(show_spinner=False)
def solve_slsqp_cached(rho=0.97, shock_year=0, shock_factor=1.0):
    if shock_year == 0:
        shock_year_real = None
    else:
        shock_year_real = int(shock_year)

    scales = np.tile(np.array([5500.0, 2500.0, 2500.0, 2500.0]), T)

    x0_real = np.tile(np.array([2800.0, 650.0, 450.0, 600.0]), T)
    u0 = x0_real / scales

    callback_history = []

    def obj_u(u):
        x = np.asarray(u, dtype=float) * scales
        return objective_from_x(
            x,
            rho=float(rho),
            shock_year=shock_year_real,
            shock_factor=float(shock_factor),
        )

    def callback(u):
        x = np.asarray(u, dtype=float) * scales
        sim = simulate_path(x, shock_year=shock_year_real, shock_factor=float(shock_factor))
        callback_history.append(
            {
                "Iteration": len(callback_history) + 1,
                "Welfare": float(welfare_from_sim(sim, rho=float(rho))),
                "Penalty": float(penalty_from_sim(sim)),
            }
        )

    try:
        res = minimize(
            obj_u,
            u0,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * (T * 4),
            callback=callback,
            options={
                "maxiter": 160,
                "ftol": 1e-8,
                "disp": False,
            },
        )

        x_opt = np.asarray(res.x, dtype=float) * scales

        sim = simulate_path(
            x_opt,
            shock_year=shock_year_real,
            shock_factor=float(shock_factor),
        )

        return {
            "success": bool(res.success),
            "message": str(res.message),
            "x": x_opt,
            "sim": sim,
            "welfare": float(welfare_from_sim(sim, rho=float(rho))),
            "penalty": float(penalty_from_sim(sim)),
            "callback": pd.DataFrame(callback_history),
            "rho": float(rho),
            "shock_year": shock_year_real,
            "shock_factor": float(shock_factor),
        }

    except Exception as exc:
        x_fallback = x0_real.copy()
        sim = simulate_path(
            x_fallback,
            shock_year=shock_year_real,
            shock_factor=float(shock_factor),
        )

        return {
            "success": False,
            "message": str(exc),
            "x": x_fallback,
            "sim": sim,
            "welfare": float(welfare_from_sim(sim, rho=float(rho))),
            "penalty": float(penalty_from_sim(sim)),
            "callback": pd.DataFrame(callback_history),
            "rho": float(rho),
            "shock_year": shock_year_real,
            "shock_factor": float(shock_factor),
        }


def path_table(sim):
    X = sim["X"]

    return pd.DataFrame(
        {
            "Năm": YEARS,
            "K": sim["K"][:-1],
            "D": sim["D"][:-1],
            "AI": sim["AI"][:-1],
            "H": sim["H"][:-1],
            "Y": sim["Y"],
            "C": sim["C"],
            "I_K": X[:, 0],
            "I_D": X[:, 1],
            "I_AI": X[:, 2],
            "I_H": X[:, 3],
            "Tổng đầu tư": X.sum(axis=1),
        }
    )


def terminal_state_table(sim):
    return pd.DataFrame(
        {
            "Trạng thái": ["K 2036", "D 2036", "AI 2036", "H 2036", "Y terminal"],
            "Giá trị": [
                sim["K"][-1],
                sim["D"][-1],
                sim["AI"][-1],
                sim["H"][-1],
                sim["terminal_y"],
            ],
        }
    )


def build_index_state_df(sim):
    base = {
        "K": sim["K"][0],
        "D": sim["D"][0],
        "AI": sim["AI"][0],
        "H": sim["H"][0],
    }

    rows = []

    for t in range(T):
        rows.append(
            {
                "Năm": YEARS[t],
                "K": sim["K"][t] / base["K"] * 100,
                "D": sim["D"][t] / base["D"] * 100,
                "AI": sim["AI"][t] / base["AI"] * 100,
                "H": sim["H"][t] / base["H"] * 100,
            }
        )

    return pd.DataFrame(rows)


def simulate_rule_strategy(strategy="even", rho=0.97):
    if strategy == "front":
        rates = np.array([0.34, 0.32, 0.30, 0.25, 0.22, 0.19, 0.16, 0.15, 0.14, 0.13])
    else:
        rates = np.ones(T) * 0.22

    shares = np.array([0.45, 0.18, 0.15, 0.22])

    K = K0
    D = D0
    AI = AI0
    H = H0

    X = np.zeros((T, 4))

    for t in range(T):
        y_t = production(K, D, AI, H, t)
        total_invest = rates[t] * y_t
        X[t, :] = total_invest * shares

        K = (1 - DEP_K) * K + X[t, 0]
        D = (1 - DEP_D) * D + PHI_D * X[t, 1]
        AI = (1 - DEP_AI) * AI + PHI_AI * X[t, 2]
        H = (1 - DEP_H) * H + PHI_H * X[t, 3]

    sim = simulate_path(X.reshape(-1))
    welfare = welfare_from_sim(sim, rho=rho)

    return {
        "x": X.reshape(-1),
        "sim": sim,
        "welfare": float(welfare),
        "penalty": float(penalty_from_sim(sim)),
    }


def compare_shock_table(base_sim, shock_sim):
    base = path_table(base_sim)
    shock = path_table(shock_sim)

    return pd.DataFrame(
        {
            "Năm": YEARS,
            "Y kế hoạch": base["Y"],
            "Y có cú sốc": shock["Y"],
            "C kế hoạch": base["C"],
            "C có cú sốc": shock["C"],
            "Đầu tư kế hoạch": base["Tổng đầu tư"],
            "Đầu tư có cú sốc": shock["Tổng đầu tư"],
            "Chênh lệch đầu tư": shock["Tổng đầu tư"] - base["Tổng đầu tư"],
        }
    )


def investment_adjustment_after_shock(base_sim, shock_sim):
    base = path_table(base_sim)
    shock = path_table(shock_sim)

    post = base["Năm"] >= 2029

    rows = []

    for col, label in [
        ("I_K", "K"),
        ("I_D", "D"),
        ("I_AI", "AI"),
        ("I_H", "H"),
    ]:
        rows.append(
            {
                "Hạng mục": label,
                "Đầu tư gốc sau 2028": base.loc[post, col].sum(),
                "Đầu tư sốc sau 2028": shock.loc[post, col].sum(),
                "Chênh lệch": shock.loc[post, col].sum() - base.loc[post, col].sum(),
            }
        )

    return pd.DataFrame(rows)


def strategy_compare_table(opt_res, even_res, front_res):
    rows = []

    for name, res in [
        ("Tối ưu SLSQP", opt_res),
        ("Đầu tư trải đều", even_res),
        ("Đầu tư front-load", front_res),
    ]:
        sim = res["sim"]
        path = path_table(sim)

        rows.append(
            {
                "Chiến lược": name,
                "Welfare tổng": res["welfare"],
                "Penalty": res["penalty"],
                "Tổng đầu tư": path["Tổng đầu tư"].sum(),
                "Y 2035": path["Y"].iloc[-1],
                "C bình quân": path["C"].mean(),
                "Terminal Y": sim["terminal_y"],
            }
        )

    return pd.DataFrame(rows).sort_values("Welfare tổng", ascending=False).reset_index(drop=True)


def rho_policy_table(opt_097, opt_090):
    path_097 = path_table(opt_097["sim"])
    path_090 = path_table(opt_090["sim"])

    return pd.DataFrame(
        {
            "Chỉ tiêu": [
                "Welfare",
                "Tổng đầu tư",
                "Đầu tư AI",
                "Đầu tư H",
                "C bình quân",
                "Y 2035",
                "Terminal Y",
            ],
            "ρ = 0.97": [
                opt_097["welfare"],
                path_097["Tổng đầu tư"].sum(),
                path_097["I_AI"].sum(),
                path_097["I_H"].sum(),
                path_097["C"].mean(),
                path_097["Y"].iloc[-1],
                opt_097["sim"]["terminal_y"],
            ],
            "ρ = 0.90": [
                opt_090["welfare"],
                path_090["Tổng đầu tư"].sum(),
                path_090["I_AI"].sum(),
                path_090["I_H"].sum(),
                path_090["C"].mean(),
                path_090["Y"].iloc[-1],
                opt_090["sim"]["terminal_y"],
            ],
        }
    )


def make_styled_table(df, decimals=3):
    show_df = df.copy()

    format_dict = {}

    for col in show_df.columns:
        if pd.api.types.is_numeric_dtype(show_df[col]):
            if str(col).lower() in ["năm", "iteration"]:
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
    st.title("Bài 8. Mô hình động Cobb-Douglas 2026-2035")
    st.caption("Tối ưu động bằng scipy SLSQP, quỹ đạo đầu tư, cú sốc 2028 và so sánh chiến lược đầu tư")

    if not SCIPY_AVAILABLE:
        st.error("Chưa cài scipy. Hãy thêm `scipy` vào requirements.txt.")
        return

    opt_res = solve_slsqp_cached(rho=0.97, shock_year=0, shock_factor=1.0)
    shock_res = solve_slsqp_cached(rho=0.97, shock_year=2028, shock_factor=0.92)
    opt_short_res = solve_slsqp_cached(rho=0.90, shock_year=0, shock_factor=1.0)

    even_res = simulate_rule_strategy(strategy="even", rho=0.97)
    front_res = simulate_rule_strategy(strategy="front", rho=0.97)

    path_opt = path_table(opt_res["sim"])

    tabs = st.tabs(
        [
            "Cài đặt mô hình bằng SLSQP",
            "Quỹ đạo tối ưu",
            "Cú sốc 2028",
            " So sánh chiến lược",
            "Thảo luận Chính sách",
        ]
    )

    # =====================================================
    # 8.3.1
    # =====================================================
    with tabs[0]:


        c1, c2, c3 = st.columns(3)
        c1.metric("Trạng thái", "Optimal" if opt_res["success"] else "Kiểm tra")
        c2.metric("Welfare", f"{opt_res['welfare']:.3f}")
        c3.metric("Penalty", f"{opt_res['penalty']:.4f}")

        if not opt_res["success"]:
            st.warning(opt_res["message"])

        model_df = pd.DataFrame(
            {
                "Thành phần": [
                    "Hàm mục tiêu",
                    "Phương pháp",
                    "Giai đoạn",
                    "Số biến quyết định",
                    "ρ",
                    "Terminal weight",
                    "Ràng buộc tiêu dùng",
                    "Ràng buộc đầu tư",
                ],
                "Giá trị": [
                    "Max Σ ρ^t log(C_t) + terminal value",
                    "scipy.optimize.minimize - SLSQP",
                    "2026-2035",
                    "40 biến đầu tư: I_K, I_D, I_AI, I_H trong 10 năm",
                    "0.97",
                    TERMINAL_WEIGHT,
                    f"C_t >= {MIN_CONSUME_SHARE:.0%}Y_t",
                    f"I_t <= {MAX_INVEST_SHARE:.0%}Y_t",
                ],
            }
        )

        show_table(model_df, decimals=3)

        param_df = pd.DataFrame(
            {
                "Tham số": [
                    "K0",
                    "D0",
                    "AI0",
                    "H0",
                    "L0",
                    "α",
                    "β",
                    "γ",
                    "δ",
                    "θ",
                    "A0 hiệu chỉnh",
                ],
                "Giá trị": [
                    K0,
                    D0,
                    AI0,
                    H0,
                    L0,
                    ALPHA,
                    BETA_L,
                    GAMMA,
                    DELTA,
                    THETA,
                    A0,
                ],
            }
        )

        st.subheader("Tham số mô hình")
        show_table(param_df, decimals=4)

        st.subheader("Callback SLSQP")
        cb = opt_res["callback"]

        if len(cb) > 0:
            show_table(cb.tail(15), decimals=5)

            fig_cb = px.line(
                cb,
                x="Iteration",
                y="Welfare",
                markers=True,
                title="Quỹ đạo hội tụ của welfare trong callback",
            )
            fig_cb.update_traces(
                line=dict(color=BRAND, width=4),
                marker=dict(color=BRAND, size=8),
            )
            fig_cb.update_layout(xaxis_title="Iteration", yaxis_title="Welfare")
            style_base_fig(fig_cb, height=390)
            st.plotly_chart(fig_cb, use_container_width=True)
        else:
            st.info("SLSQP hội tụ nhanh nên callback có ít quan sát.")

        st.success(
            "Mô hình đã được giải bằng SLSQP. Biến quyết định là quỹ đạo đầu tư vào K, D, AI và H trong giai đoạn 2026-2035."
        )

    # =====================================================
    # 8.3.2
    # =====================================================
    with tabs[1]:
   

        c1, c2, c3 = st.columns(3)
        c1.metric("Y 2026", f"{path_opt['Y'].iloc[0]:,.1f}")
        c2.metric("Y 2035", f"{path_opt['Y'].iloc[-1]:,.1f}")
        c3.metric("C bình quân", f"{path_opt['C'].mean():,.1f}")

        st.subheader("Bảng quỹ đạo tối ưu")
        show_table(path_opt, decimals=3)

        st.subheader("Trạng thái cuối kỳ")
        show_table(terminal_state_table(opt_res["sim"]), decimals=3)

        index_df = build_index_state_df(opt_res["sim"])
        index_long = index_df.melt(
            id_vars="Năm",
            value_vars=["K", "D", "AI", "H"],
            var_name="Biến trạng thái",
            value_name="Chỉ số 2026=100",
        )

        fig_state = px.line(
            index_long,
            x="Năm",
            y="Chỉ số 2026=100",
            color="Biến trạng thái",
            markers=True,
            title="Quỹ đạo K, D, AI, H - chuẩn hóa 2026 = 100",
            color_discrete_map=MULTI_COLORS,
        )
        fig_state.update_traces(line=dict(width=4), marker=dict(size=8))
        fig_state.update_layout(xaxis_title="Năm", yaxis_title="Chỉ số")
        style_base_fig(fig_state, height=470)
        st.plotly_chart(fig_state, use_container_width=True)

        yc_long = path_opt[["Năm", "Y", "C"]].melt(
            id_vars="Năm",
            value_vars=["Y", "C"],
            var_name="Biến",
            value_name="Giá trị",
        )

        fig_yc = px.line(
            yc_long,
            x="Năm",
            y="Giá trị",
            color="Biến",
            markers=True,
            title="Quỹ đạo GDP Y và tiêu dùng C",
            color_discrete_map=MULTI_COLORS,
        )
        fig_yc.update_traces(line=dict(width=4), marker=dict(size=8))
        fig_yc.update_layout(xaxis_title="Năm", yaxis_title="Nghìn tỷ VND")
        style_base_fig(fig_yc, height=470)
        st.plotly_chart(fig_yc, use_container_width=True)

        invest_long = path_opt[["Năm", "I_K", "I_D", "I_AI", "I_H"]].rename(
            columns={
                "I_K": "K",
                "I_D": "D",
                "I_AI": "AI",
                "I_H": "H",
            }
        ).melt(
            id_vars="Năm",
            value_vars=["K", "D", "AI", "H"],
            var_name="Hạng mục đầu tư",
            value_name="Đầu tư",
        )

        fig_inv = px.bar(
            invest_long,
            x="Năm",
            y="Đầu tư",
            color="Hạng mục đầu tư",
            title="Phân bổ đầu tư tối ưu theo năm",
            color_discrete_map=MULTI_COLORS,
        )
        fig_inv.update_traces(marker_line_color="white", marker_line_width=1)
        fig_inv.update_layout(barmode="stack", xaxis_title="Năm", yaxis_title="Nghìn tỷ VND")
        style_base_fig(fig_inv, height=470)
        st.plotly_chart(fig_inv, use_container_width=True)

        st.success(
            "Quỹ đạo tối ưu cho thấy đầu tư được phân bổ động theo thời gian, không cố định đều giữa các năm."
        )

    # =====================================================
    # 8.3.3
    # =====================================================
    with tabs[2]:


        shock_table = compare_shock_table(opt_res["sim"], shock_res["sim"])

        c1, c2, c3 = st.columns(3)
        y2028_base = shock_table.loc[shock_table["Năm"] == 2028, "Y kế hoạch"].iloc[0]
        y2028_shock = shock_table.loc[shock_table["Năm"] == 2028, "Y có cú sốc"].iloc[0]

        c1.metric("Y 2028 kế hoạch", f"{y2028_base:,.1f}")
        c2.metric("Y 2028 cú sốc", f"{y2028_shock:,.1f}")
        c3.metric("Welfare thay đổi", f"{shock_res['welfare'] - opt_res['welfare']:.3f}")

        show_table(shock_table, decimals=3)

        y_compare = shock_table[["Năm", "Y kế hoạch", "Y có cú sốc"]].melt(
            id_vars="Năm",
            value_vars=["Y kế hoạch", "Y có cú sốc"],
            var_name="Kịch bản",
            value_name="Y",
        )

        fig_y = px.line(
            y_compare,
            x="Năm",
            y="Y",
            color="Kịch bản",
            markers=True,
            title="So sánh quỹ đạo GDP khi có cú sốc 2028",
            color_discrete_map={
                "Y kế hoạch": "#053151",
                "Y có cú sốc": "#E76F51",
            },
        )
        fig_y.update_traces(line=dict(width=4), marker=dict(size=8))
        fig_y.update_layout(xaxis_title="Năm", yaxis_title="Y")
        style_base_fig(fig_y, height=470)
        st.plotly_chart(fig_y, use_container_width=True)

        inv_compare = shock_table[["Năm", "Đầu tư kế hoạch", "Đầu tư có cú sốc"]].melt(
            id_vars="Năm",
            value_vars=["Đầu tư kế hoạch", "Đầu tư có cú sốc"],
            var_name="Kịch bản",
            value_name="Đầu tư",
        )

        fig_inv = px.bar(
            inv_compare,
            x="Năm",
            y="Đầu tư",
            color="Kịch bản",
            barmode="group",
            title="Điều chỉnh tổng đầu tư khi có cú sốc",
            color_discrete_map={
                "Đầu tư kế hoạch": "#053151",
                "Đầu tư có cú sốc": "#E76F51",
            },
        )
        fig_inv.update_traces(marker_line_color="white", marker_line_width=1)
        fig_inv.update_layout(xaxis_title="Năm", yaxis_title="Nghìn tỷ VND")
        style_base_fig(fig_inv, height=470)
        st.plotly_chart(fig_inv, use_container_width=True)

        st.subheader("Điều chỉnh đầu tư sau năm 2028")
        adjust_df = investment_adjustment_after_shock(opt_res["sim"], shock_res["sim"])
        show_table(adjust_df, decimals=3)

        fig_adjust = px.bar(
            adjust_df,
            x="Hạng mục",
            y="Chênh lệch",
            text=adjust_df["Chênh lệch"].round(1),
            title="Chênh lệch đầu tư sau cú sốc theo hạng mục",
        )
        fig_adjust.update_traces(
            marker_color=BRAND,
            textposition="outside",
            textfont=dict(color=BRAND),
        )
        fig_adjust.update_layout(xaxis_title="Hạng mục", yaxis_title="Chênh lệch đầu tư")
        style_base_fig(fig_adjust, height=420)
        st.plotly_chart(fig_adjust, use_container_width=True)

        st.success(
            "Sau cú sốc 2028, mô hình tái phân bổ đầu tư để cân bằng giữa phục hồi sản lượng và duy trì tiêu dùng."
        )

    # =====================================================
    # 8.3.4
    # =====================================================
    with tabs[3]:


        strategy_df = strategy_compare_table(opt_res, even_res, front_res)
        best_strategy = strategy_df.iloc[0]["Chiến lược"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Chiến lược tốt nhất", best_strategy)
        c2.metric("Welfare cao nhất", f"{strategy_df.iloc[0]['Welfare tổng']:.3f}")
        c3.metric("Terminal Y cao nhất", f"{strategy_df['Terminal Y'].max():,.1f}")

        show_table(strategy_df, decimals=3)

        fig = px.bar(
            strategy_df,
            x="Chiến lược",
            y="Welfare tổng",
            color="Chiến lược",
            text=strategy_df["Welfare tổng"].round(3),
            title="So sánh welfare tổng giữa các chiến lược",
            color_discrete_map=MULTI_COLORS,
        )
        fig.update_traces(textposition="outside", marker_line_color="white", marker_line_width=1)
        fig.update_layout(xaxis_title="Chiến lược", yaxis_title="Welfare tổng")
        style_base_fig(fig, height=430)
        st.plotly_chart(fig, use_container_width=True)

        rows = []

        for name, res in [
            ("Tối ưu SLSQP", opt_res),
            ("Đầu tư trải đều", even_res),
            ("Đầu tư front-load", front_res),
        ]:
            temp = path_table(res["sim"])[["Năm", "Y", "C", "Tổng đầu tư"]].copy()
            temp["Chiến lược"] = name
            rows.append(temp)

        compare_paths = pd.concat(rows, ignore_index=True)

        y_strategy = compare_paths[["Năm", "Y", "Chiến lược"]]

        fig_y = px.line(
            y_strategy,
            x="Năm",
            y="Y",
            color="Chiến lược",
            markers=True,
            title="So sánh quỹ đạo GDP theo chiến lược",
            color_discrete_map=MULTI_COLORS,
        )
        fig_y.update_traces(line=dict(width=4), marker=dict(size=8))
        fig_y.update_layout(xaxis_title="Năm", yaxis_title="Y")
        style_base_fig(fig_y, height=470)
        st.plotly_chart(fig_y, use_container_width=True)

        inv_strategy = compare_paths[["Năm", "Tổng đầu tư", "Chiến lược"]]

        fig_inv_strategy = px.line(
            inv_strategy,
            x="Năm",
            y="Tổng đầu tư",
            color="Chiến lược",
            markers=True,
            title="So sánh quỹ đạo tổng đầu tư",
            color_discrete_map=MULTI_COLORS,
        )
        fig_inv_strategy.update_traces(line=dict(width=4), marker=dict(size=8))
        fig_inv_strategy.update_layout(xaxis_title="Năm", yaxis_title="Tổng đầu tư")
        style_base_fig(fig_inv_strategy, height=470)
        st.plotly_chart(fig_inv_strategy, use_container_width=True)

        st.success(
            f"Chiến lược có welfare tổng cao nhất là {best_strategy}. "
            "Front-load thường có lợi khi đầu tư sớm giúp tích lũy năng lực sản xuất cho nhiều năm sau."
        )

    # =====================================================
    # 8.4
    # =====================================================
    with tabs[4]:
      

        total_inv = path_opt["Tổng đầu tư"].values
        first3_share = total_inv[:3].sum() / total_inv.sum() * 100 if total_inv.sum() > 0 else np.nan

        ai_h_ratio = path_opt["I_AI"].sum() / path_opt["I_H"].sum() if path_opt["I_H"].sum() > 0 else np.nan

        st.markdown("### a) Quỹ đạo tối ưu front-loaded hay back-loaded?")

        st.success(
            f"Tỷ trọng đầu tư trong 3 năm đầu chiếm khoảng **{first3_share:.2f}%** tổng đầu tư giai đoạn 2026-2035."
        )

        st.markdown(
            """
            Nếu tỷ trọng đầu tư tập trung lớn ở các năm đầu, quỹ đạo có tính **front-loaded**.
            Điều này xuất hiện vì đầu tư sớm vào K, D, AI và H tạo ra năng lực sản xuất cho nhiều năm sau.
            Với hệ số chiết khấu ρ = 0,97, Chính phủ vẫn đánh giá cao lợi ích tương lai,
            nên mô hình có xu hướng chấp nhận giảm một phần tiêu dùng hiện tại để đổi lấy sản lượng cao hơn trong tương lai.
            """
        )

        st.markdown("### b) Tỷ lệ đầu tư AI/H theo thời gian có ổn định không?")

        st.success(
            f"Tỷ lệ tổng đầu tư AI/H trong nghiệm tối ưu khoảng **{ai_h_ratio:.3f}**."
        )

        st.markdown(
            """
            Tỷ lệ AI/H thường không hoàn toàn ổn định vì hai loại đầu tư có vai trò khác nhau.
            AI giúp tăng năng lực công nghệ trực tiếp, nhưng nhân lực số H quyết định khả năng hấp thụ và vận hành AI.
            Hàm ý chính sách là đào tạo nhân lực số nên đi trước một bước hoặc ít nhất song hành với đầu tư AI.
            """
        )

        st.markdown("### c) Nếu ρ = 0,90 thì kết quả thay đổi thế nào?")

        rho_df = rho_policy_table(opt_res, opt_short_res)
        show_table(rho_df, decimals=3)

        st.markdown(
            """
            Hệ số chiết khấu ρ = 0,97 thể hiện Chính phủ quan tâm nhiều đến dài hạn.
            Khi giảm xuống ρ = 0,90, mô hình coi lợi ích tương lai kém quan trọng hơn.
            Khi đó nghiệm tối ưu thường có xu hướng giữ tiêu dùng hiện tại cao hơn và giảm ưu tiên cho các khoản đầu tư dài hạn.

            Đây là một cách giải thích vì sao các chính phủ có chu kỳ chính trị ngắn thường dễ **dưới đầu tư vào R&D và công nghệ nền tảng**:
            lợi ích xuất hiện chậm, còn chi phí ngân sách xuất hiện ngay.
            """
        )

        st.info(
            "Kết luận: mô hình động cho thấy chính sách chuyển đổi số không chỉ là chọn mức đầu tư, "
            "mà còn là chọn thời điểm đầu tư. Đầu tư sớm có thể làm giảm tiêu dùng ngắn hạn nhưng tạo lợi ích tích lũy dài hạn."
        )


def run():
    render()

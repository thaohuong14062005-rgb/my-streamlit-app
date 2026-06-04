# bai08_dynamic_2026_2035.py
# Bài 8 — Tối ưu động phân bổ liên thời gian 2026-2035
# Module dùng được với streamlit_app.py có cơ chế gọi module.render()

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# =====================================================
# 1. THAM SỐ MẶC ĐỊNH
# =====================================================

YEARS = np.arange(2026, 2036)
T = len(YEARS)

ALPHA = 0.33
BETA = 0.42
GAMMA = 0.10
DELTA = 0.08
THETA = 0.07

# Điều kiện đầu kỳ 2026, lấy từ gợi ý đề bài
DEFAULT_INITIAL = {
    "K0": 27500.0,
    "L0": 53.4,
    "D0": 20.3,
    "AI0": 86.0,
    "H0": 30.0,
    "Y2025": 12847.6,
}


def calibrate_A0():
    K0 = DEFAULT_INITIAL["K0"]
    L0 = DEFAULT_INITIAL["L0"]
    D0 = DEFAULT_INITIAL["D0"]
    AI0 = DEFAULT_INITIAL["AI0"]
    H0 = DEFAULT_INITIAL["H0"]
    Y2025 = DEFAULT_INITIAL["Y2025"]

    A0 = Y2025 / (
        (K0 ** ALPHA)
        * (L0 ** BETA)
        * (D0 ** GAMMA)
        * (AI0 ** DELTA)
        * (H0 ** THETA)
    )

    return A0


# =====================================================
# 2. HÀM SẢN XUẤT VÀ MÔ PHỎNG ĐỘNG
# =====================================================

def production(A, K, L, D, AI, H):
    eps = 1e-9

    return (
        A
        * (max(K, eps) ** ALPHA)
        * (max(L, eps) ** BETA)
        * (max(D, eps) ** GAMMA)
        * (max(AI, eps) ** DELTA)
        * (max(H, eps) ** THETA)
    )


def utility_log(C):
    return np.log(np.maximum(C, 1e-6))


def simulate_policy(
    decision_vector,
    rho=0.96,
    g_A=0.012,
    g_L=0.006,
    inv_rate_max=0.28,
    shock_2028=False,
    terminal_weight=2.5,
):
    """
    decision_vector có kích thước T*4.
    Mỗi năm có 4 tỷ trọng đầu tư:
    u_K, u_D, u_AI, u_H.
    Tổng tỷ trọng đầu tư mỗi năm <= inv_rate_max.
    """

    u = np.array(decision_vector, dtype=float).reshape(T, 4)
    u = np.clip(u, 0.0, inv_rate_max)

    K = DEFAULT_INITIAL["K0"]
    L = DEFAULT_INITIAL["L0"]
    D = DEFAULT_INITIAL["D0"]
    AI = DEFAULT_INITIAL["AI0"]
    H = DEFAULT_INITIAL["H0"]
    A = calibrate_A0()

    rows = []
    welfare = 0.0

    for t, year in enumerate(YEARS):
        Y_plan = production(A, K, L, D, AI, H)

        shock_factor = 0.92 if (shock_2028 and year == 2028) else 1.0
        Y_effective = Y_plan * shock_factor

        inv_share_total = float(u[t].sum())

        if inv_share_total > inv_rate_max:
            u[t] = u[t] / inv_share_total * inv_rate_max
            inv_share_total = inv_rate_max

        I_K = u[t, 0] * Y_effective
        I_D = u[t, 1] * Y_effective
        I_AI = u[t, 2] * Y_effective
        I_H = u[t, 3] * Y_effective

        total_investment = I_K + I_D + I_AI + I_H
        C = max(Y_effective - total_investment, 1e-6)

        welfare += (rho ** t) * utility_log(C)

        rows.append({
            "Năm": year,
            "A": A,
            "K": K,
            "L": L,
            "D": D,
            "AI": AI,
            "H": H,
            "Y kế hoạch": Y_plan,
            "Y thực sau cú sốc": Y_effective,
            "C": C,
            "I_K": I_K,
            "I_D": I_D,
            "I_AI": I_AI,
            "I_H": I_H,
            "Tổng đầu tư": total_investment,
            "Tỷ lệ đầu tư": inv_share_total,
            "Cú sốc": "Có" if shock_factor < 1 else "Không"
        })

        # Phương trình động học
        # K có khấu hao, D/AI/H tích lũy chậm hơn theo hiệu suất đầu tư.
        K_next = (1 - 0.045) * K + I_K

        D_next = D + 0.0040 * I_D
        AI_next = AI + 0.0120 * I_AI
        H_next = H + 0.0030 * I_H

        # Chặn giá trị hợp lý để tránh mô phỏng phi thực tế.
        D_next = min(D_next, 55.0)
        AI_next = min(AI_next, 260.0)
        H_next = min(H_next, 70.0)

        L_next = L * (1 + g_L)
        A_next = A * (1 + g_A)

        K, D, AI, H, L, A = K_next, D_next, AI_next, H_next, L_next, A_next

    # Thêm terminal value để mô hình không chỉ ưu tiên tiêu dùng hiện tại.
    terminal_Y = production(A, K, L, D, AI, H)
    welfare += terminal_weight * (rho ** T) * utility_log(terminal_Y)

    traj = pd.DataFrame(rows)

    terminal_state = {
        "K_2036": K,
        "D_2036": D,
        "AI_2036": AI,
        "H_2036": H,
        "Y_terminal": terminal_Y,
        "Welfare": welfare,
    }

    return traj, terminal_state


# =====================================================
# 3. TỐI ƯU BẰNG SCIPY SLSQP
# =====================================================

def initial_guess(inv_rate_max=0.28):
    """
    Điểm khởi tạo cân bằng:
    Tổng đầu tư khoảng 24% GDP, chia cho K, D, AI, H.
    """
    base = np.array([0.40, 0.25, 0.15, 0.20])
    total_rate = min(0.24, inv_rate_max * 0.90)
    shares = base * total_rate
    return np.tile(shares, T)


def objective_to_minimize(
    z,
    rho,
    g_A,
    g_L,
    inv_rate_max,
    shock_2028,
    terminal_weight,
):
    _, terminal = simulate_policy(
        z,
        rho=rho,
        g_A=g_A,
        g_L=g_L,
        inv_rate_max=inv_rate_max,
        shock_2028=shock_2028,
        terminal_weight=terminal_weight,
    )

    return -terminal["Welfare"]


def constraint_values(
    z,
    rho,
    g_A,
    g_L,
    inv_rate_max,
    min_D_2035,
    min_H_2035,
    min_AI_2035,
    shock_2028,
    terminal_weight,
):
    u = np.array(z).reshape(T, 4)

    # Mỗi năm tổng tỷ trọng đầu tư <= inv_rate_max
    annual_constraints = inv_rate_max - u.sum(axis=1)

    traj, terminal = simulate_policy(
        z,
        rho=rho,
        g_A=g_A,
        g_L=g_L,
        inv_rate_max=inv_rate_max,
        shock_2028=shock_2028,
        terminal_weight=terminal_weight,
    )

    # Lấy giá trị cuối năm 2035 từ dòng cuối cùng.
    last = traj.iloc[-1]

    d_constraint = last["D"] - min_D_2035
    h_constraint = last["H"] - min_H_2035
    ai_constraint = last["AI"] - min_AI_2035

    return np.concatenate([
        annual_constraints,
        np.array([d_constraint, h_constraint, ai_constraint])
    ])


def solve_dynamic_problem(
    rho=0.96,
    g_A=0.012,
    g_L=0.006,
    inv_rate_max=0.28,
    min_D_2035=30.0,
    min_H_2035=40.0,
    min_AI_2035=120.0,
    shock_2028=False,
    terminal_weight=2.5,
):
    if not SCIPY_AVAILABLE:
        z0 = initial_guess(inv_rate_max)
        traj, terminal = simulate_policy(
            z0,
            rho=rho,
            g_A=g_A,
            g_L=g_L,
            inv_rate_max=inv_rate_max,
            shock_2028=shock_2028,
            terminal_weight=terminal_weight,
        )

        return {
            "success": False,
            "message": "SciPy chưa được cài đặt. App dùng nghiệm khởi tạo cân bằng làm fallback.",
            "z": z0,
            "traj": traj,
            "terminal": terminal,
            "method": "Fallback"
        }

    z0 = initial_guess(inv_rate_max)

    bounds = [(0.0, inv_rate_max) for _ in range(T * 4)]

    cons = {
        "type": "ineq",
        "fun": lambda z: constraint_values(
            z,
            rho=rho,
            g_A=g_A,
            g_L=g_L,
            inv_rate_max=inv_rate_max,
            min_D_2035=min_D_2035,
            min_H_2035=min_H_2035,
            min_AI_2035=min_AI_2035,
            shock_2028=shock_2028,
            terminal_weight=terminal_weight,
        )
    }

    result = minimize(
        objective_to_minimize,
        z0,
        args=(rho, g_A, g_L, inv_rate_max, shock_2028, terminal_weight),
        method="SLSQP",
        bounds=bounds,
        constraints=[cons],
        options={
            "maxiter": 600,
            "ftol": 1e-8,
            "disp": False
        }
    )

    if result.success:
        z = result.x
        method = "SLSQP optimal"
    else:
        # Dùng nghiệm tốt nhất solver trả về, nếu không tốt thì dùng z0.
        z = result.x if result.x is not None else z0
        method = "SLSQP returned best effort"

    traj, terminal = simulate_policy(
        z,
        rho=rho,
        g_A=g_A,
        g_L=g_L,
        inv_rate_max=inv_rate_max,
        shock_2028=shock_2028,
        terminal_weight=terminal_weight,
    )

    return {
        "success": bool(result.success),
        "message": result.message,
        "z": z,
        "traj": traj,
        "terminal": terminal,
        "method": method,
        "raw": result,
    }


# =====================================================
# 4. CHIẾN LƯỢC CỐ ĐỊNH: TRẢI ĐỀU VÀ FRONT-LOAD
# =====================================================

def fixed_strategy_vector(strategy="even", inv_rate_max=0.28):
    base_even = np.array([0.40, 0.25, 0.15, 0.20])

    z = []

    for t, year in enumerate(YEARS):
        if strategy == "even":
            total_rate = min(0.24, inv_rate_max * 0.90)
            mix = base_even

        elif strategy == "front_load":
            if t <= 2:
                total_rate = min(0.32, inv_rate_max)
                mix = np.array([0.38, 0.27, 0.20, 0.15])
            elif t <= 5:
                total_rate = min(0.24, inv_rate_max)
                mix = np.array([0.35, 0.25, 0.20, 0.20])
            else:
                total_rate = min(0.17, inv_rate_max)
                mix = np.array([0.30, 0.22, 0.18, 0.30])

        elif strategy == "human_first":
            total_rate = min(0.24, inv_rate_max * 0.90)
            mix = np.array([0.25, 0.20, 0.15, 0.40])

        else:
            total_rate = min(0.24, inv_rate_max * 0.90)
            mix = base_even

        z.extend(list(mix * total_rate))

    return np.array(z)


def simulate_fixed_strategy(
    strategy,
    rho=0.96,
    g_A=0.012,
    g_L=0.006,
    inv_rate_max=0.28,
    shock_2028=False,
    terminal_weight=2.5,
):
    z = fixed_strategy_vector(strategy=strategy, inv_rate_max=inv_rate_max)

    traj, terminal = simulate_policy(
        z,
        rho=rho,
        g_A=g_A,
        g_L=g_L,
        inv_rate_max=inv_rate_max,
        shock_2028=shock_2028,
        terminal_weight=terminal_weight,
    )

    return {
        "z": z,
        "traj": traj,
        "terminal": terminal,
        "strategy": strategy
    }


# =====================================================
# 5. HÀM HIỂN THỊ BIỂU ĐỒ
# =====================================================

def plot_trajectory(df, title):
    fig = px.line(
        df,
        x="Năm",
        y=["K", "D", "AI", "H"],
        markers=True,
        title=title
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_y_c(df, title):
    fig = px.line(
        df,
        x="Năm",
        y=["Y kế hoạch", "Y thực sau cú sốc", "C"],
        markers=True,
        title=title
    )
    st.plotly_chart(fig, use_container_width=True)


def allocation_long(df):
    cols = ["I_K", "I_D", "I_AI", "I_H"]

    long_df = df[["Năm"] + cols].melt(
        id_vars="Năm",
        value_vars=cols,
        var_name="Hạng mục",
        value_name="Đầu tư"
    )

    mapping = {
        "I_K": "Đầu tư vật chất K",
        "I_D": "Hạ tầng/chuyển đổi số D",
        "I_AI": "Năng lực AI",
        "I_H": "Nhân lực số H",
    }

    long_df["Hạng mục"] = long_df["Hạng mục"].map(mapping)

    return long_df


# =====================================================
# 6. GIAO DIỆN STREAMLIT
# =====================================================

def render():
    st.markdown(
        """
        <div class="card">
            <h1>⏳ Bài 8 — Tối ưu động phân bổ liên thời gian 2026–2035</h1>
            <p>
            Module này xây dựng mô hình tối ưu động phi tuyến cho chiến lược phân bổ đầu tư vào
            vốn vật chất K, số hóa D, AI và nhân lực số H trong giai đoạn 2026–2035.
            Mô hình dùng scipy.optimize SLSQP để tối đa hóa tổng phúc lợi liên thời gian.
            </p>
            <span class="pill">Dynamic Optimization</span>
            <span class="pill">SLSQP</span>
            <span class="pill">Cobb-Douglas</span>
            <span class="pill">2026–2035</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.sidebar:
        st.markdown("### ⚙️ Tham số Bài 8")

        rho = st.number_input(
            "ρ - Hệ số chiết khấu",
            min_value=0.80,
            max_value=0.999,
            value=0.96,
            step=0.01,
            format="%.3f"
        )

        g_A = st.number_input(
            "Tăng trưởng TFP A mỗi năm",
            min_value=0.000,
            max_value=0.050,
            value=0.012,
            step=0.001,
            format="%.3f"
        )

        g_L = st.number_input(
            "Tăng trưởng lao động L mỗi năm",
            min_value=0.000,
            max_value=0.030,
            value=0.006,
            step=0.001,
            format="%.3f"
        )

        inv_rate_max = st.number_input(
            "Trần tỷ lệ đầu tư/GDP mỗi năm",
            min_value=0.10,
            max_value=0.50,
            value=0.28,
            step=0.01,
            format="%.2f"
        )

        min_D_2035 = st.number_input(
            "Ràng buộc D tối thiểu cuối kỳ",
            min_value=20.0,
            max_value=55.0,
            value=30.0,
            step=1.0
        )

        min_AI_2035 = st.number_input(
            "Ràng buộc AI tối thiểu cuối kỳ",
            min_value=80.0,
            max_value=260.0,
            value=120.0,
            step=5.0
        )

        min_H_2035 = st.number_input(
            "Ràng buộc H tối thiểu cuối kỳ",
            min_value=30.0,
            max_value=70.0,
            value=40.0,
            step=1.0
        )

        terminal_weight = st.number_input(
            "Trọng số giá trị cuối kỳ",
            min_value=0.0,
            max_value=10.0,
            value=2.5,
            step=0.5
        )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📌 Mô hình",
        "8.3.1 SLSQP",
        "8.3.2 Quỹ đạo tối ưu",
        "8.3.3 Cú sốc 2028",
        "8.3.4 So sánh chiến lược"
    ])

    # =====================================================
    # TAB 1 — MÔ HÌNH
    # =====================================================

    with tab1:
        st.header("1. Mô hình tối ưu động")

        st.markdown("### Hàm mục tiêu phúc lợi liên thời gian")

        st.latex(r"\max \sum_{t=2026}^{2035} \rho^{t-2026} U(C_t)")

        st.markdown("Trong module này sử dụng hàm thỏa dụng log:")

        st.latex(r"U(C_t) = \ln(C_t)")

        st.markdown("### Hàm sản xuất Cobb-Douglas mở rộng")

        st.latex(
            r"Y_t = A_t K_t^{0.33} L_t^{0.42} D_t^{0.10} AI_t^{0.08} H_t^{0.07}"
        )

        st.markdown("### Phương trình động học")

        st.latex(r"K_{t+1} = (1-\delta_K)K_t + I^K_t")
        st.latex(r"D_{t+1} = D_t + \eta_D I^D_t")
        st.latex(r"AI_{t+1} = AI_t + \eta_{AI} I^{AI}_t")
        st.latex(r"H_{t+1} = H_t + \eta_H I^H_t")

        st.markdown("### Điều kiện đầu kỳ")

        initial_df = pd.DataFrame({
            "Biến": ["K0", "L0", "D0", "AI0", "H0", "Y2025", "A0 hiệu chỉnh"],
            "Giá trị": [
                DEFAULT_INITIAL["K0"],
                DEFAULT_INITIAL["L0"],
                DEFAULT_INITIAL["D0"],
                DEFAULT_INITIAL["AI0"],
                DEFAULT_INITIAL["H0"],
                DEFAULT_INITIAL["Y2025"],
                calibrate_A0(),
            ]
        })

        st.dataframe(initial_df.round(4), use_container_width=True)

        st.info(
            "Mô hình dùng tỷ trọng đầu tư theo GDP làm biến quyết định. "
            "Mỗi năm, tổng tỷ trọng đầu tư vào K, D, AI và H không vượt quá trần đầu tư/GDP do người dùng chọn ở sidebar."
        )

    # =====================================================
    # TAB 2 — SLSQP
    # =====================================================

    with tab2:
        st.header("Câu 8.3.1 — Giải bài toán bằng scipy.optimize.minimize SLSQP")

        result = solve_dynamic_problem(
            rho=rho,
            g_A=g_A,
            g_L=g_L,
            inv_rate_max=inv_rate_max,
            min_D_2035=min_D_2035,
            min_H_2035=min_H_2035,
            min_AI_2035=min_AI_2035,
            shock_2028=False,
            terminal_weight=terminal_weight,
        )

        traj = result["traj"]
        terminal = result["terminal"]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Trạng thái", "Optimal" if result["success"] else "Best effort")
        c2.metric("Welfare", f"{terminal['Welfare']:,.4f}")
        c3.metric("Y terminal", f"{terminal['Y_terminal']:,.2f}")
        c4.metric("Phương pháp", result["method"])

        if not result["success"]:
            st.warning(
                f"SLSQP chưa hội tụ hoàn toàn: {result['message']}. "
                "App vẫn hiển thị nghiệm tốt nhất solver trả về để phân tích."
            )

        st.markdown("### Bảng quỹ đạo tối ưu")
        st.dataframe(traj.round(4), use_container_width=True)

        st.markdown("### Cơ cấu đầu tư tối ưu theo năm")

        long_inv = allocation_long(traj)

        fig_inv = px.bar(
            long_inv,
            x="Năm",
            y="Đầu tư",
            color="Hạng mục",
            barmode="stack",
            title="Phân bổ đầu tư tối ưu theo năm"
        )
        st.plotly_chart(fig_inv, use_container_width=True)

    # =====================================================
    # TAB 3 — QUỸ ĐẠO
    # =====================================================

    with tab3:
        st.header("Câu 8.3.2 — Quỹ đạo tối ưu của K, D, AI, H, Y, C")

        result = solve_dynamic_problem(
            rho=rho,
            g_A=g_A,
            g_L=g_L,
            inv_rate_max=inv_rate_max,
            min_D_2035=min_D_2035,
            min_H_2035=min_H_2035,
            min_AI_2035=min_AI_2035,
            shock_2028=False,
            terminal_weight=terminal_weight,
        )

        traj = result["traj"]

        plot_trajectory(
            traj,
            "Quỹ đạo trạng thái tối ưu: K, D, AI, H"
        )

        plot_y_c(
            traj,
            "Quỹ đạo sản lượng Y và tiêu dùng C"
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("D 2035", f"{traj['D'].iloc[-1]:.2f}")
        c2.metric("AI 2035", f"{traj['AI'].iloc[-1]:.2f}")
        c3.metric("H 2035", f"{traj['H'].iloc[-1]:.2f}")
        c4.metric("C 2035", f"{traj['C'].iloc[-1]:,.2f}")

        st.success(
            "Quỹ đạo tối ưu cho thấy mô hình không chỉ chọn đầu tư trong một năm, "
            "mà thiết kế đường đi liên thời gian: đầu tư hôm nay làm thay đổi K, D, AI, H và do đó ảnh hưởng đến Y, C các năm sau."
        )

    # =====================================================
    # TAB 4 — CÚ SỐC 2028
    # =====================================================

    with tab4:
        st.header("Câu 8.3.3 — Cú sốc năm 2028: Y giảm 8% so với kế hoạch")

        base = solve_dynamic_problem(
            rho=rho,
            g_A=g_A,
            g_L=g_L,
            inv_rate_max=inv_rate_max,
            min_D_2035=min_D_2035,
            min_H_2035=min_H_2035,
            min_AI_2035=min_AI_2035,
            shock_2028=False,
            terminal_weight=terminal_weight,
        )

        shock = solve_dynamic_problem(
            rho=rho,
            g_A=g_A,
            g_L=g_L,
            inv_rate_max=inv_rate_max,
            min_D_2035=min_D_2035,
            min_H_2035=min_H_2035,
            min_AI_2035=min_AI_2035,
            shock_2028=True,
            terminal_weight=terminal_weight,
        )

        base_traj = base["traj"].copy()
        shock_traj = shock["traj"].copy()

        base_traj["Kịch bản"] = "Không cú sốc"
        shock_traj["Kịch bản"] = "Cú sốc 2028"

        both = pd.concat([base_traj, shock_traj], ignore_index=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Welfare không sốc", f"{base['terminal']['Welfare']:,.4f}")
        c2.metric("Welfare cú sốc", f"{shock['terminal']['Welfare']:,.4f}")
        c3.metric(
            "Chênh lệch welfare",
            f"{shock['terminal']['Welfare'] - base['terminal']['Welfare']:,.4f}"
        )

        fig_y = px.line(
            both,
            x="Năm",
            y="Y thực sau cú sốc",
            color="Kịch bản",
            markers=True,
            title="So sánh sản lượng thực tế giữa hai kịch bản"
        )
        st.plotly_chart(fig_y, use_container_width=True)

        fig_c = px.line(
            both,
            x="Năm",
            y="C",
            color="Kịch bản",
            markers=True,
            title="So sánh tiêu dùng C giữa hai kịch bản"
        )
        st.plotly_chart(fig_c, use_container_width=True)

        inv_compare = both[[
            "Năm", "Kịch bản", "I_K", "I_D", "I_AI", "I_H"
        ]].melt(
            id_vars=["Năm", "Kịch bản"],
            value_vars=["I_K", "I_D", "I_AI", "I_H"],
            var_name="Hạng mục",
            value_name="Đầu tư"
        )

        fig_inv = px.line(
            inv_compare,
            x="Năm",
            y="Đầu tư",
            color="Hạng mục",
            line_dash="Kịch bản",
            markers=True,
            title="Điều chỉnh phân bổ đầu tư sau cú sốc 2028"
        )
        st.plotly_chart(fig_inv, use_container_width=True)

        st.markdown("### Nhận xét")

        st.warning(
            "Cú sốc 2028 làm giảm sản lượng và tiêu dùng trong năm bị sốc, đồng thời làm giảm quy mô nguồn lực khả dụng cho đầu tư. "
            "Trong các năm sau, mô hình thường có xu hướng tái phân bổ để bảo vệ các biến tích lũy dài hạn, đặc biệt là K, AI hoặc H, "
            "tùy theo ràng buộc cuối kỳ đang đặt ở sidebar."
        )

    # =====================================================
    # TAB 5 — SO SÁNH CHIẾN LƯỢC
    # =====================================================

    with tab5:
        st.header("Câu 8.3.4 — So sánh đầu tư trải đều và front-load")

        even = simulate_fixed_strategy(
            "even",
            rho=rho,
            g_A=g_A,
            g_L=g_L,
            inv_rate_max=inv_rate_max,
            shock_2028=False,
            terminal_weight=terminal_weight,
        )

        front = simulate_fixed_strategy(
            "front_load",
            rho=rho,
            g_A=g_A,
            g_L=g_L,
            inv_rate_max=inv_rate_max,
            shock_2028=False,
            terminal_weight=terminal_weight,
        )

        human = simulate_fixed_strategy(
            "human_first",
            rho=rho,
            g_A=g_A,
            g_L=g_L,
            inv_rate_max=inv_rate_max,
            shock_2028=False,
            terminal_weight=terminal_weight,
        )

        rows = []

        for label, res in [
            ("Đầu tư trải đều", even),
            ("Front-load", front),
            ("Ưu tiên nhân lực số", human)
        ]:
            traj = res["traj"]
            terminal = res["terminal"]

            rows.append({
                "Chiến lược": label,
                "Welfare": terminal["Welfare"],
                "Y terminal": terminal["Y_terminal"],
                "Y 2035": traj["Y kế hoạch"].iloc[-1],
                "C 2035": traj["C"].iloc[-1],
                "D 2035": traj["D"].iloc[-1],
                "AI 2035": traj["AI"].iloc[-1],
                "H 2035": traj["H"].iloc[-1],
                "Tổng đầu tư 2026-2035": traj["Tổng đầu tư"].sum(),
            })

        compare_df = pd.DataFrame(rows).sort_values("Welfare", ascending=False)

        st.dataframe(compare_df.round(4), use_container_width=True)

        c1, c2 = st.columns(2)
        c1.metric("Chiến lược welfare cao nhất", compare_df.iloc[0]["Chiến lược"])
        c2.metric("Welfare cao nhất", f"{compare_df.iloc[0]['Welfare']:,.4f}")

        strategy_trajs = []

        for label, res in [
            ("Trải đều", even),
            ("Front-load", front),
            ("Nhân lực số", human)
        ]:
            temp = res["traj"].copy()
            temp["Chiến lược"] = label
            strategy_trajs.append(temp)

        all_strategy = pd.concat(strategy_trajs, ignore_index=True)

        fig_y = px.line(
            all_strategy,
            x="Năm",
            y="Y kế hoạch",
            color="Chiến lược",
            markers=True,
            title="So sánh quỹ đạo Y giữa các chiến lược"
        )
        st.plotly_chart(fig_y, use_container_width=True)

        fig_c = px.line(
            all_strategy,
            x="Năm",
            y="C",
            color="Chiến lược",
            markers=True,
            title="So sánh quỹ đạo tiêu dùng C giữa các chiến lược"
        )
        st.plotly_chart(fig_c, use_container_width=True)

        st.markdown("### Giải thích kết quả")

        if compare_df.iloc[0]["Chiến lược"] == "Front-load":
            st.success(
                "Front-load có welfare cao hơn vì đầu tư mạnh ở 3 năm đầu giúp các trạng thái K, D, AI, H tăng sớm hơn, "
                "từ đó tạo hiệu ứng tích lũy cho các năm sau. Đây là logic quan trọng của tối ưu động."
            )
        elif compare_df.iloc[0]["Chiến lược"] == "Đầu tư trải đều":
            st.success(
                "Đầu tư trải đều có welfare cao hơn trong cấu hình hiện tại vì chiến lược này tránh hy sinh tiêu dùng quá mạnh ở giai đoạn đầu, "
                "đồng thời vẫn bảo đảm tích lũy dần các trạng thái K, D, AI, H."
            )
        else:
            st.success(
                "Chiến lược ưu tiên nhân lực số có welfare cao hơn trong cấu hình hiện tại, cho thấy H có vai trò quan trọng trong hấp thụ công nghệ, "
                "giảm rủi ro đầu tư AI quá sớm khi nền tảng nhân lực chưa đủ mạnh."
            )


def run():
    render()

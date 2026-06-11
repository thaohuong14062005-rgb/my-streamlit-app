# bai10_stochastic_sp.py
# Bài 10 - Quy hoạch ngẫu nhiên hai giai đoạn dưới bất định
# Module Streamlit, dùng hàm render()

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import linprog


ITEMS = ["I", "D", "AI", "H"]

ITEM_LABELS = {
    "I": "Hạ tầng số",
    "D": "Chuyển đổi số",
    "AI": "Trí tuệ nhân tạo",
    "H": "Nhân lực số",
}

SCENARIOS = ["s1", "s2", "s3", "s4"]

SCENARIO_LABELS = {
    "s1": "Lạc quan",
    "s2": "Cơ sở",
    "s3": "Bi quan",
    "s4": "Khủng hoảng",
}


def get_scenario_data():
    return pd.DataFrame({
        "Kịch bản": SCENARIOS,
        "Tên kịch bản": ["Lạc quan", "Cơ sở", "Bi quan", "Khủng hoảng"],
        "Tăng trưởng TG (%)": [3.5, 2.8, 1.5, 0.2],
        "FDI VN (tỷ USD/năm)": [32.0, 27.0, 20.0, 12.0],
        "Xuất khẩu VN tăng (%)": [12.0, 8.0, 3.0, -5.0],
        "Xác suất": [0.30, 0.45, 0.20, 0.05],
    })


def get_beta_data():
    beta_base = pd.DataFrame({
        "Hạng mục": ITEMS,
        "Tên hạng mục": [ITEM_LABELS[i] for i in ITEMS],
        "Beta cơ bản": [1.00, 1.10, 1.25, 0.95],
    })

    beta_s = pd.DataFrame({
        "Kịch bản": SCENARIOS,
        "I": [1.25, 1.00, 0.75, 0.40],
        "D": [1.35, 1.10, 0.85, 0.50],
        "AI": [1.55, 1.25, 0.90, 0.55],
        "H": [1.05, 0.95, 1.00, 1.10],
    })

    return beta_base, beta_s


def solve_stochastic_model(
    first_budget=65000.0,
    reserve_budget=15000.0,
    ai_capacity_ratio=0.5,
    min_h_share=0.0,
    max_item_share=1.0,
    probabilities=None,
    beta_base=None,
    beta_second=None,
    fixed_x=None,
):
    """
    Giải bài toán two-stage stochastic programming bằng scipy.optimize.linprog.

    Biến:
        x_j      : quyết định first-stage, j thuộc {I, D, AI, H}
        y_sj     : quyết định recourse/second-stage theo kịch bản s

    Hàm mục tiêu:
        max sum_j beta_j*x_j + sum_s p_s * sum_j beta_sj*y_sj

    Ràng buộc:
        sum_j x_j <= first_budget
        sum_j y_sj <= reserve_budget, với mọi s
        y_s,AI <= ai_capacity_ratio * x_H, với mọi s
        x_j, y_sj >= 0
    """
    scenario_df = get_scenario_data()
    beta_base_df, beta_s_df = get_beta_data()

    if probabilities is None:
        probabilities = scenario_df["Xác suất"].to_numpy(dtype=float)

    if beta_base is None:
        beta_base = beta_base_df["Beta cơ bản"].to_numpy(dtype=float)

    if beta_second is None:
        beta_second = beta_s_df[ITEMS].to_numpy(dtype=float)

    probabilities = np.array(probabilities, dtype=float)
    beta_base = np.array(beta_base, dtype=float)
    beta_second = np.array(beta_second, dtype=float)

    n_items = len(ITEMS)
    n_scenarios = len(probabilities)

    # Vector biến: [x_I, x_D, x_AI, x_H, y_s1_I, y_s1_D, ..., y_s4_H]
    n_vars = n_items + n_scenarios * n_items

    # scipy linprog là minimize, nên đổi dấu hàm mục tiêu
    c = np.zeros(n_vars)
    c[:n_items] = -beta_base

    for s in range(n_scenarios):
        start = n_items + s * n_items
        end = start + n_items
        c[start:end] = -probabilities[s] * beta_second[s]

    A_ub = []
    b_ub = []

    # Ràng buộc ngân sách first-stage
    row = np.zeros(n_vars)
    row[:n_items] = 1.0
    A_ub.append(row)
    b_ub.append(first_budget)

    # Ràng buộc sàn tỷ trọng nhân lực số H trong first-stage
    if min_h_share > 0:
        row = np.zeros(n_vars)
        row[ITEMS.index("H")] = -1.0
        A_ub.append(row)
        b_ub.append(-min_h_share * first_budget)

    # Ràng buộc trần tỷ trọng mỗi hạng mục, tránh nghiệm dồn 100%
    if max_item_share < 1.0:
        for j in range(n_items):
            row = np.zeros(n_vars)
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(max_item_share * first_budget)

    # Ràng buộc second-stage cho từng kịch bản
    for s in range(n_scenarios):
        start = n_items + s * n_items
        end = start + n_items

        # Tổng ngân sách recourse theo kịch bản
        row = np.zeros(n_vars)
        row[start:end] = 1.0
        A_ub.append(row)
        b_ub.append(reserve_budget)

        # y_AI^s <= ai_capacity_ratio * x_H
        row = np.zeros(n_vars)
        row[start + ITEMS.index("AI")] = 1.0
        row[ITEMS.index("H")] = -ai_capacity_ratio
        A_ub.append(row)
        b_ub.append(0.0)

    bounds = [(0.0, None)] * n_vars

    # Nếu muốn đánh giá một nghiệm x cố định, dùng fixed_x
    if fixed_x is not None:
        fixed_x = np.array(fixed_x, dtype=float)
        for j in range(n_items):
            bounds[j] = (fixed_x[j], fixed_x[j])

    res = linprog(
        c=c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        bounds=bounds,
        method="highs",
    )

    info = {
        "probabilities": probabilities,
        "beta_base": beta_base,
        "beta_second": beta_second,
        "first_budget": first_budget,
        "reserve_budget": reserve_budget,
        "ai_capacity_ratio": ai_capacity_ratio,
    }

    return res, info


def unpack_solution(res, info, scenario_names=None):
    if not res.success:
        return pd.DataFrame(), pd.DataFrame(), {}

    probabilities = info["probabilities"]
    beta_base = info["beta_base"]
    beta_second = info["beta_second"]

    n_items = len(ITEMS)
    n_scenarios = len(probabilities)

    if scenario_names is None:
        scenario_names = SCENARIOS[:n_scenarios]

    x = res.x[:n_items]
    y = res.x[n_items:].reshape(n_scenarios, n_items)

    first_df = pd.DataFrame({
        "Hạng mục": ITEMS,
        "Tên hạng mục": [ITEM_LABELS[i] for i in ITEMS],
        "x first-stage": x,
        "Beta cơ bản": beta_base,
        "Lợi ích first-stage": x * beta_base,
        "Tỷ trọng (%)": np.where(x.sum() > 0, x / x.sum() * 100, 0),
    })

    second_rows = []
    for s in range(n_scenarios):
        for j, item in enumerate(ITEMS):
            second_rows.append({
                "Kịch bản": scenario_names[s],
                "Tên kịch bản": SCENARIO_LABELS.get(scenario_names[s], scenario_names[s]),
                "Hạng mục": item,
                "Tên hạng mục": ITEM_LABELS[item],
                "y recourse": y[s, j],
                "Beta kịch bản": beta_second[s, j],
                "Xác suất": probabilities[s],
                "Lợi ích nếu kịch bản xảy ra": beta_second[s, j] * y[s, j],
                "Lợi ích kỳ vọng": probabilities[s] * beta_second[s, j] * y[s, j],
            })

    second_df = pd.DataFrame(second_rows)

    summary = {
        "objective": -res.fun,
        "first_budget_used": x.sum(),
        "first_benefit": np.sum(beta_base * x),
        "expected_recourse_benefit": sum(
            probabilities[s] * np.sum(beta_second[s] * y[s])
            for s in range(n_scenarios)
        ),
        "x": x,
        "y": y,
    }

    return first_df, second_df, summary


def solve_expected_value_model(
    first_budget,
    reserve_budget,
    ai_capacity_ratio,
    min_h_share,
    max_item_share,
):
    """
    Giải mô hình EV: thay beta kịch bản bằng beta kỳ vọng.
    """
    scenario_df = get_scenario_data()
    beta_base_df, beta_s_df = get_beta_data()

    p = scenario_df["Xác suất"].to_numpy(dtype=float)
    beta_base = beta_base_df["Beta cơ bản"].to_numpy(dtype=float)
    beta_s = beta_s_df[ITEMS].to_numpy(dtype=float)

    beta_expected = p @ beta_s

    res_ev, info_ev = solve_stochastic_model(
        first_budget=first_budget,
        reserve_budget=reserve_budget,
        ai_capacity_ratio=ai_capacity_ratio,
        min_h_share=min_h_share,
        max_item_share=max_item_share,
        probabilities=np.array([1.0]),
        beta_base=beta_base,
        beta_second=beta_expected.reshape(1, len(ITEMS)),
    )

    return res_ev, info_ev, beta_expected


def evaluate_fixed_x(
    fixed_x,
    first_budget,
    reserve_budget,
    ai_capacity_ratio,
    min_h_share,
    max_item_share,
):
    """
    Đánh giá nghiệm x_EV dưới cây kịch bản thật để tính EEV.
    """
    res, info = solve_stochastic_model(
        first_budget=first_budget,
        reserve_budget=reserve_budget,
        ai_capacity_ratio=ai_capacity_ratio,
        min_h_share=min_h_share,
        max_item_share=max_item_share,
        fixed_x=fixed_x,
    )
    return res, info


def solve_wait_and_see_one_scenario(
    scenario_index,
    first_budget,
    reserve_budget,
    ai_capacity_ratio,
    min_h_share,
    max_item_share,
):
    """
    Giải bài toán deterministic khi biết trước kịch bản.
    Dùng beta của kịch bản cho cả first-stage và second-stage.
    """
    beta_base_df, beta_s_df = get_beta_data()
    beta_s = beta_s_df[ITEMS].to_numpy(dtype=float)
    beta = beta_s[scenario_index]

    res, info = solve_stochastic_model(
        first_budget=first_budget,
        reserve_budget=reserve_budget,
        ai_capacity_ratio=ai_capacity_ratio,
        min_h_share=min_h_share,
        max_item_share=max_item_share,
        probabilities=np.array([1.0]),
        beta_base=beta,
        beta_second=beta.reshape(1, len(ITEMS)),
    )

    return res, info


def compute_vss_evpi(
    sp_value,
    first_budget,
    reserve_budget,
    ai_capacity_ratio,
    min_h_share,
    max_item_share,
):
    """
    Tính:
        EEV = giá trị kỳ vọng khi dùng nghiệm EV
        VSS = SP - EEV
        WS = giá trị wait-and-see kỳ vọng
        EVPI = WS - SP
    """
    scenario_df = get_scenario_data()
    p = scenario_df["Xác suất"].to_numpy(dtype=float)

    # EV model
    res_ev, info_ev, beta_expected = solve_expected_value_model(
        first_budget=first_budget,
        reserve_budget=reserve_budget,
        ai_capacity_ratio=ai_capacity_ratio,
        min_h_share=min_h_share,
        max_item_share=max_item_share,
    )

    if not res_ev.success:
        return None

    x_ev = res_ev.x[:len(ITEMS)]

    # EEV: dùng x_EV trong cây kịch bản thật
    res_eev, info_eev = evaluate_fixed_x(
        fixed_x=x_ev,
        first_budget=first_budget,
        reserve_budget=reserve_budget,
        ai_capacity_ratio=ai_capacity_ratio,
        min_h_share=min_h_share,
        max_item_share=max_item_share,
    )

    if not res_eev.success:
        return None

    eev_value = -res_eev.fun
    vss = sp_value - eev_value

    # Wait-and-see theo từng kịch bản
    ws_rows = []
    ws_values = []

    for s_idx, s in enumerate(SCENARIOS):
        res_ws, info_ws = solve_wait_and_see_one_scenario(
            scenario_index=s_idx,
            first_budget=first_budget,
            reserve_budget=reserve_budget,
            ai_capacity_ratio=ai_capacity_ratio,
            min_h_share=min_h_share,
            max_item_share=max_item_share,
        )

        if res_ws.success:
            first_df, second_df, summary = unpack_solution(
                res_ws,
                info_ws,
                scenario_names=[s],
            )

            x = summary["x"]
            y = summary["y"][0]
            value = summary["objective"]

            row = {
                "Kịch bản": s,
                "Tên kịch bản": SCENARIO_LABELS[s],
                "WS_s": value,
            }

            for j, item in enumerate(ITEMS):
                row[f"x_{item}"] = x[j]
                row[f"y_{item}"] = y[j]

            ws_rows.append(row)
            ws_values.append(value)
        else:
            ws_rows.append({
                "Kịch bản": s,
                "Tên kịch bản": SCENARIO_LABELS[s],
                "WS_s": np.nan,
            })
            ws_values.append(np.nan)

    ws_df = pd.DataFrame(ws_rows)
    ws_values = np.array(ws_values, dtype=float)
    ws_expected = np.nansum(p * ws_values)
    evpi = ws_expected - sp_value

    result = {
        "res_ev": res_ev,
        "info_ev": info_ev,
        "res_eev": res_eev,
        "info_eev": info_eev,
        "x_ev": x_ev,
        "ev_value": -res_ev.fun,
        "eev_value": eev_value,
        "sp_value": sp_value,
        "vss": vss,
        "ws_df": ws_df,
        "ws_expected": ws_expected,
        "evpi": evpi,
        "beta_expected": beta_expected,
    }

    return result


def solve_robust_regret(
    first_budget,
    reserve_budget,
    ai_capacity_ratio,
    min_h_share,
    max_item_share,
):
    """
    Robust optimization mở rộng: minimize maximum regret.

    Biến:
        x_j
        y_sj
        z = regret lớn nhất

    Ràng buộc regret:
        WS_s - value_s(x, y_s) <= z
    """
    scenario_df = get_scenario_data()
    beta_base_df, beta_s_df = get_beta_data()

    p = scenario_df["Xác suất"].to_numpy(dtype=float)
    beta_s = beta_s_df[ITEMS].to_numpy(dtype=float)

    n_items = len(ITEMS)
    n_scenarios = len(SCENARIOS)

    # Tính WS_s trước
    ws_values = []
    for s_idx in range(n_scenarios):
        res_ws, info_ws = solve_wait_and_see_one_scenario(
            scenario_index=s_idx,
            first_budget=first_budget,
            reserve_budget=reserve_budget,
            ai_capacity_ratio=ai_capacity_ratio,
            min_h_share=min_h_share,
            max_item_share=max_item_share,
        )

        if not res_ws.success:
            return None, None

        ws_values.append(-res_ws.fun)

    ws_values = np.array(ws_values, dtype=float)

    # Vector biến: x(4) + y(4*4) + z(1)
    n_vars = n_items + n_scenarios * n_items + 1
    z_index = n_vars - 1

    c = np.zeros(n_vars)
    c[z_index] = 1.0

    A_ub = []
    b_ub = []

    # Budget first-stage
    row = np.zeros(n_vars)
    row[:n_items] = 1.0
    A_ub.append(row)
    b_ub.append(first_budget)

    # Min H
    if min_h_share > 0:
        row = np.zeros(n_vars)
        row[ITEMS.index("H")] = -1.0
        A_ub.append(row)
        b_ub.append(-min_h_share * first_budget)

    # Max share
    if max_item_share < 1.0:
        for j in range(n_items):
            row = np.zeros(n_vars)
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(max_item_share * first_budget)

    # Recourse constraints
    for s in range(n_scenarios):
        start = n_items + s * n_items
        end = start + n_items

        row = np.zeros(n_vars)
        row[start:end] = 1.0
        A_ub.append(row)
        b_ub.append(reserve_budget)

        row = np.zeros(n_vars)
        row[start + ITEMS.index("AI")] = 1.0
        row[ITEMS.index("H")] = -ai_capacity_ratio
        A_ub.append(row)
        b_ub.append(0.0)

    # Regret constraints:
    # WS_s - beta_s*x - beta_s*y_s <= z
    # - beta_s*x - beta_s*y_s - z <= -WS_s
    for s in range(n_scenarios):
        start = n_items + s * n_items
        end = start + n_items

        row = np.zeros(n_vars)
        row[:n_items] = -beta_s[s]
        row[start:end] = -beta_s[s]
        row[z_index] = -1.0

        A_ub.append(row)
        b_ub.append(-ws_values[s])

    bounds = [(0.0, None)] * n_vars

    res = linprog(
        c=c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        bounds=bounds,
        method="highs",
    )

    info = {
        "ws_values": ws_values,
        "beta_s": beta_s,
        "probabilities": p,
        "first_budget": first_budget,
        "reserve_budget": reserve_budget,
    }

    return res, info


def unpack_robust_solution(res, info):
    if res is None or not res.success:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    beta_s = info["beta_s"]
    ws_values = info["ws_values"]
    p = info["probabilities"]

    n_items = len(ITEMS)
    n_scenarios = len(SCENARIOS)

    x = res.x[:n_items]
    y = res.x[n_items:n_items + n_scenarios * n_items].reshape(n_scenarios, n_items)
    z = res.x[-1]

    first_df = pd.DataFrame({
        "Hạng mục": ITEMS,
        "Tên hạng mục": [ITEM_LABELS[i] for i in ITEMS],
        "x robust": x,
        "Tỷ trọng (%)": np.where(x.sum() > 0, x / x.sum() * 100, 0),
    })

    second_rows = []
    regret_rows = []

    for s in range(n_scenarios):
        scenario = SCENARIOS[s]
        value_s = np.sum(beta_s[s] * x) + np.sum(beta_s[s] * y[s])
        regret = ws_values[s] - value_s

        regret_rows.append({
            "Kịch bản": scenario,
            "Tên kịch bản": SCENARIO_LABELS[scenario],
            "WS_s": ws_values[s],
            "Giá trị robust": value_s,
            "Regret": regret,
            "Xác suất": p[s],
            "Regret kỳ vọng": p[s] * regret,
        })

        for j, item in enumerate(ITEMS):
            second_rows.append({
                "Kịch bản": scenario,
                "Tên kịch bản": SCENARIO_LABELS[scenario],
                "Hạng mục": item,
                "Tên hạng mục": ITEM_LABELS[item],
                "y robust": y[s, j],
            })

    regret_df = pd.DataFrame(regret_rows)
    second_df = pd.DataFrame(second_rows)

    summary = {
        "max_regret": z,
        "expected_regret": regret_df["Regret kỳ vọng"].sum(),
        "x": x,
        "y": y,
    }

    return first_df, second_df, regret_df, summary


def format_df(df):
    return df.style.format(precision=2, thousands=",")


def render():
    st.title("Bài 10. Quy hoạch ngẫu nhiên hai giai đoạn dưới bất định")
    st.caption("Two-stage stochastic programming | First-stage | Recourse | VSS | EVPI | Robust regret")

    st.markdown(
        """
        Bài toán mô phỏng quyết định phân bổ ngân sách trong điều kiện bất định.
        Chính phủ phải chọn ngân sách **first-stage** trước khi biết kịch bản kinh tế,
        sau đó dùng ngân sách dự phòng **recourse/second-stage** khi kịch bản xảy ra.
        """
    )

    scenario_df = get_scenario_data()
    beta_base_df, beta_s_df = get_beta_data()

    with st.expander("Dữ liệu đầu vào của Bài 10", expanded=False):
        st.subheader("Cấu trúc kịch bản")
        st.dataframe(scenario_df, use_container_width=True)

        st.subheader("Beta cơ bản")
        st.dataframe(beta_base_df, use_container_width=True)

        st.subheader("Beta theo kịch bản")
        beta_show = beta_s_df.copy()
        beta_show["Tên kịch bản"] = beta_show["Kịch bản"].map(SCENARIO_LABELS)
        st.dataframe(beta_show, use_container_width=True)

    st.sidebar.header("Thiết lập mô hình")

    first_budget = st.sidebar.number_input(
        "Ngân sách first-stage",
        min_value=10000.0,
        max_value=150000.0,
        value=65000.0,
        step=5000.0,
    )

    reserve_budget = st.sidebar.number_input(
        "Ngân sách dự phòng second-stage",
        min_value=1000.0,
        max_value=100000.0,
        value=15000.0,
        step=1000.0,
    )

    ai_capacity_ratio = st.sidebar.slider(
        "Ràng buộc y_AI <= ratio * x_H",
        min_value=0.0,
        max_value=2.0,
        value=0.5,
        step=0.1,
    )

    min_h_share = st.sidebar.slider(
        "Sàn tỷ trọng H trong first-stage",
        min_value=0.0,
        max_value=0.8,
        value=0.0,
        step=0.05,
    )

    max_item_share = st.sidebar.slider(
        "Trần tỷ trọng mỗi hạng mục trong first-stage",
        min_value=0.25,
        max_value=1.0,
        value=1.0,
        step=0.05,
    )

    st.subheader("1. Mô hình Stochastic Programming")

    res_sp, info_sp = solve_stochastic_model(
        first_budget=first_budget,
        reserve_budget=reserve_budget,
        ai_capacity_ratio=ai_capacity_ratio,
        min_h_share=min_h_share,
        max_item_share=max_item_share,
    )

    if not res_sp.success:
        st.error("Bài toán không khả thi hoặc solver không tìm được nghiệm.")
        st.code(res_sp.message)
        return

    first_df, second_df, summary = unpack_solution(res_sp, info_sp)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Z* SP", f"{summary['objective']:,.2f}")
    col2.metric("First-stage dùng", f"{summary['first_budget_used']:,.0f}")
    col3.metric("Lợi ích first-stage", f"{summary['first_benefit']:,.2f}")
    col4.metric("Lợi ích recourse kỳ vọng", f"{summary['expected_recourse_benefit']:,.2f}")

    st.markdown("### Phân bổ first-stage tối ưu")
    st.dataframe(format_df(first_df), use_container_width=True)

    st.markdown("### Phân bổ second-stage/recourse theo kịch bản")
    st.dataframe(format_df(second_df), use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Biểu đồ",
        "EV - VSS - EVPI",
        "Deterministic từng kịch bản",
        "Robust regret",
    ])

    with tab1:
        st.markdown("### Biểu đồ phân bổ first-stage")
        chart_first = first_df.set_index("Tên hạng mục")[["x first-stage"]]
        st.bar_chart(chart_first)

        st.markdown("### Bảng pivot recourse theo kịch bản")
        pivot_second = second_df.pivot_table(
            index="Tên kịch bản",
            columns="Tên hạng mục",
            values="y recourse",
            aggfunc="sum",
            fill_value=0,
        )
        st.dataframe(format_df(pivot_second), use_container_width=True)
        st.bar_chart(pivot_second)

    with tab2:
        st.markdown("### Tính EV, EEV, VSS và EVPI")

        vss_evpi = compute_vss_evpi(
            sp_value=summary["objective"],
            first_budget=first_budget,
            reserve_budget=reserve_budget,
            ai_capacity_ratio=ai_capacity_ratio,
            min_h_share=min_h_share,
            max_item_share=max_item_share,
        )

        if vss_evpi is None:
            st.error("Không tính được EV/VSS/EVPI.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("SP", f"{vss_evpi['sp_value']:,.2f}")
            c2.metric("EEV", f"{vss_evpi['eev_value']:,.2f}")
            c3.metric("VSS = SP - EEV", f"{vss_evpi['vss']:,.2f}")
            c4.metric("EVPI = WS - SP", f"{vss_evpi['evpi']:,.2f}")

            beta_expected_df = pd.DataFrame({
                "Hạng mục": ITEMS,
                "Tên hạng mục": [ITEM_LABELS[i] for i in ITEMS],
                "Beta kỳ vọng": vss_evpi["beta_expected"],
            })

            st.markdown("#### Beta kỳ vọng dùng trong mô hình EV")
            st.dataframe(format_df(beta_expected_df), use_container_width=True)

            ev_first_df, ev_second_df, ev_summary = unpack_solution(
                vss_evpi["res_ev"],
                vss_evpi["info_ev"],
                scenario_names=["EV"],
            )

            compare_first = pd.DataFrame({
                "Hạng mục": ITEMS,
                "Tên hạng mục": [ITEM_LABELS[i] for i in ITEMS],
                "SP first-stage": summary["x"],
                "EV first-stage": vss_evpi["x_ev"],
            })

            st.markdown("#### So sánh first-stage giữa SP và EV")
            st.dataframe(format_df(compare_first), use_container_width=True)

            st.bar_chart(compare_first.set_index("Tên hạng mục")[["SP first-stage", "EV first-stage"]])

            st.markdown(
                """
                **Diễn giải nhanh:**

                - **VSS > 0**: xét bất định bằng stochastic programming tốt hơn dùng mô hình kỳ vọng đơn giản.
                - **VSS = 0 hoặc rất nhỏ**: nghiệm SP và EV gần giống nhau với bộ tham số hiện tại.
                - **EVPI** cho biết giá trị tối đa của thông tin hoàn hảo nếu biết trước kịch bản tương lai.
                """
            )

    with tab3:
        st.markdown("### Bài toán deterministic khi biết trước từng kịch bản")

        deterministic_rows = []

        for s_idx, scenario in enumerate(SCENARIOS):
            res_ws, info_ws = solve_wait_and_see_one_scenario(
                scenario_index=s_idx,
                first_budget=first_budget,
                reserve_budget=reserve_budget,
                ai_capacity_ratio=ai_capacity_ratio,
                min_h_share=min_h_share,
                max_item_share=max_item_share,
            )

            if res_ws.success:
                first_ws, second_ws, sum_ws = unpack_solution(
                    res_ws,
                    info_ws,
                    scenario_names=[scenario],
                )

                row = {
                    "Kịch bản": scenario,
                    "Tên kịch bản": SCENARIO_LABELS[scenario],
                    "Z* deterministic": sum_ws["objective"],
                }

                for j, item in enumerate(ITEMS):
                    row[f"x_{item}"] = sum_ws["x"][j]
                    row[f"y_{item}"] = sum_ws["y"][0, j]

                deterministic_rows.append(row)

        deterministic_df = pd.DataFrame(deterministic_rows)
        st.dataframe(format_df(deterministic_df), use_container_width=True)

        if not deterministic_df.empty:
            chart_det = deterministic_df.set_index("Tên kịch bản")[
                ["x_I", "x_D", "x_AI", "x_H"]
            ]
            st.bar_chart(chart_det)

    with tab4:
        st.markdown("### Robust optimization: cực tiểu hóa regret lớn nhất")

        res_robust, info_robust = solve_robust_regret(
            first_budget=first_budget,
            reserve_budget=reserve_budget,
            ai_capacity_ratio=ai_capacity_ratio,
            min_h_share=min_h_share,
            max_item_share=max_item_share,
        )

        robust_first, robust_second, regret_df, robust_summary = unpack_robust_solution(
            res_robust,
            info_robust,
        )

        if robust_first.empty:
            st.error("Không giải được mô hình robust regret.")
        else:
            c1, c2 = st.columns(2)
            c1.metric("Max regret", f"{robust_summary['max_regret']:,.2f}")
            c2.metric("Expected regret", f"{robust_summary['expected_regret']:,.2f}")

            st.markdown("#### First-stage robust")
            st.dataframe(format_df(robust_first), use_container_width=True)

            st.markdown("#### Regret theo từng kịch bản")
            st.dataframe(format_df(regret_df), use_container_width=True)

            st.bar_chart(robust_first.set_index("Tên hạng mục")[["x robust"]])

    st.subheader("2. Gợi ý thảo luận chính sách")

    st.markdown(
        """
        **a) SP có xu hướng đầu tư H nhiều hơn hay ít hơn lời giải xác định?**  
        Nếu bật sàn nhân lực số hoặc siết ràng buộc `y_AI <= ratio*x_H`, mô hình sẽ tăng đầu tư H vì H tạo năng lực hấp thụ AI trong tương lai.

        **b) VSS dương nói lên điều gì?**  
        VSS dương cho thấy việc xét bất định có giá trị. Chính sách không nên chỉ dựa vào một kịch bản trung bình.

        **c) EVPI có ý nghĩa gì?**  
        EVPI là giá trị của thông tin hoàn hảo. Nếu EVPI cao, cần đầu tư mạnh hơn vào dự báo, dữ liệu thời gian thực và hệ thống cảnh báo sớm.

        **d) Vì sao cần giữ ngân sách dự phòng?**  
        Vì Việt Nam phụ thuộc vào xuất khẩu, FDI và chuỗi cung ứng toàn cầu, nên một phần ngân sách cần được giữ lại để phản ứng với kịch bản bi quan hoặc khủng hoảng.
        """
    )

    st.warning(
        "Lưu ý: Vì đây là mô hình LP tuyến tính, nếu không đặt sàn H hoặc trần tỷ trọng mỗi hạng mục, nghiệm có thể dồn nhiều ngân sách vào hạng mục có beta cao nhất. "
        "Bạn có thể chỉnh ở thanh bên để kết quả cân bằng và thực tế hơn."
    )

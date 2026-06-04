# bai09_ai_labor_market.py
# Bài 9: Tác động AI tới thị trường lao động Việt Nam
# Streamlit module: cần có hàm render()

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import linprog

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None


SECTOR_NAMES = [
    "Nông-Lâm-Thủy sản",
    "CN chế biến chế tạo",
    "Xây dựng",
    "Bán buôn-bán lẻ",
    "Tài chính-Ngân hàng",
    "Logistics-Vận tải",
    "CNTT-Truyền thông",
    "Giáo dục-Đào tạo",
]


def make_base_data() -> pd.DataFrame:
    """Tạo bảng tham số Bài 9 theo đề bài."""
    return pd.DataFrame({
        "Ngành": SECTOR_NAMES,
        "Lao động (triệu)": [13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15],
        "Risk (%)": [18, 42, 25, 38, 52, 35, 28, 22],
        "a1 - việc mới AI / tỷ": [8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5],
        "a2 - việc mới CĐS / tỷ": [12.0, 18.5, 8.5, 15.2, 12.5, 16.8, 15.0, 22.0],
        "b1 - nâng cấp việc làm / tỷ": [45.0, 28.0, 35.0, 32.0, 22.0, 30.0, 20.0, 55.0],
        "c1 - dịch chuyển / tỷ": [5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5],
        "d1 - năng lực đào tạo lại / tỷ": [50.0, 32.0, 42.0, 38.0, 26.0, 36.0, 24.0, 62.0],
    })


def solve_labor_lp(
    budget: float = 30000.0,
    include_loss_cap: bool = False,
    sector_cap: float | None = None,
    min_ai_share: float = 0.0,
    min_h_share: float = 0.0,
) -> tuple:
    """
    Giải bài toán LP:

    max sum_i NetJob_i

    NetJob_i = NewJob_i + UpgradeJob_i - DisplacedJob_i
             = a1_i*x_AI_i + b1_i*x_H_i - c1_i*risk_i*x_AI_i

    Ràng buộc:
    - Tổng ngân sách <= budget
    - NetJob_i >= 0
    - DisplacedJob_i <= RetrainingCapacity_i
    - Tùy chọn: DisplacedJob_i <= 5% lao động ngành
    - Tùy chọn: trần ngân sách mỗi ngành
    - Tùy chọn: sàn tỷ trọng AI / H
    """
    df = make_base_data()
    n = len(df)

    risk = df["Risk (%)"].to_numpy(dtype=float) / 100.0
    labor = df["Lao động (triệu)"].to_numpy(dtype=float)
    a1 = df["a1 - việc mới AI / tỷ"].to_numpy(dtype=float)
    b1 = df["b1 - nâng cấp việc làm / tỷ"].to_numpy(dtype=float)
    c1 = df["c1 - dịch chuyển / tỷ"].to_numpy(dtype=float)
    d1 = df["d1 - năng lực đào tạo lại / tỷ"].to_numpy(dtype=float)

    # Vector biến: [x_AI_1..x_AI_8, x_H_1..x_H_8]
    ai_net_coef = a1 - c1 * risk
    h_net_coef = b1

    # scipy.linprog là bài toán min, nên đổi dấu để max
    objective = -np.concatenate([ai_net_coef, h_net_coef])

    A_ub = []
    b_ub = []

    # C1: tổng ngân sách
    A_ub.append(np.ones(2 * n))
    b_ub.append(budget)

    # Sàn tỷ trọng AI nếu người dùng muốn mô phỏng thêm
    if min_ai_share > 0:
        row = np.zeros(2 * n)
        row[:n] = -1
        A_ub.append(row)
        b_ub.append(-budget * min_ai_share)

    # Sàn tỷ trọng H nếu người dùng muốn mô phỏng thêm
    if min_h_share > 0:
        row = np.zeros(2 * n)
        row[n:] = -1
        A_ub.append(row)
        b_ub.append(-budget * min_h_share)

    # C2: NetJob_i >= 0
    # -NetJob_i <= 0
    for i in range(n):
        row = np.zeros(2 * n)
        row[i] = -ai_net_coef[i]
        row[n + i] = -h_net_coef[i]
        A_ub.append(row)
        b_ub.append(0.0)

    # C3: DisplacedJob_i <= RetrainingCapacity_i
    # c1_i*risk_i*x_AI_i - d1_i*x_H_i <= 0
    for i in range(n):
        row = np.zeros(2 * n)
        row[i] = c1[i] * risk[i]
        row[n + i] = -d1[i]
        A_ub.append(row)
        b_ub.append(0.0)

    # C4 mở rộng: không ngành nào mất quá 5% lao động
    # DisplacedJob_i <= 0.05 * L_i
    # L_i đang là triệu người, đổi sang số việc làm: triệu * 1_000_000
    if include_loss_cap:
        for i in range(n):
            row = np.zeros(2 * n)
            row[i] = c1[i] * risk[i]
            A_ub.append(row)
            b_ub.append(0.05 * labor[i] * 1_000_000)

    # Tùy chọn mô phỏng thực tế: trần hấp thụ ngân sách mỗi ngành
    if sector_cap is not None and sector_cap > 0:
        for i in range(n):
            row = np.zeros(2 * n)
            row[i] = 1
            row[n + i] = 1
            A_ub.append(row)
            b_ub.append(sector_cap)

    bounds = [(0, None)] * (2 * n)

    res = linprog(
        c=objective,
        A_ub=np.array(A_ub, dtype=float),
        b_ub=np.array(b_ub, dtype=float),
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        return res, pd.DataFrame(), {}

    x = res.x
    x_ai = x[:n]
    x_h = x[n:]

    new_job_ai = a1 * x_ai
    upgrade_job = b1 * x_h
    displaced_job = c1 * risk * x_ai
    retrain_cap = d1 * x_h
    net_job = new_job_ai + upgrade_job - displaced_job

    result = df.copy()
    result["x_AI tối ưu (tỷ VND)"] = x_ai
    result["x_H tối ưu (tỷ VND)"] = x_h
    result["Tổng đầu tư (tỷ VND)"] = x_ai + x_h
    result["NewJob AI"] = new_job_ai
    result["UpgradeJob"] = upgrade_job
    result["DisplacedJob"] = displaced_job
    result["RetrainingCapacity"] = retrain_cap
    result["NetJob"] = net_job
    result["Ngưỡng mất việc 5% LĐ"] = 0.05 * labor * 1_000_000
    result["Có vượt 5% LĐ?"] = result["DisplacedJob"] > result["Ngưỡng mất việc 5% LĐ"] + 1e-8

    summary = {
        "total_budget_used": float((x_ai + x_h).sum()),
        "total_netjob": float(net_job.sum()),
        "total_newjob": float(new_job_ai.sum()),
        "total_upgrade": float(upgrade_job.sum()),
        "total_displaced": float(displaced_job.sum()),
        "objective_value": float(-res.fun),
    }

    return res, result, summary


def manufacturing_retraining_threshold(x_ai_sector_2: float, budget: float) -> dict:
    """
    Tính ngưỡng x_H tối thiểu cho ngành 2: CN chế biến chế tạo.

    Có 2 cách hiểu:
    1. Chỉ cần NetJob_2 >= 0.
    2. Vừa NetJob_2 >= 0 vừa DisplacedJob_2 <= RetrainingCapacity_2.
    """
    df = make_base_data()
    idx = 1

    risk = df.loc[idx, "Risk (%)"] / 100.0
    a1 = df.loc[idx, "a1 - việc mới AI / tỷ"]
    b1 = df.loc[idx, "b1 - nâng cấp việc làm / tỷ"]
    c1 = df.loc[idx, "c1 - dịch chuyển / tỷ"]
    d1 = df.loc[idx, "d1 - năng lực đào tạo lại / tỷ"]

    ai_net_coef = a1 - c1 * risk

    # Điều kiện NetJob >= 0:
    # ai_net_coef*x_AI + b1*x_H >= 0
    if ai_net_coef >= 0:
        xh_for_netjob = 0.0
    else:
        xh_for_netjob = (-ai_net_coef * x_ai_sector_2) / b1

    # Điều kiện đào tạo lại:
    # c1*risk*x_AI <= d1*x_H
    xh_for_capacity = (c1 * risk * x_ai_sector_2) / d1

    # Ngưỡng chính sách nên dùng: thỏa cả 2 điều kiện
    xh_required_policy = max(xh_for_netjob, xh_for_capacity)

    # Nếu toàn bộ ngân sách chỉ dùng cho ngành 2,
    # max x_AI khả thi khi vẫn đủ x_H đào tạo lại:
    capacity_ratio = (c1 * risk) / d1
    max_ai_feasible_with_budget = budget / (1.0 + capacity_ratio)

    return {
        "ai_net_coef": ai_net_coef,
        "xh_for_netjob": xh_for_netjob,
        "xh_for_capacity": xh_for_capacity,
        "xh_required_policy": xh_required_policy,
        "capacity_ratio": capacity_ratio,
        "max_ai_feasible_with_budget": max_ai_feasible_with_budget,
    }


def format_number(x: float, digits: int = 2) -> str:
    return f"{x:,.{digits}f}"


def make_bar_chart(result: pd.DataFrame):
    if px is None:
        st.warning("Chưa có plotly. Bạn thêm `plotly` vào requirements.txt để hiện biểu đồ đẹp hơn.")
        return

    plot_df = result.melt(
        id_vars="Ngành",
        value_vars=["NewJob AI", "UpgradeJob", "DisplacedJob", "NetJob"],
        var_name="Chỉ tiêu",
        value_name="Số việc làm",
    )

    fig = px.bar(
        plot_df,
        x="Ngành",
        y="Số việc làm",
        color="Chỉ tiêu",
        barmode="group",
        title="So sánh NewJob, UpgradeJob, DisplacedJob và NetJob theo ngành",
    )
    fig.update_layout(xaxis_tickangle=-35, height=520)
    st.plotly_chart(fig, use_container_width=True)


def make_budget_chart(result: pd.DataFrame):
    if px is None:
        st.warning("Chưa có plotly. Bạn thêm `plotly` vào requirements.txt để hiện biểu đồ.")
        return

    plot_df = result.melt(
        id_vars="Ngành",
        value_vars=["x_AI tối ưu (tỷ VND)", "x_H tối ưu (tỷ VND)"],
        var_name="Hạng mục",
        value_name="Ngân sách",
    )

    fig = px.bar(
        plot_df,
        x="Ngành",
        y="Ngân sách",
        color="Hạng mục",
        title="Phân bổ ngân sách tối ưu theo ngành",
    )
    fig.update_layout(xaxis_tickangle=-35, height=520)
    st.plotly_chart(fig, use_container_width=True)


def make_sankey(result: pd.DataFrame):
    if go is None:
        st.warning("Chưa có plotly. Bạn thêm `plotly` vào requirements.txt để hiện Sankey.")
        return

    vulnerable_idx = [0, 2, 3]  # ngành 1, 3, 4
    sub = result.iloc[vulnerable_idx].copy()

    total_flow = (
        sub["DisplacedJob"].sum()
        + sub["NewJob AI"].sum()
        + sub["UpgradeJob"].sum()
    )

    if total_flow <= 1e-8:
        st.info(
            "Nghiệm tối ưu hiện tại không tạo luồng đáng kể ở nhóm ngành 1, 3, 4. "
            "Bạn có thể bật trần ngân sách mỗi ngành hoặc tăng sàn đầu tư AI ở thanh bên để Sankey có dữ liệu trực quan hơn."
        )
        return

    labels = []
    sources = []
    targets = []
    values = []

    def get_node(label: str) -> int:
        if label not in labels:
            labels.append(label)
        return labels.index(label)

    for _, row in sub.iterrows():
        sector = row["Ngành"]

        displaced = max(float(row["DisplacedJob"]), 0.0)
        retrained = min(displaced, max(float(row["RetrainingCapacity"]), 0.0))
        unemployment_risk = max(displaced - retrained, 0.0)
        new_upgrade = max(float(row["NewJob AI"] + row["UpgradeJob"]), 0.0)

        n_sector = get_node(f"LĐ phổ thông - {sector}")
        n_displaced = get_node("Bị dịch chuyển do tự động hóa")
        n_retrained = get_node("Được đào tạo lại")
        n_unemployment = get_node("Nguy cơ thất nghiệp")
        n_new = get_node("Việc làm mới / nâng cấp")

        if displaced > 0:
            sources.append(n_sector)
            targets.append(n_displaced)
            values.append(displaced)

            if retrained > 0:
                sources.append(n_displaced)
                targets.append(n_retrained)
                values.append(retrained)

            if unemployment_risk > 0:
                sources.append(n_displaced)
                targets.append(n_unemployment)
                values.append(unemployment_risk)

        if new_upgrade > 0:
            sources.append(n_sector)
            targets.append(n_new)
            values.append(new_upgrade)

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=18,
            thickness=18,
            line=dict(width=0.5),
            label=labels,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
        )
    )])

    fig.update_layout(
        title_text="Swimming lane / Sankey: luồng dịch chuyển lao động nhóm dễ bị tổn thương",
        height=560,
    )
    st.plotly_chart(fig, use_container_width=True)


def make_policy_commentary(result: pd.DataFrame, summary: dict) -> None:
    top_h = result.sort_values("x_H tối ưu (tỷ VND)", ascending=False).head(3)
    top_ai = result.sort_values("x_AI tối ưu (tỷ VND)", ascending=False).head(3)
    top_net = result.sort_values("NetJob", ascending=False).head(3)

    with st.expander("Gợi ý diễn giải chính sách", expanded=True):
        st.markdown(
            f"""
            **1. Ngành cần đào tạo lại nhiều nhất**

            Theo nghiệm hiện tại, ngành nhận đầu tư `x_H` lớn nhất là **{top_h.iloc[0]["Ngành"]}**
            với khoảng **{format_number(top_h.iloc[0]["x_H tối ưu (tỷ VND)"])} tỷ VND**.
            Điều này xuất hiện vì hệ số tạo/nâng cấp việc làm từ đào tạo của ngành này cao trong bộ tham số mô hình.

            **2. Tài chính - Ngân hàng**

            Ngành Tài chính - Ngân hàng có rủi ro tự động hóa cao, nhưng cũng có hệ số tạo việc làm AI cao.
            Vì vậy, nếu muốn đẩy AI vào ngành này, chính sách nên đi kèm đào tạo lại để kiểm soát điều kiện
            `DisplacedJob <= RetrainingCapacity`.

            **3. Nông - Lâm - Thủy sản**

            Dù hệ số tạo việc làm AI trực tiếp không cao, đây là ngành có quy mô lao động lớn.
            Vì vậy, mô hình gợi ý không nên chỉ nhìn vào hiệu suất AI, mà cần xét thêm tác động an sinh,
            đào tạo lại và khả năng chuyển đổi của lao động phổ thông.

            **4. Ràng buộc quan trọng nhất**

            Mệnh đề “tốc độ tự động hóa không nên vượt quá năng lực đào tạo lại” được biểu diễn trực tiếp bằng:

            `DisplacedJob_i <= RetrainingCapacity_i`

            Đây là ràng buộc bảo vệ xã hội quan trọng hơn so với việc chỉ tối đa hóa tổng NetJob.
            """
        )


def render():
    st.title("Bài 9. Tác động AI tới thị trường lao động Việt Nam")
    st.caption("Mô phỏng NetJob theo ngành dưới tác động của AI, tự động hóa và đào tạo lại lao động.")

    with st.expander("Mô hình toán học sử dụng trong bài", expanded=False):
        st.markdown(
            """
            Mô hình được viết theo dạng tuyến tính:

            **NetJobᵢ = NewJobᵢ + UpgradeJobᵢ − DisplacedJobᵢ**

            Trong phiên bản đơn giản của đề bài:

            - `NewJobᵢ = a1ᵢ × x_AIᵢ`
            - `UpgradeJobᵢ = b1ᵢ × x_Hᵢ`
            - `DisplacedJobᵢ = c1ᵢ × riskᵢ × x_AIᵢ`
            - `RetrainingCapacityᵢ = d1ᵢ × x_Hᵢ`

            Bài toán tối ưu:

            `max Σ NetJobᵢ`

            với các ràng buộc:

            - Tổng ngân sách `Σ(x_AIᵢ + x_Hᵢ) ≤ 30.000`
            - `NetJobᵢ ≥ 0`
            - `DisplacedJobᵢ ≤ RetrainingCapacityᵢ`
            - Mở rộng: `DisplacedJobᵢ ≤ 0,05 × Lᵢ`
            """
        )

    df = make_base_data()

    st.subheader("1. Bảng tham số 8 ngành")
    st.dataframe(df, use_container_width=True)

    st.sidebar.header("Thiết lập mô hình Bài 9")

    budget = st.sidebar.number_input(
        "Tổng ngân sách (tỷ VND)",
        min_value=1000.0,
        max_value=100000.0,
        value=30000.0,
        step=1000.0,
    )

    include_loss_cap = st.sidebar.checkbox(
        "Bật ràng buộc không ngành nào mất quá 5% lao động",
        value=False,
    )

    use_sector_cap = st.sidebar.checkbox(
        "Thêm trần ngân sách mỗi ngành để nghiệm phân bổ đều hơn",
        value=False,
    )

    sector_cap = None
    if use_sector_cap:
        sector_cap = st.sidebar.number_input(
            "Trần ngân sách mỗi ngành (tỷ VND)",
            min_value=1000.0,
            max_value=float(budget),
            value=6000.0,
            step=500.0,
        )

    min_ai_share = st.sidebar.slider(
        "Sàn tỷ trọng đầu tư AI",
        min_value=0.0,
        max_value=0.8,
        value=0.0,
        step=0.05,
    )

    min_h_share = st.sidebar.slider(
        "Sàn tỷ trọng đầu tư đào tạo lại H",
        min_value=0.0,
        max_value=0.8,
        value=0.0,
        step=0.05,
    )

    if min_ai_share + min_h_share > 1.0:
        st.error("Tổng sàn tỷ trọng AI và H đang vượt 100%. Hãy giảm một trong hai giá trị.")
        return

    res, result, summary = solve_labor_lp(
        budget=budget,
        include_loss_cap=include_loss_cap,
        sector_cap=sector_cap,
        min_ai_share=min_ai_share,
        min_h_share=min_h_share,
    )

    st.subheader("2. Kết quả tối ưu hóa")

    if not res.success:
        st.error("Bài toán không khả thi hoặc solver không tìm được nghiệm.")
        st.code(res.message)
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ngân sách sử dụng", f"{summary['total_budget_used']:,.0f} tỷ")
    col2.metric("Tổng NetJob", f"{summary['total_netjob']:,.0f} việc làm")
    col3.metric("Việc làm mới AI", f"{summary['total_newjob']:,.0f}")
    col4.metric("Việc làm bị dịch chuyển", f"{summary['total_displaced']:,.0f}")

    display_cols = [
        "Ngành",
        "x_AI tối ưu (tỷ VND)",
        "x_H tối ưu (tỷ VND)",
        "Tổng đầu tư (tỷ VND)",
        "NewJob AI",
        "UpgradeJob",
        "DisplacedJob",
        "RetrainingCapacity",
        "NetJob",
        "Ngưỡng mất việc 5% LĐ",
        "Có vượt 5% LĐ?",
    ]

    st.dataframe(
        result[display_cols].style.format({
            "x_AI tối ưu (tỷ VND)": "{:,.2f}",
            "x_H tối ưu (tỷ VND)": "{:,.2f}",
            "Tổng đầu tư (tỷ VND)": "{:,.2f}",
            "NewJob AI": "{:,.0f}",
            "UpgradeJob": "{:,.0f}",
            "DisplacedJob": "{:,.0f}",
            "RetrainingCapacity": "{:,.0f}",
            "NetJob": "{:,.0f}",
            "Ngưỡng mất việc 5% LĐ": "{:,.0f}",
        }),
        use_container_width=True,
    )

    if not use_sector_cap and min_ai_share == 0 and min_h_share == 0:
        st.info(
            "Lưu ý: với mô hình gốc không có trần ngân sách theo ngành, nghiệm LP có thể dồn phần lớn ngân sách "
            "vào ngành có hệ số tạo việc làm cao nhất. Đây là đặc điểm bình thường của quy hoạch tuyến tính."
        )

    tab1, tab2, tab3, tab4 = st.tabs([
        "Biểu đồ ngân sách",
        "Biểu đồ NetJob",
        "Ngưỡng ngành chế biến",
        "Sankey nhóm dễ tổn thương",
    ])

    with tab1:
        make_budget_chart(result)

    with tab2:
        make_bar_chart(result)

    with tab3:
        st.markdown("### Câu 9.4.2 - Ngưỡng đầu tư đào tạo lại ngành 2")

        x_ai_max_default = budget
        x_ai_sector_2 = st.slider(
            "Giả định x_AI cho ngành CN chế biến chế tạo (tỷ VND)",
            min_value=0.0,
            max_value=float(budget),
            value=float(x_ai_max_default),
            step=500.0,
        )

        threshold = manufacturing_retraining_threshold(
            x_ai_sector_2=x_ai_sector_2,
            budget=budget,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "x_H để NetJob₂ ≥ 0",
            f"{threshold['xh_for_netjob']:,.2f} tỷ",
        )
        c2.metric(
            "x_H để đủ đào tạo lại",
            f"{threshold['xh_for_capacity']:,.2f} tỷ",
        )
        c3.metric(
            "Ngưỡng chính sách nên dùng",
            f"{threshold['xh_required_policy']:,.2f} tỷ",
        )

        st.markdown(
            f"""
            Với ngành **CN chế biến chế tạo**, hệ số NetJob biên của đầu tư AI là:

            `a1 - c1 × risk = {threshold["ai_net_coef"]:.3f}` việc làm / tỷ VND.

            Vì hệ số này đang dương, nếu chỉ xét điều kiện `NetJob₂ ≥ 0` thì về mặt toán học
            có thể cần rất ít hoặc không cần `x_H`. Tuy nhiên, điều kiện chính sách quan trọng hơn là:

            `DisplacedJob₂ ≤ RetrainingCapacity₂`

            tức lao động bị dịch chuyển phải nằm trong năng lực đào tạo lại.
            Với ngân sách **{budget:,.0f} tỷ VND**, nếu chỉ tập trung vào riêng ngành 2,
            mức `x_AI` tối đa vẫn đủ ngân sách cho đào tạo lại là khoảng
            **{threshold["max_ai_feasible_with_budget"]:,.2f} tỷ VND**.
            """
        )

    with tab4:
        st.markdown("### Câu 9.4.3 - Luồng dịch chuyển lao động nhóm dễ bị tổn thương")
        st.caption("Nhóm dễ bị tổn thương gồm ngành 1, 3, 4: Nông-Lâm-Thủy sản, Xây dựng, Bán buôn-bán lẻ.")
        make_sankey(result)

    st.subheader("3. Kiểm tra ràng buộc mở rộng 5% lao động")
    if include_loss_cap:
        if result["Có vượt 5% LĐ?"].any():
            st.error("Có ít nhất một ngành vượt ngưỡng 5% lao động. Cần xem lại ràng buộc hoặc nghiệm.")
        else:
            st.success("Bài toán khả thi và không ngành nào vượt ngưỡng mất việc 5% lao động.")
    else:
        st.info("Bạn có thể bật ràng buộc 5% ở thanh bên để kiểm tra Câu 9.4.4.")

    make_policy_commentary(result, summary)

# bai04_lp_nganh_vung.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

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


BRAND = "#053151"

REGIONS = ["NMM", "RRD", "NCC", "CH", "SE", "MD"]
REGION_LABELS = {
    "NMM": "Trung du & miền núi Bắc Bộ",
    "RRD": "Đồng bằng sông Hồng",
    "NCC": "Bắc Trung Bộ & DHMT",
    "CH": "Tây Nguyên",
    "SE": "Đông Nam Bộ",
    "MD": "Đồng bằng sông Cửu Long",
}

ITEMS = ["I", "D", "AI", "H"]
ITEM_LABELS = {
    "I": "Hạ tầng số",
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


def beta_table():
    rows = []

    for r in REGIONS:
        row = {"Vùng": r, "Tên vùng": REGION_LABELS[r]}
        for j in ITEMS:
            row[ITEM_LABELS[j]] = BETA[(r, j)]
        rows.append(row)

    return pd.DataFrame(rows)


def solve_pulp(
    budget=50000.0,
    floor_region=5000.0,
    cap_region=12000.0,
    h_min=12000.0,
    gamma=0.002,
    lam=0.68,
    fairness=True,
    use_cap=True,
):
    if not PULP_AVAILABLE:
        return {
            "success": False,
            "status": "PuLP chưa được cài",
            "x": None,
            "objective": np.nan,
            "M": np.nan,
        }

    model = pulp.LpProblem("VN_Digital_Budget", pulp.LpMaximize)

    x = pulp.LpVariable.dicts("x", (REGIONS, ITEMS), lowBound=0)

    model += pulp.lpSum(BETA[(r, j)] * x[r][j] for r in REGIONS for j in ITEMS), "Z"

    model += pulp.lpSum(x[r][j] for r in REGIONS for j in ITEMS) <= budget, "C1_Ngan_sach_tong"

    for r in REGIONS:
        model += pulp.lpSum(x[r][j] for j in ITEMS) >= floor_region, f"C2_San_{r}"

    if use_cap:
        for r in REGIONS:
            model += pulp.lpSum(x[r][j] for j in ITEMS) <= cap_region, f"C3_Tran_{r}"

    model += pulp.lpSum(x[r]["H"] for r in REGIONS) >= h_min, "C4_Nhan_luc_so"

    M = None
    if fairness:
        M = pulp.LpVariable("Dmax", lowBound=0)
        for r in REGIONS:
            model += D0[r] + gamma * x[r]["D"] <= M, f"C5_Max_{r}"
        for r in REGIONS:
            model += D0[r] + gamma * x[r]["D"] >= lam * M, f"C5_Min_{r}"

    solver = pulp.PULP_CBC_CMD(msg=False)
    status_code = model.solve(solver)
    status = pulp.LpStatus[status_code]

    if status != "Optimal":
        return {
            "success": False,
            "status": status,
            "x": None,
            "objective": np.nan,
            "M": np.nan,
            "model": model,
        }

    x_matrix = pd.DataFrame(index=REGIONS, columns=ITEMS, dtype=float)

    for r in REGIONS:
        for j in ITEMS:
            x_matrix.loc[r, j] = float(pulp.value(x[r][j]))

    obj = float(pulp.value(model.objective))

    if fairness and M is not None:
        m_value = float(pulp.value(M))
    else:
        m_value = np.nan

    return {
        "success": True,
        "status": status,
        "x": x_matrix,
        "objective": obj,
        "M": m_value,
        "model": model,
    }


def solve_cvxpy(
    budget=50000.0,
    floor_region=5000.0,
    cap_region=12000.0,
    h_min=12000.0,
    gamma=0.002,
    lam=0.68,
    fairness=True,
    use_cap=True,
):
    if not CVXPY_AVAILABLE:
        return {
            "success": False,
            "status": "CVXPY chưa được cài",
            "x": None,
            "objective": np.nan,
            "M": np.nan,
        }

    n_r = len(REGIONS)
    n_j = len(ITEMS)

    beta_matrix = np.array([[BETA[(r, j)] for j in ITEMS] for r in REGIONS], dtype=float)

    X = cp.Variable((n_r, n_j), nonneg=True)

    constraints = [
        cp.sum(X) <= budget,
        cp.sum(X, axis=1) >= floor_region,
        cp.sum(X[:, ITEMS.index("H")]) >= h_min,
    ]

    if use_cap:
        constraints.append(cp.sum(X, axis=1) <= cap_region)

    M = None
    if fairness:
        M = cp.Variable(nonneg=True)
        d_scores = np.array([D0[r] for r in REGIONS], dtype=float) + gamma * X[:, ITEMS.index("D")]
        constraints += [
            d_scores <= M,
            d_scores >= lam * M,
        ]

    objective = cp.Maximize(cp.sum(cp.multiply(beta_matrix, X)))
    problem = cp.Problem(objective, constraints)

    installed = cp.installed_solvers()
    candidate_solvers = ["CLARABEL", "SCIPY", "ECOS", "SCS"]
    used_solver = None

    for solver_name in candidate_solvers:
        if solver_name in installed:
            try:
                problem.solve(solver=solver_name, verbose=False)
                used_solver = solver_name
                break
            except Exception:
                continue

    if used_solver is None:
        try:
            problem.solve(verbose=False)
            used_solver = "default"
        except Exception as exc:
            return {
                "success": False,
                "status": f"Lỗi solver: {exc}",
                "x": None,
                "objective": np.nan,
                "M": np.nan,
            }

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        return {
            "success": False,
            "status": problem.status,
            "x": None,
            "objective": np.nan,
            "M": np.nan,
            "solver": used_solver,
        }

    x_matrix = pd.DataFrame(X.value, index=REGIONS, columns=ITEMS)

    m_value = float(M.value) if fairness and M is not None and M.value is not None else np.nan

    return {
        "success": True,
        "status": problem.status,
        "x": x_matrix,
        "objective": float(problem.value),
        "M": m_value,
        "solver": used_solver,
    }


def matrix_display(x_matrix):
    out = x_matrix.copy()
    out.index = [REGION_LABELS[r] for r in out.index]
    out = out.rename(columns=ITEM_LABELS)
    out.insert(0, "Vùng", out.index)
    out = out.reset_index(drop=True)
    return out


def long_allocation(x_matrix):
    rows = []

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

    return pd.DataFrame(rows)


def region_summary(x_matrix):
    long_df = long_allocation(x_matrix)

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

    top_item = (
        long_df.sort_values(["Mã vùng", "Ngân sách"], ascending=[True, False])
        .groupby("Mã vùng")
        .head(1)[["Mã vùng", "Hạng mục", "Ngân sách"]]
        .rename(columns={"Hạng mục": "Hạng mục ưu tiên", "Ngân sách": "Ngân sách hạng mục ưu tiên"})
    )

    region_df = region_df.merge(top_item, on="Mã vùng", how="left")

    return region_df


def digital_fairness_table(x_matrix, gamma=0.002):
    rows = []

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

    return pd.DataFrame(rows).sort_values("D sau đầu tư", ascending=False).reset_index(drop=True)


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
    st.title("Bài 4. Phân bổ ngân sách số theo vùng")
    st.caption("PuLP, CVXPY, heatmap phân bổ, ràng buộc công bằng vùng miền và chi phí công bằng")

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
    # 4.4.1
    # =====================================================
    with tabs[0]:
        st.header("4.4.1. Mô hình PuLP và nghiệm tối ưu")

        show_table(beta_table(), decimals=3)

        if not pulp_res["success"]:
            st.error(f"Mô hình PuLP không khả thi hoặc không tối ưu. Trạng thái: {pulp_res['status']}")
            st.info(
                "Với λ = 0,70 theo đề và trần 12.000 tỷ/vùng, mô hình có thể không khả thi. "
                "Bản app đang dùng λ = 0,68 để minh họa nghiệm khả thi."
            )
        else:
            x_matrix = pulp_res["x"]
            region_df = region_summary(x_matrix)

            top_region = region_df.iloc[0]["Vùng"]
            top_budget = region_df.iloc[0]["Tổng ngân sách"]

            c1, c2, c3 = st.columns(3)
            c1.metric("Trạng thái", "Optimal")
            c2.metric("Z*", f"{pulp_res['objective']:,.1f}")
            c3.metric("Vùng nhận nhiều nhất", top_region)

            st.subheader("Ma trận phân bổ tối ưu 6×4")
            show_table(matrix_display(x_matrix), decimals=2)

            st.subheader("Tổng hợp theo vùng")
            show_table(region_df, decimals=2)

            long_df = long_allocation(x_matrix)

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
                f"PuLP tìm được nghiệm tối ưu với Z* = {pulp_res['objective']:,.1f}. "
                f"Vùng nhận ngân sách lớn nhất là {top_region}, khoảng {top_budget:,.1f} tỷ VND."
            )

    # =====================================================
    # 4.4.2
    # =====================================================
    with tabs[1]:
        st.header("4.4.2. So sánh PuLP và CVXPY")

        if not CVXPY_AVAILABLE:
            st.warning("Chưa cài CVXPY. Hãy thêm `cvxpy` vào requirements.txt để chạy phần này.")
        elif not cvx_res["success"]:
            st.warning(f"CVXPY chưa tìm được nghiệm tối ưu. Trạng thái: {cvx_res['status']}")
        elif not pulp_res["success"]:
            st.warning("PuLP chưa có nghiệm tối ưu nên chưa thể so sánh.")
        else:
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

            c1, c2, c3 = st.columns(3)
            c1.metric("Z* PuLP", f"{z_pulp:,.1f}")
            c2.metric("Z* CVXPY", f"{z_cvx:,.1f}")
            c3.metric("Sai khác max x", f"{max_diff:,.4f}")

            show_table(compare_summary, decimals=4)

            st.subheader("Ma trận nghiệm CVXPY")
            show_table(matrix_display(cvx_res["x"]), decimals=2)

            plot_compare = pd.DataFrame(
                {
                    "Phương pháp": ["PuLP", "CVXPY"],
                    "Z*": [z_pulp, z_cvx],
                }
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
                st.info(
                    "Hai phương pháp có thể cho phân bổ hơi khác nhau do sai số solver hoặc nhiều nghiệm tối ưu, "
                    "nhưng giá trị mục tiêu cần được so sánh theo Z*."
                )

    # =====================================================
    # 4.4.3
    # =====================================================
    with tabs[2]:
        st.header("4.4.3. Heatmap phân bổ tối ưu")

        if not pulp_res["success"]:
            st.warning("Chưa có nghiệm tối ưu để vẽ heatmap.")
        else:
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
                matrix_named,
                aspect="auto",
                color_continuous_scale="YlGnBu",
                title="Heatmap phân bổ ngân sách tối ưu 6×4",
                labels=dict(x="Hạng mục", y="Vùng", color="Tỷ VND"),
            )
            fig_heat.update_traces(
                text=matrix_named.values,
                texttemplate="%{text:.0f}",
                hovertemplate="<b>%{y}</b><br>%{x}: %{z:,.1f} tỷ VND<extra></extra>",
            )
            fig_heat.update_layout(
                height=560,
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(color=BRAND),
                title_font=dict(color=BRAND, size=20),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            st.subheader("Hạng mục ưu tiên ở từng vùng")
            show_table(
                region_df[["Vùng", "Tổng ngân sách", "Hạng mục ưu tiên", "Ngân sách hạng mục ưu tiên"]],
                decimals=2,
            )

            st.subheader("Chỉ số số hóa sau đầu tư D")
            fair_df = digital_fairness_table(x_matrix, gamma=gamma)
            show_table(fair_df, decimals=3)

            st.success(
                f"Vùng nhận nhiều ngân sách nhất là {top_region}. "
                f"Hạng mục chiếm tỷ trọng lớn nhất tại vùng này là {top_item_text}."
            )

    # =====================================================
    # 4.4.4
    # =====================================================
    with tabs[3]:
        st.header("4.4.4. Chi phí kinh tế của công bằng vùng miền")

        if not pulp_res["success"] or not pulp_no_fair["success"]:
            st.warning("Không đủ nghiệm để so sánh có/không có ràng buộc công bằng.")
        else:
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

            show_table(compare_df, decimals=3)

            fig = px.bar(
                compare_df,
                x="Mô hình",
                y="Z*",
                text=compare_df["Z*"].round(1),
                title="So sánh Z* giữa các mô hình",
            )
            fig.update_traces(
                marker_color=BRAND,
                textposition="outside",
                textfont=dict(color=BRAND),
            )
            fig.update_layout(xaxis_title="Mô hình", yaxis_title="Z*")
            style_base_fig(fig, height=420)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("So sánh phân bổ có và không có công bằng")

            long_fair = long_allocation(pulp_res["x"])
            long_fair["Mô hình"] = "Có công bằng"

            long_no_fair = long_allocation(pulp_no_fair["x"])
            long_no_fair["Mô hình"] = "Không công bằng"

            compare_alloc = pd.concat([long_fair, long_no_fair], ignore_index=True)

            region_compare = (
                compare_alloc.groupby(["Mô hình", "Vùng"], as_index=False)["Ngân sách"]
                .sum()
            )

            fig_region = px.bar(
                region_compare,
                x="Vùng",
                y="Ngân sách",
                color="Mô hình",
                barmode="group",
                title="Tổng ngân sách theo vùng: có và không có công bằng",
                color_discrete_map=MULTI_COLORS,
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
                st.error(f"Với λ = {test_lam:.2f}, mô hình không khả thi. Trạng thái: {test_res['status']}.")

            st.success(
                f"Chi phí kinh tế của công bằng vùng miền là khoảng {cost_fair:,.1f} tỷ VND GDP gain, "
                f"tương đương {cost_pct:.2f}% so với mô hình không có C5."
            )

    # =====================================================
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

        st.success(
            f"Chi phí công bằng ước tính: {cost_fair:,.1f} tỷ VND GDP gain, khoảng {cost_pct:.2f}% so với mô hình bỏ C5."
        )

        st.markdown(
            """
            Hậu quả xã hội dài hạn nếu bỏ công bằng là khoảng cách số giữa các vùng có thể nới rộng.
            Vùng đã phát triển sẽ tiếp tục nhận nhiều vốn hơn, có hạ tầng tốt hơn, doanh nghiệp số mạnh hơn
            và nhân lực chất lượng cao hơn. Ngược lại, vùng có nền số hóa thấp có thể bị mắc kẹt trong vòng lặp
            thiếu hạ tầng, thiếu nhân lực, thiếu doanh nghiệp công nghệ và năng suất thấp.

            Vì vậy, ràng buộc công bằng làm giảm một phần hiệu quả kinh tế ngắn hạn,
            nhưng giúp giảm rủi ro phân hóa vùng miền trong dài hạn.
            """
        )

        st.markdown("### b) Trần ngân sách mỗi vùng C3 có thể coi là chính sách phân quyền không?")

        st.success(
            f"Khi bỏ trần C3, Z* có thể tăng thêm khoảng {cost_cap:,.1f}, tương đương {cost_cap_pct:.2f}%."
        )

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

        st.markdown("### c) Tây Nguyên nên đầu tư AI hay tập trung H và I trước?")

        st.markdown(
            """
            Trong dữ liệu của bài, Tây Nguyên có hệ số AI thấp nhất, chỉ khoảng 0,45.
            Trong khi đó, hệ số của H và I tại Tây Nguyên lại cao hơn đáng kể.
            Điều này cho thấy nếu xét theo hiệu quả biên trực tiếp, đầu tư vào AI tại Tây Nguyên chưa phải lựa chọn tốt nhất.

            Mô hình hàm ý rằng Tây Nguyên nên ưu tiên **nhân lực số H** và **hạ tầng số I** trước.
            Đây là hai nền tảng giúp vùng cải thiện năng lực hấp thụ công nghệ.
            Khi nhân lực, kết nối, hạ tầng dữ liệu và nền tảng vận hành được cải thiện,
            đầu tư AI sau đó mới có khả năng tạo hiệu quả thực chất hơn.

            Nói cách khác, không nên áp dụng cùng một công thức AI cho mọi vùng.
            Vùng có nền tảng số thấp cần đi theo lộ trình: **hạ tầng số → nhân lực số → dữ liệu/nền tảng → AI ứng dụng**.
            """
        )

        st.info(
            "Kết luận: công bằng vùng miền có chi phí kinh tế ngắn hạn, nhưng là điều kiện quan trọng để tránh phân hóa số dài hạn."
        )


def run():
    render()

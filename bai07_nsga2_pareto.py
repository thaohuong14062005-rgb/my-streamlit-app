# bai07_nsga2_pareto.py

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

try:
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    PYMOO_AVAILABLE = True
except Exception:
    PYMOO_AVAILABLE = False


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
    "Nghiệm thỏa hiệp": "#053151",
    "Tăng trưởng cao nhất": "#E76F51",
}


BETA = np.array(
    [
        [1.15, 0.85, 0.55, 1.30],
        [0.95, 1.25, 1.40, 1.05],
        [1.05, 0.95, 0.85, 1.15],
        [1.20, 0.75, 0.45, 1.35],
        [0.90, 1.30, 1.55, 1.00],
        [1.10, 0.85, 0.65, 1.25],
    ],
    dtype=float,
)

E = np.array([0.42, 0.55, 0.48, 0.32, 0.62, 0.38], dtype=float)
RHO = np.array([0.18, 0.45, 0.28, 0.12, 0.52, 0.22], dtype=float)
SIG = np.array([0.32, 0.28, 0.30, 0.35, 0.25, 0.30], dtype=float)

TOPSIS_WEIGHTS = np.array([0.40, 0.25, 0.20, 0.15], dtype=float)


if PYMOO_AVAILABLE:
    class VietnamDigitalProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(
                n_var=24,
                n_obj=4,
                n_ieq_constr=14,
                xl=np.zeros(24),
                xu=np.ones(24) * 12000,
            )

            self.beta = BETA
            self.e = E
            self.rho = RHO
            self.sig = SIG

        def _evaluate(self, x, out, *args, **kwargs):
            X = x.reshape(6, 4)

            region_sums = X.sum(axis=1)
            total_budget = X.sum()
            total_H = X[:, 3].sum()

            # Pymoo mặc định là minimize.
            # f1 càng nhỏ càng tốt, nên dùng -GDP gain.
            f1 = -(self.beta * X).sum()

            # f2: bất bình đẳng phân bổ vùng, càng nhỏ càng bao trùm.
            f2 = np.abs(region_sums - region_sums.mean()).mean()

            # f3: phát thải từ hạ tầng và AI, càng nhỏ càng tốt.
            f3 = (self.e * (X[:, 0] + X[:, 2])).sum()

            # f4: rủi ro ròng = rủi ro AI - giảm rủi ro nhờ nhân lực.
            f4 = (self.rho * X[:, 2]).sum() - (self.sig * X[:, 3]).sum()

            out["F"] = [f1, f2, f3, f4]

            G = []

            # C1: tổng ngân sách <= 50.000
            G.append(total_budget - 50000)

            # C2: mỗi vùng có sàn 5.000
            for r in range(6):
                G.append(5000 - region_sums[r])

            # C3: mỗi vùng có trần 12.000
            for r in range(6):
                G.append(region_sums[r] - 12000)

            # C4: tổng đầu tư nhân lực số >= 12.000
            G.append(12000 - total_H)

            out["G"] = np.array(G, dtype=float)


def beta_table():
    rows = []

    for i, r in enumerate(REGIONS):
        row = {"Vùng": REGION_LABELS[r]}
        for j, item in enumerate(ITEMS):
            row[ITEM_LABELS[item]] = BETA[i, j]
        rows.append(row)

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def solve_nsga2_cached(pop_size=100, n_gen=200, seed=42):
    if not PYMOO_AVAILABLE:
        return {
            "success": False,
            "message": "Chưa cài pymoo.",
            "X": None,
            "F": None,
            "pareto": pd.DataFrame(),
        }

    problem = VietnamDigitalProblem()

    algorithm = NSGA2(
        pop_size=int(pop_size),
        eliminate_duplicates=True,
    )

    try:
        res = minimize(
            problem,
            algorithm,
            ("n_gen", int(n_gen)),
            seed=int(seed),
            verbose=False,
        )

        X = res.X
        F = res.F

        if X is None or F is None:
            X_pop = res.pop.get("X")
            F_pop = res.pop.get("F")
            G_pop = res.pop.get("G")

            feasible = np.all(G_pop <= 1e-6, axis=1)
            X = X_pop[feasible]
            F = F_pop[feasible]

            if len(F) > 0:
                nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
                X = X[nds]
                F = F[nds]

        if X is None or F is None or len(F) == 0:
            return {
                "success": False,
                "message": "Không tìm thấy nghiệm Pareto khả thi.",
                "X": None,
                "F": None,
                "pareto": pd.DataFrame(),
            }

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if F.ndim == 1:
            F = F.reshape(1, -1)

        pareto = pd.DataFrame(
            {
                "Nghiệm": [f"S{i + 1}" for i in range(len(F))],
                "GDP gain": -F[:, 0],
                "Bao trùm cost": F[:, 1],
                "Phát thải": F[:, 2],
                "Rủi ro ròng": F[:, 3],
                "Tổng ngân sách": X.sum(axis=1),
            }
        )

        return {
            "success": True,
            "message": "Optimal Pareto population extracted.",
            "X": X,
            "F": F,
            "pareto": pareto,
        }

    except Exception as exc:
        return {
            "success": False,
            "message": str(exc),
            "X": None,
            "F": None,
            "pareto": pd.DataFrame(),
        }


def normalize_weights(weights):
    weights = np.array(weights, dtype=float)

    if np.isclose(weights.sum(), 0):
        return np.ones_like(weights) / len(weights)

    return weights / weights.sum()


def topsis_on_pareto(pareto_df, weights=TOPSIS_WEIGHTS):
    w = normalize_weights(weights)

    # Chuyển mọi tiêu chí thành dạng benefit:
    # GDP gain càng cao càng tốt.
    # Bao trùm cost, phát thải, rủi ro ròng càng thấp càng tốt nên lấy dấu âm.
    D = np.column_stack(
        [
            pareto_df["GDP gain"].values,
            -pareto_df["Bao trùm cost"].values,
            -pareto_df["Phát thải"].values,
            -pareto_df["Rủi ro ròng"].values,
        ]
    )

    denom = np.sqrt((D ** 2).sum(axis=0))
    denom = np.where(np.isclose(denom, 0), 1, denom)

    R = D / denom
    V = R * w

    ideal = V.max(axis=0)
    nadir = V.min(axis=0)

    s_pos = np.sqrt(((V - ideal) ** 2).sum(axis=1))
    s_neg = np.sqrt(((V - nadir) ** 2).sum(axis=1))

    c_star = s_neg / (s_pos + s_neg + 1e-12)

    out = pareto_df.copy()
    out["TOPSIS C*"] = c_star
    out["TOPSIS Rank"] = out["TOPSIS C*"].rank(ascending=False, method="min").astype(int)

    out = out.sort_values("TOPSIS C*", ascending=False).reset_index(drop=True)

    return out


def allocation_matrix_from_x(x_vector):
    X = x_vector.reshape(6, 4)

    df = pd.DataFrame(
        X,
        index=[REGION_LABELS[r] for r in REGIONS],
        columns=[ITEM_LABELS[j] for j in ITEMS],
    )

    df.insert(0, "Vùng", df.index)
    df = df.reset_index(drop=True)

    return df


def allocation_long_from_x(x_vector):
    X = x_vector.reshape(6, 4)
    rows = []

    for i, r in enumerate(REGIONS):
        for j, item in enumerate(ITEMS):
            rows.append(
                {
                    "Vùng": REGION_LABELS[r],
                    "Hạng mục": ITEM_LABELS[item],
                    "Ngân sách": X[i, j],
                    "Beta": BETA[i, j],
                    "Đóng góp GDP": X[i, j] * BETA[i, j],
                }
            )

    return pd.DataFrame(rows)


def objective_summary(row, label):
    return pd.DataFrame(
        {
            "Mục tiêu": [
                "GDP gain",
                "Bao trùm cost",
                "Phát thải",
                "Rủi ro ròng",
                "Tổng ngân sách",
            ],
            label: [
                row["GDP gain"],
                row["Bao trùm cost"],
                row["Phát thải"],
                row["Rủi ro ròng"],
                row["Tổng ngân sách"],
            ],
        }
    )


def opportunity_cost(compromise_row, growth_row):
    inclusion_loss = (
        (growth_row["Bao trùm cost"] - compromise_row["Bao trùm cost"])
        / abs(compromise_row["Bao trùm cost"])
        * 100
        if abs(compromise_row["Bao trùm cost"]) > 1e-12
        else np.nan
    )

    env_loss = (
        (growth_row["Phát thải"] - compromise_row["Phát thải"])
        / abs(compromise_row["Phát thải"])
        * 100
        if abs(compromise_row["Phát thải"]) > 1e-12
        else np.nan
    )

    risk_loss = (
        (growth_row["Rủi ro ròng"] - compromise_row["Rủi ro ròng"])
        / abs(compromise_row["Rủi ro ròng"])
        * 100
        if abs(compromise_row["Rủi ro ròng"]) > 1e-12
        else np.nan
    )

    growth_gain = (
        (growth_row["GDP gain"] - compromise_row["GDP gain"])
        / abs(compromise_row["GDP gain"])
        * 100
        if abs(compromise_row["GDP gain"]) > 1e-12
        else np.nan
    )

    return pd.DataFrame(
        {
            "Chỉ tiêu": [
                "Tăng thêm GDP gain",
                "Hi sinh bao trùm",
                "Hi sinh môi trường",
                "Hi sinh an ninh/rủi ro",
            ],
            "Tỷ lệ thay đổi (%)": [
                growth_gain,
                inclusion_loss,
                env_loss,
                risk_loss,
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
    st.title("Bài 7. NSGA-II tối ưu đa mục tiêu chuyển đổi số")
    st.caption("Pareto frontier, scatter 3D, parallel coordinates, TOPSIS và chi phí cơ hội chính sách")

    if not PYMOO_AVAILABLE:
        st.error("Chưa cài pymoo. Hãy thêm `pymoo` vào requirements.txt.")
        return

    res = solve_nsga2_cached(pop_size=100, n_gen=200, seed=42)

    if not res["success"]:
        st.error(res["message"])
        return

    X = res["X"]
    pareto_df = res["pareto"]

    topsis_df = topsis_on_pareto(pareto_df, TOPSIS_WEIGHTS)
    compromise = topsis_df.iloc[0]

    compromise_id = compromise["Nghiệm"]
    compromise_idx = int(str(compromise_id).replace("S", "")) - 1

    growth_idx = pareto_df["GDP gain"].idxmax()
    growth_row = pareto_df.loc[growth_idx]
    growth_x = X[growth_idx]

    compromise_x = X[compromise_idx]

    tabs = st.tabs(
        [
            "7.4.1 NSGA-II",
            "7.4.2 Pareto",
            "7.4.3 TOPSIS",
            "7.4.4 Chi phí cơ hội",
            "7.5 Chính sách",
        ]
    )

    # =====================================================
    # 7.4.1
    # =====================================================
    with tabs[0]:
        st.header("7.4.1. Cài đặt bài toán bằng pymoo NSGA-II")

        c1, c2, c3 = st.columns(3)
        c1.metric("Số biến", "24")
        c2.metric("Số mục tiêu", "4")
        c3.metric("Số nghiệm Pareto", f"{len(pareto_df)}")

        st.subheader("Ma trận hệ số beta 6×4")
        show_table(beta_table(), decimals=3)

        objective_df = pd.DataFrame(
            {
                "Mục tiêu": [
                    "f1 = -GDP gain",
                    "f2 = Bao trùm cost",
                    "f3 = Phát thải",
                    "f4 = Rủi ro ròng",
                ],
                "Ý nghĩa": [
                    "Tối đa hóa tăng trưởng bằng cách tối thiểu hóa giá trị âm",
                    "Giảm bất bình đẳng phân bổ ngân sách giữa các vùng",
                    "Giảm phát thải từ hạ tầng và AI",
                    "Giảm rủi ro ròng từ AI sau khi xét đầu tư nhân lực",
                ],
                "Chiều tối ưu": [
                    "Min",
                    "Min",
                    "Min",
                    "Min",
                ],
            }
        )

        st.subheader("Cấu trúc mục tiêu")
        show_table(objective_df, decimals=3)

        param_df = pd.DataFrame(
            {
                "Tham số": ["pop_size", "n_gen", "seed", "Ngân sách", "Sàn vùng", "Trần vùng", "Sàn nhân lực H"],
                "Giá trị": [100, 200, 42, 50000, 5000, 12000, 12000],
            }
        )

        st.subheader("Tham số thuật toán và ràng buộc")
        show_table(param_df, decimals=0)

        st.success(
            "Đã chạy NSGA-II với pop_size = 100, n_gen = 200 và trích xuất quần thể Pareto cuối cùng."
        )

    # =====================================================
    # 7.4.2
    # =====================================================
    with tabs[1]:
        st.header("7.4.2. Tập Pareto, scatter 3D và parallel coordinates")

        c1, c2, c3 = st.columns(3)
        c1.metric("GDP gain cao nhất", f"{pareto_df['GDP gain'].max():,.1f}")
        c2.metric("Bao trùm cost thấp nhất", f"{pareto_df['Bao trùm cost'].min():,.1f}")
        c3.metric("Phát thải thấp nhất", f"{pareto_df['Phát thải'].min():,.1f}")

        st.subheader("Một phần tập Pareto")
        show_table(pareto_df.head(20), decimals=3)

        fig_3d = px.scatter_3d(
            pareto_df,
            x="GDP gain",
            y="Bao trùm cost",
            z="Phát thải",
            color="Rủi ro ròng",
            hover_name="Nghiệm",
            title="Scatter 3D tập Pareto: tăng trưởng - bao trùm - môi trường",
            color_continuous_scale="Viridis",
        )
        fig_3d.update_layout(
            height=640,
            paper_bgcolor="white",
            font=dict(color=BRAND),
            title_font=dict(color=BRAND, size=20),
            scene=dict(
                xaxis_title="GDP gain",
                yaxis_title="Bao trùm cost",
                zaxis_title="Phát thải",
            ),
        )
        st.plotly_chart(fig_3d, use_container_width=True)

        parallel_df = pareto_df[
            ["GDP gain", "Bao trùm cost", "Phát thải", "Rủi ro ròng"]
        ].copy()

        fig_parallel = px.parallel_coordinates(
            parallel_df,
            dimensions=["GDP gain", "Bao trùm cost", "Phát thải", "Rủi ro ròng"],
            color="GDP gain",
            color_continuous_scale="Viridis",
            title="Parallel coordinates cho 4 mục tiêu",
        )
        fig_parallel.update_layout(
            height=560,
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color=BRAND),
            title_font=dict(color=BRAND, size=20),
        )
        st.plotly_chart(fig_parallel, use_container_width=True)

        st.success(
            "Scatter 3D và parallel coordinates cho thấy mỗi nghiệm Pareto là một cấu hình đánh đổi giữa tăng trưởng, bao trùm, môi trường và an ninh."
        )

    # =====================================================
    # 7.4.3
    # =====================================================
    with tabs[2]:
        st.header("7.4.3. Chọn nghiệm thỏa hiệp bằng TOPSIS")

        weight_df = pd.DataFrame(
            {
                "Mục tiêu chính sách": ["Tăng trưởng", "Bao trùm", "Môi trường", "An ninh"],
                "Trọng số": TOPSIS_WEIGHTS,
            }
        )

        st.subheader("Trọng số TOPSIS")
        show_table(weight_df, decimals=3)

        c1, c2, c3 = st.columns(3)
        c1.metric("Nghiệm thỏa hiệp", compromise["Nghiệm"])
        c2.metric("TOPSIS C*", f"{compromise['TOPSIS C*']:.4f}")
        c3.metric("GDP gain", f"{compromise['GDP gain']:,.1f}")

        st.subheader("Top nghiệm Pareto theo TOPSIS")
        show_table(topsis_df.head(15), decimals=4)

        st.subheader("Ma trận phân bổ của nghiệm thỏa hiệp")
        show_table(allocation_matrix_from_x(compromise_x), decimals=2)

        long_compromise = allocation_long_from_x(compromise_x)

        fig = px.bar(
            long_compromise,
            x="Vùng",
            y="Ngân sách",
            color="Hạng mục",
            title="Phân bổ ngân sách của nghiệm thỏa hiệp TOPSIS",
            color_discrete_map=MULTI_COLORS,
        )
        fig.update_traces(marker_line_color="white", marker_line_width=1)
        fig.update_layout(
            barmode="stack",
            xaxis_title="Vùng",
            yaxis_title="Ngân sách",
        )
        style_base_fig(fig, height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.success(
            f"TOPSIS chọn nghiệm thỏa hiệp {compromise['Nghiệm']} với C* = {compromise['TOPSIS C*']:.4f}."
        )

    # =====================================================
    # 7.4.4
    # =====================================================
    with tabs[3]:
        st.header("7.4.4. Chi phí cơ hội của mục tiêu tăng trưởng")

        cost_df = opportunity_cost(compromise, growth_row)

        compare_df = pd.DataFrame(
            {
                "Chỉ tiêu": ["GDP gain", "Bao trùm cost", "Phát thải", "Rủi ro ròng"],
                "Nghiệm thỏa hiệp": [
                    compromise["GDP gain"],
                    compromise["Bao trùm cost"],
                    compromise["Phát thải"],
                    compromise["Rủi ro ròng"],
                ],
                "Nghiệm tăng trưởng cao nhất": [
                    growth_row["GDP gain"],
                    growth_row["Bao trùm cost"],
                    growth_row["Phát thải"],
                    growth_row["Rủi ro ròng"],
                ],
            }
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Nghiệm thỏa hiệp", compromise["Nghiệm"])
        c2.metric("Nghiệm tăng trưởng cao nhất", growth_row["Nghiệm"])
        c3.metric("GDP gain tăng thêm", f"{cost_df.iloc[0]['Tỷ lệ thay đổi (%)']:.2f}%")

        st.subheader("So sánh mục tiêu")
        show_table(compare_df, decimals=3)

        st.subheader("Chi phí cơ hội")
        show_table(cost_df, decimals=3)

        compare_long = compare_df.melt(
            id_vars="Chỉ tiêu",
            value_vars=["Nghiệm thỏa hiệp", "Nghiệm tăng trưởng cao nhất"],
            var_name="Loại nghiệm",
            value_name="Giá trị",
        )

        fig = px.bar(
            compare_long,
            x="Chỉ tiêu",
            y="Giá trị",
            color="Loại nghiệm",
            barmode="group",
            title="So sánh nghiệm thỏa hiệp và nghiệm tăng trưởng cao nhất",
            color_discrete_map=MULTI_COLORS,
        )
        fig.update_traces(marker_line_color="white", marker_line_width=1)
        fig.update_layout(xaxis_title="Mục tiêu", yaxis_title="Giá trị")
        style_base_fig(fig, height=470)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Ma trận phân bổ nghiệm tăng trưởng cao nhất")
        show_table(allocation_matrix_from_x(growth_x), decimals=2)

        st.success(
            f"Nghiệm tăng trưởng cao nhất tăng GDP gain thêm {cost_df.iloc[0]['Tỷ lệ thay đổi (%)']:.2f}% "
            f"so với nghiệm thỏa hiệp, nhưng có thể làm tăng chi phí bao trùm và phát thải."
        )

    # =====================================================
    # 7.5
    # =====================================================
    with tabs[4]:
        st.header("7.5. Thảo luận chính sách")

        inclusion_loss = cost_df.loc[
            cost_df["Chỉ tiêu"] == "Hi sinh bao trùm", "Tỷ lệ thay đổi (%)"
        ].iloc[0]

        env_loss = cost_df.loc[
            cost_df["Chỉ tiêu"] == "Hi sinh môi trường", "Tỷ lệ thay đổi (%)"
        ].iloc[0]

        st.markdown("### a) Đánh đổi giữa tăng trưởng và bao trùm có rõ ràng không?")

        st.success(
            f"So với nghiệm thỏa hiệp, nghiệm tăng trưởng cao nhất làm bao trùm cost thay đổi khoảng **{inclusion_loss:.2f}%** "
            f"và phát thải thay đổi khoảng **{env_loss:.2f}%**."
        )

        st.markdown(
            """
            Khi quan sát đường biên Pareto, nếu các nghiệm có GDP gain cao thường đi kèm bao trùm cost cao hơn,
            điều đó cho thấy đánh đổi giữa hiệu quả kinh tế và phân bổ công bằng là khá rõ.
            Đây là đặc điểm thường gặp trong một nền kinh tế có chênh lệch năng lực vùng:
            vùng phát triển hơn có khả năng hấp thụ vốn tốt hơn, nên đầu tư vào đó tạo GDP gain cao hơn,
            nhưng đồng thời có thể làm khoảng cách vùng miền tăng lên.

            Nếu đánh đổi này mạnh, chính sách không thể chỉ tối đa hóa tăng trưởng.
            Cần có cơ chế bù đắp cho vùng yếu hơn thông qua hạ tầng số cơ bản, nhân lực số,
            dữ liệu công và hỗ trợ doanh nghiệp địa phương.
            """
        )

        st.markdown("### b) Trọng số 0,40; 0,25; 0,20; 0,15 có phù hợp không?")

        st.markdown(
            """
            Bộ trọng số này đặt tăng trưởng ở vị trí ưu tiên cao nhất, sau đó đến bao trùm,
            môi trường và an ninh. Cách đặt này phù hợp với tư duy phát triển nhanh,
            nâng cao năng suất và thúc đẩy chuyển đổi số như một động lực tăng trưởng.

            Tuy nhiên, nếu muốn phản ánh mạnh hơn cam kết khí hậu và chuyển đổi xanh,
            có thể tăng trọng số môi trường từ 0,20 lên khoảng 0,25 hoặc 0,30.
            Khi đó, trọng số tăng trưởng có thể giảm nhẹ từ 0,40 xuống 0,35,
            đồng thời giữ bao trùm ở mức đủ cao để tránh phân hóa vùng miền.

            Một phương án cân bằng hơn có thể là:
            **tăng trưởng 0,35; bao trùm 0,25; môi trường 0,25; an ninh 0,15**.

            Nếu ưu tiên mạnh cho AI quốc gia và an ninh dữ liệu, có thể nâng an ninh lên 0,20,
            nhưng cần tránh làm giảm quá sâu trọng số bao trùm.
            """
        )

        st.markdown("### c) NSGA-II khác gì so với LP đơn mục tiêu?")

        st.markdown(
            """
            LP đơn mục tiêu cho một nghiệm tối ưu duy nhất theo một hàm mục tiêu đã định trước,
            ví dụ tối đa hóa GDP gain. Cách này rõ ràng và dễ giải thích, nhưng dễ che khuất các đánh đổi
            giữa tăng trưởng, công bằng, môi trường và an ninh.

            NSGA-II không tìm một nghiệm duy nhất ngay từ đầu. Nó tạo ra một tập nghiệm Pareto,
            trong đó không nghiệm nào hoàn toàn tốt hơn nghiệm khác trên tất cả mục tiêu.
            Điều này rất hữu ích cho chính sách vì nhà quản lý có thể nhìn thấy phổ đánh đổi:
            muốn tăng trưởng cao hơn thì phải trả giá bao nhiêu về bao trùm hoặc môi trường.

            Tuy nhiên, NSGA-II không thay thế được quyết định chính trị.
            Thuật toán chỉ cung cấp bản đồ lựa chọn và đo lường trade-off.
            Việc chọn nghiệm cuối cùng vẫn phụ thuộc vào ưu tiên xã hội, chiến lược quốc gia,
            năng lực ngân sách, tính chính danh và quy trình ra quyết định công khai.

            **Kết luận:** NSGA-II là công cụ hỗ trợ quyết định, không phải công cụ thay thế quyết định chính sách.
            """
        )

        st.info(
            f"Kết luận: TOPSIS chọn nghiệm thỏa hiệp **{compromise['Nghiệm']}**; "
            f"nghiệm tăng trưởng cao nhất là **{growth_row['Nghiệm']}**, nhưng phải đánh đổi về bao trùm và môi trường."
        )


def run():
    render()

# bai07_nsga2_pareto.py
# Bài 7 — NSGA-II Pareto 2D + 3D
# Module dùng được với streamlit_app.py có cơ chế gọi module.render()

import random
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except Exception:
    DEAP_AVAILABLE = False


# =====================================================
# 1. DỮ LIỆU MẶC ĐỊNH
# =====================================================

def default_region_data():
    return pd.DataFrame({
        "Vùng": [
            "Trung du miền núi phía Bắc",
            "Đồng bằng sông Hồng",
            "Bắc Trung Bộ + DH Trung Bộ",
            "Tây Nguyên",
            "Đông Nam Bộ",
            "Đồng bằng sông Cửu Long"
        ],
        "GDP_base": [3800, 12500, 5500, 3200, 22000, 4800],
        "AI_readiness": [0.35, 0.50, 0.40, 0.30, 0.55, 0.38],
        "FDI": [5.8, 12.2, 6.5, 4.0, 15.0, 5.0],
        "Inequality_risk": [0.42, 0.32, 0.36, 0.45, 0.38, 0.40],
        "Emission_risk": [0.35, 0.48, 0.40, 0.30, 0.55, 0.37],
    })


def normalize_array(x):
    x = np.asarray(x, dtype=float)
    denom = x.max() - x.min()
    if denom == 0:
        return np.ones_like(x)
    return (x - x.min()) / denom


# =====================================================
# 2. TẠO CREATOR AN TOÀN CHO STREAMLIT
# =====================================================

def safe_create_deap_classes():
    """
    Streamlit thường rerun nhiều lần.
    Nếu creator.create bị gọi lại với cùng tên class, DEAP sẽ báo lỗi.
    Hàm này chỉ tạo class nếu chưa tồn tại.
    """
    if not DEAP_AVAILABLE:
        return

    if not hasattr(creator, "FitnessMulti2D"):
        creator.create("FitnessMulti2D", base.Fitness, weights=(1.0, 1.0))

    if not hasattr(creator, "Individual2D"):
        creator.create("Individual2D", list, fitness=creator.FitnessMulti2D)

    if not hasattr(creator, "FitnessMulti3D"):
        creator.create("FitnessMulti3D", base.Fitness, weights=(1.0, 1.0, -1.0))

    if not hasattr(creator, "Individual3D"):
        creator.create("Individual3D", list, fitness=creator.FitnessMulti3D)


# =====================================================
# 3. HÀM ĐÁNH GIÁ MỤC TIÊU
# =====================================================

def evaluate_policy_vector_2d(individual, df, budget):
    """
    Bài toán 2 mục tiêu:
    Maximize:
    - GDP gain
    - AI readiness gain

    individual: vector phân bổ tỷ trọng cho 6 vùng, giá trị trong [0,1]
    """
    x = np.clip(np.array(individual, dtype=float), 0, 1)

    if x.sum() == 0:
        x = np.ones_like(x) / len(x)
    else:
        x = x / x.sum()

    allocation = budget * x

    gdp_base_norm = normalize_array(df["GDP_base"].values)
    ai_norm = normalize_array(df["AI_readiness"].values)
    fdi_norm = normalize_array(df["FDI"].values)

    gdp_gain = np.sum(allocation * (0.50 + 0.35 * gdp_base_norm + 0.15 * fdi_norm))
    ai_gain = np.sum(allocation * (0.45 + 0.55 * ai_norm))

    return float(gdp_gain), float(ai_gain)


def evaluate_policy_vector_3d(individual, df, budget):
    """
    Bài toán 3 mục tiêu:
    Maximize:
    - GDP gain
    - AI readiness gain

    Minimize:
    - Risk index

    Trong DEAP, mục tiêu thứ ba có weight = -1.0 nên Risk càng thấp càng tốt.
    """
    x = np.clip(np.array(individual, dtype=float), 0, 1)

    if x.sum() == 0:
        x = np.ones_like(x) / len(x)
    else:
        x = x / x.sum()

    allocation = budget * x

    gdp_base_norm = normalize_array(df["GDP_base"].values)
    ai_norm = normalize_array(df["AI_readiness"].values)
    fdi_norm = normalize_array(df["FDI"].values)
    inequality = df["Inequality_risk"].values
    emission = df["Emission_risk"].values

    gdp_gain = np.sum(allocation * (0.50 + 0.35 * gdp_base_norm + 0.15 * fdi_norm))
    ai_gain = np.sum(allocation * (0.45 + 0.55 * ai_norm))

    risk_index = np.sum(x * (0.55 * inequality + 0.45 * emission)) * 100

    return float(gdp_gain), float(ai_gain), float(risk_index)


# =====================================================
# 4. NSGA-II 2D
# =====================================================

def nsga2_optimize_2d(df, budget=50000, n_gen=80, pop_size=120, seed=42):
    if not DEAP_AVAILABLE:
        return None, "DEAP chưa được cài đặt. Hãy thêm `deap` vào requirements.txt."

    safe_create_deap_classes()

    random.seed(seed)
    np.random.seed(seed)

    n_var = len(df)

    toolbox = base.Toolbox()

    toolbox.register("attr_float", random.random)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual2D,
        toolbox.attr_float,
        n=n_var
    )
    toolbox.register(
        "population",
        tools.initRepeat,
        list,
        toolbox.individual
    )

    toolbox.register(
        "evaluate",
        evaluate_policy_vector_2d,
        df=df,
        budget=budget
    )

    toolbox.register(
        "mate",
        tools.cxSimulatedBinaryBounded,
        low=0.0,
        up=1.0,
        eta=20.0
    )

    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        low=0.0,
        up=1.0,
        eta=20.0,
        indpb=1.0 / n_var
    )

    toolbox.register("select", tools.selNSGA2)

    # FIX quan trọng: toolbox.map phải là hàm map hợp lệ.
    toolbox.register("map", map)

    pop = toolbox.population(n=int(pop_size))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        algorithms.eaMuPlusLambda(
            population=pop,
            toolbox=toolbox,
            mu=int(pop_size),
            lambda_=int(pop_size),
            cxpb=0.70,
            mutpb=0.25,
            ngen=int(n_gen),
            verbose=False
        )

    pareto_front = tools.sortNondominated(
        pop,
        k=len(pop),
        first_front_only=True
    )[0]

    rows = []

    for idx, ind in enumerate(pareto_front, start=1):
        x = np.clip(np.array(ind, dtype=float), 0, 1)

        if x.sum() == 0:
            x = np.ones_like(x) / len(x)
        else:
            x = x / x.sum()

        rows.append({
            "Phương án": f"S{idx}",
            "GDP_gain": ind.fitness.values[0],
            "AI_gain": ind.fitness.values[1],
            **{f"w_{df['Vùng'].iloc[i]}": x[i] for i in range(len(df))}
        })

    pareto_df = pd.DataFrame(rows)
    pareto_df = pareto_df.sort_values("GDP_gain", ascending=False).reset_index(drop=True)

    return pareto_df, "OK"


# =====================================================
# 5. NSGA-II 3D
# =====================================================

def nsga2_optimize_3d(df, budget=50000, n_gen=80, pop_size=120, seed=42):
    if not DEAP_AVAILABLE:
        return None, "DEAP chưa được cài đặt. Hãy thêm `deap` vào requirements.txt."

    safe_create_deap_classes()

    random.seed(seed + 100)
    np.random.seed(seed + 100)

    n_var = len(df)

    toolbox = base.Toolbox()

    toolbox.register("attr_float", random.random)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual3D,
        toolbox.attr_float,
        n=n_var
    )
    toolbox.register(
        "population",
        tools.initRepeat,
        list,
        toolbox.individual
    )

    toolbox.register(
        "evaluate",
        evaluate_policy_vector_3d,
        df=df,
        budget=budget
    )

    toolbox.register(
        "mate",
        tools.cxSimulatedBinaryBounded,
        low=0.0,
        up=1.0,
        eta=20.0
    )

    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        low=0.0,
        up=1.0,
        eta=20.0,
        indpb=1.0 / n_var
    )

    toolbox.register("select", tools.selNSGA2)

    # FIX quan trọng: toolbox.map phải là hàm map hợp lệ.
    toolbox.register("map", map)

    pop = toolbox.population(n=int(pop_size))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        algorithms.eaMuPlusLambda(
            population=pop,
            toolbox=toolbox,
            mu=int(pop_size),
            lambda_=int(pop_size),
            cxpb=0.70,
            mutpb=0.25,
            ngen=int(n_gen),
            verbose=False
        )

    pareto_front = tools.sortNondominated(
        pop,
        k=len(pop),
        first_front_only=True
    )[0]

    rows = []

    for idx, ind in enumerate(pareto_front, start=1):
        x = np.clip(np.array(ind, dtype=float), 0, 1)

        if x.sum() == 0:
            x = np.ones_like(x) / len(x)
        else:
            x = x / x.sum()

        rows.append({
            "Phương án": f"S{idx}",
            "GDP_gain": ind.fitness.values[0],
            "AI_gain": ind.fitness.values[1],
            "Risk_index": ind.fitness.values[2],
            **{f"w_{df['Vùng'].iloc[i]}": x[i] for i in range(len(df))}
        })

    pareto_df = pd.DataFrame(rows)
    pareto_df = pareto_df.sort_values("GDP_gain", ascending=False).reset_index(drop=True)

    return pareto_df, "OK"


# =====================================================
# 6. CHỌN PHƯƠNG ÁN KHUYẾN NGHỊ
# =====================================================

def choose_recommended_solution(pareto_df):
    """
    Chọn phương án cân bằng:
    - GDP_gain cao
    - AI_gain cao
    - Risk_index thấp nếu có
    """
    work = pareto_df.copy()

    work["GDP_norm"] = normalize_array(work["GDP_gain"])
    work["AI_norm"] = normalize_array(work["AI_gain"])

    if "Risk_index" in work.columns:
        risk_norm = normalize_array(work["Risk_index"])
        work["Risk_good_norm"] = 1 - risk_norm
        work["Balanced_score"] = (
            0.40 * work["GDP_norm"]
            + 0.40 * work["AI_norm"]
            + 0.20 * work["Risk_good_norm"]
        )
    else:
        work["Balanced_score"] = (
            0.50 * work["GDP_norm"]
            + 0.50 * work["AI_norm"]
        )

    best = work.sort_values("Balanced_score", ascending=False).iloc[0]

    return best, work


# =====================================================
# 7. GIAO DIỆN STREAMLIT
# =====================================================

def render():
    st.markdown(
        """
        <div class="card">
            <h1>🌐 Bài 7 — NSGA-II Pareto đa mục tiêu</h1>
            <p>
            Module này dùng thuật toán NSGA-II để tìm tập nghiệm Pareto giữa các mục tiêu:
            tăng trưởng GDP, năng lực AI và kiểm soát rủi ro khi phân bổ nguồn lực giữa 6 vùng kinh tế.
            </p>
            <span class="pill">NSGA-II</span>
            <span class="pill">Pareto Front</span>
            <span class="pill">Multi-objective Optimization</span>
            <span class="pill">2D + 3D</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    df = default_region_data()

    with st.sidebar:
        st.markdown("### ⚙️ Tham số Bài 7")

        budget = st.number_input(
            "Tổng ngân sách mô phỏng",
            min_value=10000,
            max_value=200000,
            value=50000,
            step=5000
        )

        n_gen = st.number_input(
            "Số thế hệ NSGA-II",
            min_value=10,
            max_value=300,
            value=80,
            step=10
        )

        pop_size = st.number_input(
            "Kích thước quần thể",
            min_value=30,
            max_value=400,
            value=120,
            step=10
        )

        seed = st.number_input(
            "Random seed",
            min_value=1,
            max_value=9999,
            value=42,
            step=1
        )

    if not DEAP_AVAILABLE:
        st.error(
            "DEAP chưa được cài đặt. Hãy thêm `deap` vào file requirements.txt rồi Reboot app."
        )
        st.code(
            "streamlit\npandas\nnumpy\nplotly\ndeap",
            language="text"
        )
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📌 Mô hình",
        "📄 Dữ liệu",
        "7.4.1 Pareto 2D",
        "7.4.2 Pareto 3D",
        "🧠 Thảo luận chính sách"
    ])

    # =====================================================
    # TAB 1 — MÔ HÌNH
    # =====================================================

    with tab1:
        st.header("1. Mô hình tối ưu đa mục tiêu")

        st.markdown("### Biến quyết định")

        st.latex(r"x_r \geq 0,\quad \sum_{r=1}^{6} x_r = 1")

        st.write(
            "Trong đó `x_r` là tỷ trọng ngân sách phân bổ cho vùng `r`. "
            "Tổng ngân sách mô phỏng được phân bổ cho 6 vùng kinh tế."
        )

        st.markdown("### Bài toán 2 mục tiêu")

        st.latex(r"\max f_1(x) = GDPGain(x)")
        st.latex(r"\max f_2(x) = AIGain(x)")

        st.markdown("### Bài toán 3 mục tiêu")

        st.latex(r"\max f_1(x) = GDPGain(x)")
        st.latex(r"\max f_2(x) = AIGain(x)")
        st.latex(r"\min f_3(x) = RiskIndex(x)")

        st.info(
            "NSGA-II không tạo ra một nghiệm duy nhất, mà tạo ra tập nghiệm Pareto. "
            "Mỗi điểm Pareto là một phương án chính sách, trong đó không thể cải thiện một mục tiêu "
            "mà không làm xấu đi ít nhất một mục tiêu khác."
        )

    # =====================================================
    # TAB 2 — DỮ LIỆU
    # =====================================================

    with tab2:
        st.header("2. Dữ liệu 6 vùng")

        st.dataframe(df, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Số vùng", len(df))
        c2.metric("GDP_base cao nhất", f"{df['GDP_base'].max():,.0f}")
        c3.metric("AI readiness cao nhất", f"{df['AI_readiness'].max():.2f}")
        c4.metric("FDI cao nhất", f"{df['FDI'].max():.1f}")

        fig = px.scatter(
            df,
            x="GDP_base",
            y="AI_readiness",
            size="FDI",
            color="Vùng",
            hover_name="Vùng",
            title="Vị thế ban đầu của 6 vùng: GDP base, AI readiness và FDI"
        )
        st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # TAB 3 — PARETO 2D
    # =====================================================

    with tab3:
        st.header("7.4.1 — NSGA-II Pareto 2D: GDP gain và AI gain")

        pareto_2d, msg = nsga2_optimize_2d(
            df=df,
            budget=budget,
            n_gen=n_gen,
            pop_size=pop_size,
            seed=seed
        )

        if pareto_2d is None:
            st.error(msg)
        else:
            best_2d, scored_2d = choose_recommended_solution(pareto_2d)

            c1, c2, c3 = st.columns(3)
            c1.metric("Số nghiệm Pareto", len(pareto_2d))
            c2.metric("GDP gain lớn nhất", f"{pareto_2d['GDP_gain'].max():,.2f}")
            c3.metric("AI gain lớn nhất", f"{pareto_2d['AI_gain'].max():,.2f}")

            st.markdown("### Bảng nghiệm Pareto 2D")
            st.dataframe(scored_2d.round(4), use_container_width=True)

            fig = px.scatter(
                scored_2d,
                x="GDP_gain",
                y="AI_gain",
                color="Balanced_score",
                hover_name="Phương án",
                title="Pareto Front 2D — Trade-off giữa GDP gain và AI gain",
                labels={
                    "GDP_gain": "GDP gain",
                    "AI_gain": "AI gain",
                    "Balanced_score": "Điểm cân bằng"
                }
            )
            st.plotly_chart(fig, use_container_width=True)

            st.success(
                f"Phương án cân bằng đề xuất là {best_2d['Phương án']}, "
                f"với GDP gain = {best_2d['GDP_gain']:,.2f}, "
                f"AI gain = {best_2d['AI_gain']:,.2f}."
            )

            weight_cols = [c for c in scored_2d.columns if c.startswith("w_")]
            best_weights = best_2d[weight_cols].reset_index()
            best_weights.columns = ["Vùng", "Tỷ trọng ngân sách"]
            best_weights["Vùng"] = best_weights["Vùng"].str.replace("w_", "", regex=False)

            fig_weight = px.bar(
                best_weights,
                x="Vùng",
                y="Tỷ trọng ngân sách",
                title="Tỷ trọng phân bổ của phương án cân bằng 2D"
            )
            fig_weight.update_layout(xaxis_tickangle=-25)
            st.plotly_chart(fig_weight, use_container_width=True)

    # =====================================================
    # TAB 4 — PARETO 3D
    # =====================================================

    with tab4:
        st.header("7.4.2 — NSGA-II Pareto 3D: GDP gain, AI gain và Risk index")

        pareto_3d, msg = nsga2_optimize_3d(
            df=df,
            budget=budget,
            n_gen=n_gen,
            pop_size=pop_size,
            seed=seed
        )

        if pareto_3d is None:
            st.error(msg)
        else:
            best_3d, scored_3d = choose_recommended_solution(pareto_3d)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Số nghiệm Pareto", len(pareto_3d))
            c2.metric("GDP gain lớn nhất", f"{pareto_3d['GDP_gain'].max():,.2f}")
            c3.metric("AI gain lớn nhất", f"{pareto_3d['AI_gain'].max():,.2f}")
            c4.metric("Risk thấp nhất", f"{pareto_3d['Risk_index'].min():.2f}")

            st.markdown("### Bảng nghiệm Pareto 3D")
            st.dataframe(scored_3d.round(4), use_container_width=True)

            fig3d = px.scatter_3d(
                scored_3d,
                x="GDP_gain",
                y="AI_gain",
                z="Risk_index",
                color="Balanced_score",
                hover_name="Phương án",
                title="Pareto Front 3D — GDP gain, AI gain và Risk index",
                labels={
                    "GDP_gain": "GDP gain",
                    "AI_gain": "AI gain",
                    "Risk_index": "Risk index",
                    "Balanced_score": "Điểm cân bằng"
                },
                opacity=0.82
            )
            fig3d.update_traces(marker=dict(size=5))
            st.plotly_chart(fig3d, use_container_width=True)

            st.success(
                f"Phương án cân bằng đề xuất là {best_3d['Phương án']}, "
                f"với GDP gain = {best_3d['GDP_gain']:,.2f}, "
                f"AI gain = {best_3d['AI_gain']:,.2f}, "
                f"Risk index = {best_3d['Risk_index']:.2f}."
            )

            weight_cols = [c for c in scored_3d.columns if c.startswith("w_")]
            best_weights = best_3d[weight_cols].reset_index()
            best_weights.columns = ["Vùng", "Tỷ trọng ngân sách"]
            best_weights["Vùng"] = best_weights["Vùng"].str.replace("w_", "", regex=False)

            fig_weight = px.bar(
                best_weights,
                x="Vùng",
                y="Tỷ trọng ngân sách",
                title="Tỷ trọng phân bổ của phương án cân bằng 3D"
            )
            fig_weight.update_layout(xaxis_tickangle=-25)
            st.plotly_chart(fig_weight, use_container_width=True)

    # =====================================================
    # TAB 5 — THẢO LUẬN
    # =====================================================

    with tab5:
        st.header("7.5 — Thảo luận chính sách")

        st.markdown("### a) Vì sao cần Pareto thay vì một nghiệm tối ưu duy nhất?")

        st.write(
            "Trong chính sách phát triển kinh tế số, các mục tiêu thường xung đột nhau. "
            "Một phương án có thể tối đa hóa GDP gain nhưng làm tăng rủi ro bất bình đẳng hoặc phát thải. "
            "Ngược lại, một phương án giảm rủi ro có thể làm giảm tăng trưởng ngắn hạn. "
            "Vì vậy, Pareto front giúp nhà hoạch định chính sách nhìn thấy tập phương án đánh đổi thay vì chỉ một nghiệm duy nhất."
        )

        st.markdown("### b) Điểm Pareto nào nên được chọn?")

        st.write(
            "Không có một điểm Pareto luôn đúng cho mọi bối cảnh. Nếu ưu tiên tăng trưởng, có thể chọn điểm có GDP gain cao. "
            "Nếu ưu tiên chuyển đổi số dài hạn, chọn điểm có AI gain cao. Nếu muốn phát triển bền vững và bao trùm, "
            "nên chọn điểm cân bằng giữa GDP gain, AI gain và Risk index thấp."
        )

        st.markdown("### c) Ý nghĩa của nghiệm cân bằng trong app")

        st.write(
            "Trong module này, nghiệm cân bằng được chọn bằng điểm tổng hợp: GDP gain, AI gain và rủi ro thấp. "
            "Đây không phải là quyết định cuối cùng, mà là phương án tham khảo để hỗ trợ thảo luận chính sách."
        )

        st.success(
            "Kết luận: NSGA-II phù hợp với bài toán chính sách có nhiều mục tiêu xung đột. "
            "Nó giúp minh họa rõ đánh đổi giữa tăng trưởng, năng lực AI và kiểm soát rủi ro vùng miền."
        )


def run():
    render()

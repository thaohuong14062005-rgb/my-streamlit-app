# bai07_nsga2_pareto.py
# Bài 7 — NSGA-II Pareto
# Module Streamlit cho tối ưu đa mục tiêu với NSGA-II

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Sử dụng DEAP cho NSGA-II
try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except:
    DEAP_AVAILABLE = False

# =====================================================
# 1. DỮ LIỆU MẶC ĐỊNH
# =====================================================
def default_data():
    # Ví dụ 6 vùng, 3 mục tiêu (GDP gain, AI readiness, FDI)
    df = pd.DataFrame({
        "Vùng": [
            "Trung du miền núi phía Bắc",
            "Đồng bằng sông Hồng",
            "Bắc Trung Bộ + DH Trung Bộ",
            "Tây Nguyên",
            "Đông Nam Bộ",
            "Đồng bằng sông Cửu Long"
        ],
        "GDP_gain": [3800, 12500, 5500, 3200, 22000, 4800],
        "AI_readiness": [0.35, 0.50, 0.40, 0.30, 0.55, 0.38],
        "FDI": [5.8, 12.2, 6.5, 4.0, 15.0, 5.0]
    })
    return df

# =====================================================
# 2. NSGA-II SETUP
# =====================================================
def nsga2_optimize(n_gen=50, pop_size=50):
    if not DEAP_AVAILABLE:
        st.warning("DEAP chưa được cài đặt. Thêm `deap` vào requirements.txt để chạy NSGA-II.")
        return None

    # Ví dụ tối ưu giả lập: 6 biến [vùng] với mục tiêu tối đa GDP và AI
    n_var = 6

    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))  # maximize 2 mục tiêu
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.rand)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_var)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Hàm mục tiêu (GDP, AI)
    def evaluate(ind):
        # đơn giản: giả lập tradeoff
        # GDP_gain ~ sum biến * 1000
        # AI_readiness ~ sum sqrt(1-var) * 10
        x = np.array(ind)
        obj1 = np.sum(x) * 1000
        obj2 = np.sum(np.sqrt(1 - x)) * 10
        return obj1, obj2

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=0.0, up=1.0, eta=0.5, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=pop_size)
    algorithms.eaMuPlusLambda(pop, pop, mu=pop_size, lambda_=pop_size, cxpb=0.7, mutpb=0.2, ngen=n_gen, verbose=False)

    # Lấy Pareto front
    pareto_front = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]

    return pareto_front

# =====================================================
# 3. STREAMLIT INTERFACE
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
            <h1>🌐 Bài 7 — NSGA-II Pareto</h1>
            <p>
            Tối ưu đa mục tiêu với NSGA-II để tìm trade-off GDP và AI readiness cho 6 vùng.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info("Thuật toán NSGA-II tìm Pareto front cho 2 mục tiêu (GDP, AI readiness).")

    uploaded_file = st.file_uploader("Upload CSV 6 vùng nếu muốn", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.warning(f"Không đọc được file, dùng dữ liệu mặc định. Lỗi: {e}")
            df = default_data()
    else:
        df = default_data()

    st.markdown("### Dữ liệu vùng")
    st.dataframe(df, use_container_width=True)

    st.markdown("### Tham số thuật toán")
    n_gen = st.number_input("Số thế hệ (ngen)", value=50, step=10)
    pop_size = st.number_input("Kích thước quần thể (pop_size)", value=50, step=10)

    pareto_front = nsga2_optimize(n_gen=int(n_gen), pop_size=int(pop_size))
    if pareto_front is None:
        return

    # Chuyển Pareto front thành dataframe
    pareto_df = pd.DataFrame([ind.fitness.values for ind in pareto_front], columns=["GDP_gain", "AI_readiness"])
    pareto_df["Index"] = range(1, len(pareto_df)+1)

    st.markdown("### Pareto front (GDP_gain vs AI_readiness)")
    st.dataframe(pareto_df.round(3), use_container_width=True)

    fig = px.scatter(
        pareto_df,
        x="GDP_gain",
        y="AI_readiness",
        text="Index",
        title="NSGA-II Pareto front",
        size_max=12
    )
    st.plotly_chart(fig, use_container_width=True)

# Alias để streamlit_app.py gọi module.run()
def run():
    render()

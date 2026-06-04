# bai07_nsga2_pareto.py
# Bài 7 — NSGA-II Pareto 2D + 3D
# Module Streamlit cho tối ưu đa mục tiêu với NSGA-II và hiển thị Pareto front 2D + 3D

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# DEAP NSGA-II
try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except Exception:
    DEAP_AVAILABLE = False

# =====================================================
# 1. DỮ LIỆU MẶC ĐỊNH
# =====================================================
def default_data():
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
# 2. NSGA-II 2D
# =====================================================
def nsga2_optimize_2d(n_gen=50, pop_size=50):
    if not DEAP_AVAILABLE:
        st.warning("DEAP chưa được cài đặt. Thêm `deap` vào requirements.txt để chạy NSGA-II.")
        return None

    n_var = 6  # 6 vùng

    # Fitness 2D
    creator.create("FitnessMulti2D", base.Fitness, weights=(1.0, 1.0))
    creator.create("Individual2D", list, fitness=creator.FitnessMulti2D)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.rand)
    toolbox.register("individual", tools.initRepeat, creator.Individual2D, toolbox.attr_float, n=n_var)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate2D(ind):
        x = np.array(ind)
        obj1 = np.sum(x) * 1000        # GDP_gain
        obj2 = np.sum(np.sqrt(1-x)) * 10  # AI_readiness
        return obj1, obj2

    toolbox.register("evaluate", evaluate2D)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=0.0, upBound=1.0, eta=0.5, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=pop_size)
    algorithms.eaMuPlusLambda(pop, pop, mu=pop_size, lambda_=pop_size,
                              cxpb=0.7, mutpb=0.2, ngen=n_gen, verbose=False)

    pareto_front = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
    return pareto_front

# =====================================================
# 3. NSGA-II 3D
# =====================================================
def nsga2_optimize_3d(n_gen=50, pop_size=50):
    if not DEAP_AVAILABLE:
        st.warning("DEAP chưa được cài đặt. Thêm `deap` vào requirements.txt để chạy NSGA-II.")
        return None

    n_var = 6  # 6 vùng

    creator.create("FitnessMulti3D", base.Fitness, weights=(1.0,1.0,1.0))
    creator.create("Individual3D", list, fitness=creator.FitnessMulti3D)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.rand)
    toolbox.register("individual", tools.initRepeat, creator.Individual3D, toolbox.attr_float, n=n_var)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate3D(ind):
        x = np.array(ind)
        obj1 = np.sum(x) * 1000
        obj2 = np.sum(np.sqrt(1-x)) * 10
        obj3 = np.sum(x**1.5) * 5
        return obj1, obj2, obj3

    toolbox.register("evaluate", evaluate3D)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=0.0, upBound=1.0, eta=0.5, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=pop_size)
    algorithms.eaMuPlusLambda(pop, pop, mu=pop_size, lambda_=pop_size,
                              cxpb=0.7, mutpb=0.2, ngen=n_gen, verbose=False)

    pareto_front = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
    return pareto_front

# =====================================================
# 4. STREAMLIT INTERFACE
# =====================================================
def render():
    st.markdown(
        """
        <div style="
            background: rgba(15,23,42,0.86);
            border: 1px solid rgba(148,163,184,0.25);
            border-radius: 22px;
            padding: 24px;
            margin-bottom: 18px;
        ">
            <h1>🌐 Bài 7 — NSGA-II Pareto</h1>
            <p>Tối ưu đa mục tiêu (2D và 3D) cho 6 vùng Việt Nam.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    df = default_data()
    st.markdown("### Dữ liệu vùng")
    st.dataframe(df, use_container_width=True)

    n_gen = st.number_input("Số thế hệ (ngen)", value=50, step=10)
    pop_size = st.number_input("Kích thước quần thể (pop_size)", value=50, step=10)

    # 2D Pareto
    pareto2d = nsga2_optimize_2d(n_gen=int(n_gen), pop_size=int(pop_size))
    if pareto2d:
        pareto_df2d = pd.DataFrame([ind.fitness.values for ind in pareto2d], columns=["GDP_gain","AI_readiness"])
        pareto_df2d["Index"] = range(1,len(pareto_df2d)+1)

        st.markdown("### Pareto front 2D")
        st.dataframe(pareto_df2d.round(3), use_container_width=True)

        fig2d = px.scatter(pareto_df2d, x="GDP_gain", y="AI_readiness", text="Index",
                           title="NSGA-II Pareto front 2D", size_max=12)
        st.plotly_chart(fig2d, use_container_width=True)

    # 3D Pareto
    pareto3d = nsga2_optimize_3d(n_gen=int(n_gen), pop_size=int(pop_size))
    if pareto3d:
        pareto_df3d = pd.DataFrame([ind.fitness.values for ind in pareto3d],
                                   columns=["GDP_gain","AI_readiness","FDI"])
        pareto_df3d["Index"] = range(1,len(pareto_df3d)+1)

        st.markdown("### Pareto front 3D")
        fig3d = px.scatter_3d(pareto_df3d, x="GDP_gain", y="AI_readiness", z="FDI",
                               text="Index", title="NSGA-II Pareto front 3D",
                               color="GDP_gain", size="FDI", size_max=12, opacity=0.8)
        fig3d.update_layout(scene=dict(
            xaxis_title='GDP gain',
            yaxis_title='AI readiness',
            zaxis_title='FDI'
        ))
        st.plotly_chart(fig3d, use_container_width=True)

# Alias để streamlit_app.py gọi module.run()
def run():
    render()

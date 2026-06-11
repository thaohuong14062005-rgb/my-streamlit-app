# bai12_nsga2_topsis.py
# Bài 12 - Tối ưu tích hợp dự án 6 vùng, 4 hạng mục với NSGA-II + TOPSIS
# Module Streamlit

import numpy as np
import pandas as pd
import streamlit as st

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.util.non_dominated_sorting import NonDominatedSorting

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None


# ============================================================
# 1. Tham số
# ============================================================

V = 6  # vùng
H = 4  # hạng mục I,D,AI,H
budget_total = 80000
# Ví dụ beta giả định (6 vùng x 4 hạng mục)
beta = np.array([
    [21500, 20800, 32500, 9200],
    [6800, 5800, 12200, 2850],
    [11500, 12000, 18000, 4500],
    [5800, 4000, 9000, 1800],
    [20000, 3500, 13000, 7000],
    [3800, 2800, 3800, 1200]
], dtype=float)

e = np.array([0.42, 0.55, 0.48, 0.32, 0.62, 0.38])
rho = np.array([0.18, 0.45, 0.28, 0.12, 0.52, 0.22])
sig = np.array([0.32, 0.28, 0.30, 0.35, 0.25, 0.30])


# ============================================================
# 2. Định nghĩa bài toán NSGA-II
# ============================================================

class VietnamProjectProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(
            n_var=V*H,
            n_obj=4,
            n_ieq_constr=5,
            xl=np.zeros(V*H),
            xu=np.ones(V*H)*12000
        )
        self.beta = beta
        self.e = e
        self.rho = rho
        self.sig = sig

    def _evaluate(self, x, out, *args, **kwargs):
        X = x.reshape(V,H)
        f1 = -(self.beta*X).sum()  # GDP gain maximize
        sums = X.sum(axis=1)
        f2 = np.abs(sums - sums.mean()).mean()  # Gini
        f3 = (self.e * (X[:,0]+X[:,2])).sum()  # Emission
        f4 = (self.rho*X[:,2]).sum() - (self.sig*X[:,3]).sum()  # Rủi ro ròng
        out['F'] = [f1, f2, f3, f4]

        # Ràng buộc ví dụ
        g1 = X.sum() - budget_total  # tổng ngân sách <= 80k
        g2 = X[0,0]+X[1,0]-15000  # ràng buộc minh họa dự án 1-2
        g3 = X[7-1,0]-X[12-1,0]  # ràng buộc tiên quyết AI quốc gia
        g4 = X[13-1,0]-X[12-1,0] # bán dẫn cần đào tạo
        g5 = X[:,0].sum() - 40000  # ngân sách năm 1-2
        out['G'] = [g1,g2,g3,g4,g5]


# ============================================================
# 3. Streamlit UI
# ============================================================

def render():
    st.title("Bài 12 - NSGA-II và TOPSIS dự án 6 vùng 4 hạng mục")

    st.markdown("""
        Tối ưu lựa chọn dự án 6 vùng kinh tế, 4 hạng mục (I,D,AI,H) sử dụng NSGA-II.
        Sau đó áp dụng TOPSIS để chọn nghiệm thỏa hiệp dựa trên trọng số chính sách:
        0.4 tăng trưởng, 0.25 bao trùm, 0.2 môi trường, 0.15 an ninh.
    """)

    with st.expander("Cài đặt NSGA-II", expanded=True):
        pop_size = st.number_input("Kích thước quần thể", 50, 500, 100)
        n_gen = st.number_input("Số thế hệ", 50, 500, 200)

    problem = VietnamProjectProblem()
    algorithm = NSGA2(pop_size=pop_size)

    with st.spinner("Đang tối ưu NSGA-II..."):
        res = minimize(problem, algorithm, ('n_gen', n_gen), seed=42, verbose=True)

    F = res.F
    X = res.X

    st.subheader("Kết quả NSGA-II - biên Pareto")
    st.write("Số nghiệm Pareto:", len(F))

    if px is not None:
        fig = px.scatter_3d(
            x=F[:,0], y=F[:,1], z=F[:,2],
            color=F[:,3],
            labels={'x':'GDP gain','y':'Gini','z':'Emission','color':'Rủi ro ròng'},
            title="Pareto 4 objectives"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Parallel coordinates
    if px is not None:
        df_par = pd.DataFrame(F, columns=['GDP','Gini','Emission','Risk'])
        fig2 = px.parallel_coordinates(df_par, color='GDP', labels={'GDP':'GDP','Gini':'Gini','Emission':'Emission','Risk':'Risk'})
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("TOPSIS - chọn nghiệm thỏa hiệp")
    weights = np.array([0.4,0.25,0.2,0.15])
    norm_F = F / (F.max(axis=0)-F.min(axis=0)+1e-6)
    ideal = norm_F.max(axis=0)
    anti_ideal = norm_F.min(axis=0)
    distance_ideal = np.sqrt(((norm_F - ideal)**2 * weights).sum(axis=1))
    distance_anti = np.sqrt(((norm_F - anti_ideal)**2 * weights).sum(axis=1))
    score = distance_anti / (distance_ideal + distance_anti)
    best_idx = np.argmax(score)
    st.write("Nghiệm thỏa hiệp (TOPSIS) chọn index:", best_idx)
    st.write("Điểm TOPSIS:", score[best_idx])
    st.write("X thỏa hiệp:\n", X[best_idx].reshape(V,H))

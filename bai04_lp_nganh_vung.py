# bai04_lp_nganh_vung.py
import streamlit as st
import numpy as np
import pandas as pd
import pulp
import cvxpy as cp
import seaborn as sns
import matplotlib.pyplot as plt

def render():
    st.title("Bài 4. Tối ưu phân bổ ngân sách vùng và hạng mục")
    
    # Regions và Items
    regions = ['NMM','RRD','NCC','CH','SE','MD']
    items = ['I','D','AI','H']
    
    # Beta hiệu quả
    beta = {
        ('NMM','I'):1.15,('NMM','D'):0.85,('NMM','AI'):0.55,('NMM','H'):1.30,
        ('RRD','I'):0.95,('RRD','D'):1.25,('RRD','AI'):1.40,('RRD','H'):1.05,
        ('NCC','I'):1.05,('NCC','D'):0.95,('NCC','AI'):0.85,('NCC','H'):1.15,
        ('CH','I') :1.20,('CH','D') :0.75,('CH','AI') :0.45,('CH','H') :1.35,
        ('SE','I') :0.90,('SE','D') :1.30,('SE','AI') :1.55,('SE','H') :1.00,
        ('MD','I') :1.10,('MD','D') :0.85,('MD','AI') :0.65,('MD','H') :1.25
    }
    
    D0 = {'NMM':38,'RRD':78,'NCC':55,'CH':32,'SE':82,'MD':48}
    gamma, lam = 0.002, 0.7
    
    # --- PuLP Model ---
    m = pulp.LpProblem('VN_Digital_Budget', pulp.LpMaximize)
    x = pulp.LpVariable.dicts('x', (regions, items), lowBound=0)

    # Objective
    m += pulp.lpSum(beta[(r,j)]*x[r][j] for r in regions for j in items)

    # Ràng buộc tổng ngân sách
    m += pulp.lpSum(x[r][j] for r in regions for j in items) <= 50000

    # Ngân sách mỗi vùng
    for r in regions:
        m += pulp.lpSum(x[r][j] for j in items) >= 5000
        m += pulp.lpSum(x[r][j] for j in items) <= 12000

    # Ràng buộc tối thiểu H
    m += pulp.lpSum(x[r]['H'] for r in regions) >= 12000

    # Ràng buộc công bằng (linearized)
    M = pulp.LpVariable('Dmax')
    for r in regions:
        m += D0[r] + gamma*x[r]['D'] <= M
        m += D0[r] + gamma*x[r]['D'] >= lam*M

    # Solve
    m.solve(pulp.PULP_CBC_CMD(msg=False))

    # Lấy kết quả
    result = pd.DataFrame(index=regions, columns=items)
    for r in regions:
        for j in items:
            result.loc[r,j] = pulp.value(x[r][j])
    st.subheader("Phân bổ tối ưu (PuLP)")
    st.table(result)

    # --- CVXPY Model (so sánh) ---
    x_cvx = cp.Variable((len(regions),len(items)), nonneg=True)
    beta_matrix = np.array([[beta[(r,j)] for j in items] for r in regions])
    D0_vec = np.array([D0[r] for r in regions])
    
    constraints = [
        cp.sum(x_cvx) <= 50000,
        cp.sum(x_cvx, axis=1) >= 5000,
        cp.sum(x_cvx, axis=1) <= 12000,
        cp.sum(x_cvx[:,3]) >= 12000,
        D0_vec + gamma*x_cvx[:,1] <= cp.Variable(1),  # không bắt buộc linearize phức tạp
        D0_vec + gamma*x_cvx[:,1] >= lam*cp.Variable(1)
    ]
    obj = cp.Maximize(cp.sum(cp.multiply(beta_matrix, x_cvx)))
    prob = cp.Problem(obj,constraints)
    prob.solve(solver=cp.ECOS)

    st.subheader("Phân bổ tối ưu (CVXPY)")
    df_cvx = pd.DataFrame(x_cvx.value, index=regions, columns=items)
    st.table(df_cvx)

    # --- Heatmap ---
    st.subheader("Heatmap phân bổ")
    plt.figure(figsize=(8,5))
    sns.heatmap(result.astype(float), annot=True, cmap="Blues", fmt=".0f")
    st.pyplot(plt.gcf())

    # --- So sánh với mô hình bỏ ràng buộc công bằng ---
    st.subheader("So sánh với bỏ ràng buộc công bằng")
    m2 = pulp.LpProblem('NoFairBudget', pulp.LpMaximize)
    x2 = pulp.LpVariable.dicts('x2', (regions, items), lowBound=0)
    m2 += pulp.lpSum(beta[(r,j)]*x2[r][j] for r in regions for j in items)
    m2 += pulp.lpSum(x2[r][j] for r in regions for j in items) <= 50000
    for r in regions:
        m2 += pulp.lpSum(x2[r][j] for j in items) >= 5000
        m2 += pulp.lpSum(x2[r][j] for j in items) <= 12000
    m2 += pulp.lpSum(x2[:,3] for r in regions) >= 12000
    m2.solve(pulp.PULP_CBC_CMD(msg=False))
    df_nofair = pd.DataFrame(index=regions, columns=items)
    for r in regions:
        for j in items:
            df_nofair.loc[r,j] = pulp.value(x2[r][j])
    st.table(df_nofair)
    st.markdown("**Lưu ý:** Chi phí công bằng = hiệu số Z* với/bỏ ràng buộc công bằng.")

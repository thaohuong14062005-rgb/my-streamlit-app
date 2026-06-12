# bai12_aideom_vn.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

YEARS = np.arange(2026,2031)
SCENARIOS = [
    {"Kịch bản":"S1","K":0.7,"D":0.1,"AI":0.1,"H":0.1},
    {"Kịch bản":"S2","K":0.25,"D":0.45,"AI":0.15,"H":0.15},
    {"Kịch bản":"S3","K":0.2,"D":0.2,"AI":0.45,"H":0.15},
    {"Kịch bản":"S4","K":0.3,"D":0.2,"AI":0.1,"H":0.4},
]

def simulate(s):
    rows=[]
    gdp=8500
    for y in YEARS:
        gdp=gdp*(1+0.05 + 0.2*s["K"]+0.1*s["D"]+0.15*s["AI"]+0.1*s["H"])
        rows.append({"Năm":y,"Kịch bản":s["Kịch bản"],"GDP":gdp})
    return pd.DataFrame(rows)

def render():
    st.title("Bài 12. AIDEOM-VN – Tinh gọn")
    sim_df = pd.concat([simulate(s) for s in SCENARIOS],ignore_index=True)

    tab1,tab2,tab3 = st.tabs(["Đường kịch bản","KPI 2030","Khuyến nghị"])
    
    with tab1:
        fig=px.line(sim_df,x="Năm",y="GDP",color="Kịch bản",markers=True)
        st.plotly_chart(fig,use_container_width=True)
    
    with tab2:
        kpi_2030 = sim_df[sim_df["Năm"]==2030].copy()
        kpi_2030["Xếp hạng"]=kpi_2030["GDP"].rank(ascending=False).astype(int)
        st.table(kpi_2030[["Kịch bản","GDP","Xếp hạng"]])

    with tab3:
        best=kpi_2030.sort_values("Xếp hạng").iloc[0]["Kịch bản"]
        st.success(f"Kịch bản ưu tiên 2026-2030: {best}")
        st.markdown("""
        - S1: truyền thống, đối chứng
        - S2/S3: tăng tốc số hóa và AI, cần an ninh và đào tạo
        - S4: bao trùm, công bằng xã hội
        - Chọn S5 nếu muốn cân bằng giữa tăng trưởng, AI, việc làm và rủi ro
        """)

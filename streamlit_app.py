# streamlit_app.py
import streamlit as st

# Import modules 12 bài
import bai01_cobb_douglas as bai01
import bai02_lp_phan_bo as bai02
import bai03_priority as bai03
import bai04_lp_nganh_vung as bai04
import bai05_mip_15_du_an as bai05
import bai06_topsis_6_vung as bai06
import bai07_nsga2_pareto as bai07
import bai08_dynamic_2026_2035 as bai08
import bai09_lao_dong_ai as bai09
import bai10_stochastic_sp as bai10
import bai11_q_learning_rl as bai11
import bai12_aideom_vn as bai12

# Cấu hình page
st.set_page_config(page_title="AIDEOM-VN", layout="wide")
st.title("🌱 AIDEOM-VN Dashboard — Mô hình ra quyết định")

# Sidebar navigation
tab = st.sidebar.radio("Chọn bài tập / module", [
    "Trang chủ",
    "Bài 1 — Cobb-Douglas",
    "Bài 2 — LP phân bổ",
    "Bài 3 — Priority ngành",
    "Bài 4 — LP ngành-vùng",
    "Bài 5 — MIP 15 dự án",
    "Bài 6 — TOPSIS 6 vùng",
    "Bài 7 — NSGA-II Pareto",
    "Bài 8 — Quy hoạch động 2026-2035",
    "Bài 9 — Lao động & AI",
    "Bài 10 — Stochastic SP",
    "Bài 11 — Q-learning RL",
    "Bài 12 — AIDEOM-VN Dashboard"
])

# Trang chủ
if tab == "Trang chủ":
    st.markdown("""
    # Chào mừng đến với AIDEOM-VN
    Hệ thống mô hình ra quyết định cho kinh tế Việt Nam giai đoạn 2020-2035.
    
    **Hướng dẫn sử dụng:**
    - Chọn module bài tập từ sidebar.
    - Mỗi module hiển thị dữ liệu, đồ thị và kết quả tính toán.
    - Các module chạy độc lập, nhưng có thể chia sẻ dữ liệu chung trong thư mục `data/`.
    """)

# Gọi module tương ứng
elif tab == "Bài 1 — Cobb-Douglas":
    bai01.run()
elif tab == "Bài 2 — LP phân bổ":
    bai02.run()
elif tab == "Bài 3 — Priority ngành":
    bai03.run()
elif tab == "Bài 4 — LP ngành-vùng":
    bai04.run()
elif tab == "Bài 5 — MIP 15 dự án":
    bai05.run()
elif tab == "Bài 6 — TOPSIS 6 vùng":
    bai06.run()
elif tab == "Bài 7 — NSGA-II Pareto":
    bai07.run()
elif tab == "Bài 8 — Quy hoạch động 2026-2035":
    bai08.run()
elif tab == "Bài 9 — Lao động & AI":
    bai09.run()
elif tab == "Bài 10 — Stochastic SP":
    bai10.run()
elif tab == "Bài 11 — Q-learning RL":
    bai11.run()
elif tab == "Bài 12 — AIDEOM-VN Dashboard":
    bai12.run()

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pulp

# ==============================================================================
# 1. CẤU HÌNH GIAO DIỆN CHUNG (DARK THEME ĐƯỢC TỰ ĐỘNG ÁP DỤNG THEO HỆ THỐNG)
# ==============================================================================
st.set_page_config(
    page_title="AIDEOM-VN | Mô Hình Ra Quyết Định",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Thêm CSS tùy chỉnh để giao diện trông chuyên nghiệp, đồng bộ màu sắc
st.markdown("""
    <style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #ffffff; margin-bottom: 1rem; }
    .sub-title { font-size: 1.2rem; color: #a0aec0; margin-bottom: 2rem; }
    .stTabs [data-baseweb="tab"] { font-size: 14px; font-weight: 600; }
    </style>
""", unsafe_allowed_html=True)

# ==============================================================================
# 2. THANH ĐIỀU HƯỚNG BÊN TRÁI (SIDEBAR NAVIGATION)
# ==============================================================================
st.sidebar.markdown("# 🌱 VN AIDEOM-VN")
st.sidebar.markdown("*Mô hình ra quyết định phát triển kinh tế VN trong kỉ nguyên AI*")
st.sidebar.markdown("---")

# Danh sách 12 bài tập chuẩn theo tài liệu
pages = [
    "🏠 Trang chủ",
    "Bài 1 — Cobb-Douglas + AI",
    "Bài 2 — LP ngân sách số",
    "Bài 3 — Priority 10 ngành",
    "Bài 4 — LP ngành-vùng",
    "Bài 5 — MIP 15 dự án",
    "Bài 6 — TOPSIS 6 vùng",
    "Bài 7 — NSGA-II Pareto",
    "Bài 8 — Động 2026-2035",
    "Bài 9 — Lao động & AI",
    "Bài 10 — Stochastic SP",
    "Bài 11 — Q-learning RL",
    "Bài 12 — AIDEOM tích hợp"
]

# Sử dụng radio button cho danh sách bài tập (mặc định trỏ vào Bài 4)
selected_page = st.sidebar.radio("Danh sách bài tập thực hành", pages, index=4)

st.sidebar.markdown("---")
st.sidebar.caption("Dữ liệu tham chiếu: NSO, MoST, MIC, MPI, WB, GII 2025")

# ==============================================================================
# 3. ĐỊNH NGHĨA GIAO DIỆN TRANG CHỦ
# ==============================================================================
def render_home():
    st.markdown('<div class="main-title">🏠 Trang chủ - Hệ Thống Mô Hình Ra Quyết Định</div>', unsafe_allowed_html=True)
    st.markdown("""
    Bộ bài tập thực hành này được thiết kế nhằm mô hình hóa các bài toán chính sách kinh tế Việt Nam trong kỷ nguyên trí tuệ nhân tạo.
    Hệ thống được tổ chức theo độ khó tăng dần qua 4 cấp độ:
    
    * 🟢 **Cấp độ Dễ (Bài 1-3):** Ước lượng hàm sản xuất Cobb-Douglas mở rộng, Quy hoạch tuyến tính (LP) phân bổ ngân sách đơn giản và tính chỉ số ưu tiên ngành (TOPSIS cơ bản).
    * 🟡 **Cấp độ Trung bình (Bài 4-6):** Quy hoạch tuyến tính đa biến (LP ngành - vùng), Quy hoạch nguyên hỗn hợp (MIP) lựa chọn dự án công nghệ và Xếp hạng vùng theo thuật toán TOPSIS Entropy.
    * 🟠 **Cấp độ Khá khó (Bài 7-9):** Tối ưu đa mục tiêu bằng thuật toán tiến hóa NSGA-II, Quy hoạch động phi tuyến liên thời gian (2026-2035) và Mô phỏng dịch chuyển thị trường lao động.
    * 🔴 **Cấp độ Khó (Bài 10-12):** Quy hoạch ngẫu nhiên hai giai đoạn (Stochastic SP), Học tăng cường thích nghị (Q-learning RL) và Đồ án tích hợp hệ thống dashboard AIDEOM-VN toàn diện.
    
    👈 *Vui lòng chọn bài tập cụ thể trên thanh bên trái để xem chi tiết thuật toán và chạy mô phỏng chính sách.*
    """)

# ==============================================================================
# 4. ĐỊNH NGHĨA GIAO DIỆN VÀ CODE GIẢI BÀI 4
# ==============================================================================
def render_bai_4():
    st.markdown('<div class="main-title">Bài 4 — Quy hoạch tuyến tính phân bổ ngân sách số theo ngành - vùng</div>', unsafe_allowed_html=True)
    st.markdown("<div class=\"sub-title\">Tối ưu hóa phân bổ 50.000 tỷ VND ngân sách kinh tế số quốc gia cho 6 vùng kinh tế xã hội và 4 hạng mục đầu tư chiến lược nhằm tối đa hóa tăng trưởng GDP và bảo đảm công bằng vùng miền.</div>", unsafe_allowed_html=True)
    
    # 3 Tab chức năng tương tự như sơ đồ thiết kế UI ứng dụng
    tab_model, tab_result, tab_analysis = st.tabs(["📋 Mô hình toán & Tham số", "📊 Kết quả tối ưu", "📈 Phân tích độ nhạy"])
    
    # Thiết lập bộ dữ liệu gốc từ tệp đính kèm tài liệu
    regions = ['1. Trung du miền núi phía Bắc', '2. Đồng bằng sông Hồng', '3. Bắc Trung Bộ + DH Trung Bộ', '4. Tây Nguyên', '5. Đông Nam Bộ', '6. Đồng bằng sông Cửu Long']
    items = ['I', 'D', 'AI', 'H'] # I: Hạ tầng số, D: Chuyển đổi số DN, AI: Năng lực AI, H: Nhân lực số
    
    # Ma trận hệ số tác động biên beta (đơn vị: nghìn VND GDP / 1 triệu VND đầu tư)
    beta = {
        ('1. Trung du miền núi phía Bắc', 'I'): 1.15, ('1. Trung du miền núi phía Bắc', 'D'): 0.85, ('1. Trung du miền núi phía Bắc', 'AI'): 0.55, ('1. Trung du miền núi phía Bắc', 'H'): 1.30,
        ('2. Đồng bằng sông Hồng', 'I'): 0.95, ('2. Đồng bằng sông Hồng', 'D'): 1.25, ('2. Đồng bằng sông Hồng', 'AI'): 1.40, ('2. Đồng bằng sông Hồng', 'H'): 1.05,
        ('3. Bắc Trung Bộ + DH Trung Bộ', 'I'): 1.05, ('3. Bắc Trung Bộ + DH Trung Bộ', 'D'): 0.95, ('3. Bắc Trung Bộ + DH Trung Bộ', 'AI'): 0.85, ('3. Bắc Trung Bộ + DH Trung Bộ', 'H'): 1.15,
        ('4. Tây Nguyên', 'I'): 1.20, ('4. Tây Nguyên', 'D'): 0.75, ('4. Tây Nguyên', 'AI'): 0.45, ('4. Tây Nguyên', 'H'): 1.35,
        ('5. Đông Nam Bộ', 'I'): 0.90, ('5. Đông Nam Bộ', 'D'): 1.30, ('5. Đông Nam Bộ', 'AI'): 1.55, ('5. Đông Nam Bộ', 'H'): 1.00,
        ('6. Đồng bằng sông Cửu Long', 'I'): 1.10, ('6. Đồng bằng sông Cửu Long', 'D'): 0.85, ('6. Đồng bằng sông Cửu Long', 'AI'): 0.65, ('6. Đồng bằng sông Cửu Long', 'H'): 1.25
    }
    
    # Chỉ số số hóa ban đầu D0 của 6 vùng
    D0 = {
        '1. Trung du miền núi phía Bắc': 38,
        '2. Đồng bằng sông Hồng': 78,
        '3. Bắc Trung Bộ + DH Trung Bộ': 55,
        '4. Tây Nguyên': 32,
        '5. Đông Nam Bộ': 82,
        '6. Đồng bằng sông Cửu Long': 48
    }
    
    with tab_model:
        st.markdown("### 🎛️ Cấu hình các tham số ràng buộc hệ thống")
        
        col1, col2 = st.columns(2)
        with col1:
            total_budget = st.number_input("Tổng ngân sách phân bổ tối đa (C1) - Tỷ VND", value=50000, step=1000)
            min_region_budget = st.number_input("Sàn ngân sách bắt buộc cho mỗi vùng (C2) - Tỷ VND", value=5000, step=500)
            max_region_budget = st.number_input("Trần ngân sách giới hạn cho mỗi vùng (C3) - Tỷ VND", value=12000, step=500)
        with col2:
            min_h_budget = st.number_input("Sàn tổng đầu tư phát triển nhân lực số H (C4) - Tỷ VND", value=12000, step=500)
            gamma = st.slider("Hệ số co giãn chuyển đổi số Gamma (γ) trong ràng buộc C5", 0.001, 0.010, 0.002, format="%.3f")
            lam = st.slider("Hệ số mức độ ưu tiên công bằng vùng miền Lambda (λ)", 0.0, 1.0, 0.9, step=0.05)
            
        enable_fairness = st.checkbox("Kích hoạt ràng buộc công bằng vùng miền (C5)", value=True)
        
        # Lưu tham số vào session state để đồng bộ dữ liệu tính toán giữa các tab
        st.session_state['total_budget'] = total_budget
        st.session_state['min_region_budget'] = min_region_budget
        st.session_state['max_region_budget'] = max_region_budget
        st.session_state['min_h_budget'] = min_h_budget
        st.session_state['gamma'] = gamma
        st.session_state['lam'] = lam
        st.session_state['enable_fairness'] = enable_fairness

    # PHẦN XỬ LÝ TOÁN TỐI ƯU HÓA BẰNG PULP SOLVER
    t_b = st.session_state.get('total_budget', 50000)
    m_r = st.session_state.get('min_region_budget', 5000)
    M_r = st.session_state.get('max_region_budget', 12000)
    m_h = st.session_state.get('min_h_budget', 12000)
    g = st.session_state.get('gamma', 0.002)
    l = st.session_state.get('lam', 0.9)
    f_on = st.session_state.get('enable_fairness', True)
    
    # Khởi tạo mô hình bài toán quy hoạch tuyến tính Maximize
    prob = pulp.LpProblem('VN_Digital_Budget_Optimization', pulp.LpMaximize)
    
    # Khai báo các biến quyết định (Ma trận x[r][j] >= 0)
    x = pulp.LpVariable.dicts('x', (regions, items), lowBound=0, cat='Continuous')
    
    # Biến phụ M để tuyến tính hóa hàm cực đại Max(D_r + gamma * x_D,r)
    M_var = pulp.LpVariable('D_max_bound', lowBound=0)
    
    # Định nghĩa hàm mục tiêu: Tối đa hóa GDP Gain tổng thể
    prob += pulp.lpSum(beta[(r, j)] * x[r][j] for r in regions for j in items), "Total_GDP_Gain"
    
    # Ràng buộc C1: Tổng ngân sách số quốc gia
    prob += pulp.lpSum(x[r][j] for r in regions for j in items) <= t_b, "C1_Total_Budget"
    
    # Ràng buộc C4: Sàn đầu tư ưu tiên phát triển nguồn nhân lực số
    prob += pulp.lpSum(x[r]['H'] for r in regions) >= m_h, "C4_Min_Human_Resource"
    
    # Áp đặt ràng buộc cục bộ trên từng vùng kinh tế
    for r in regions:
        # Ràng buộc C2 & C3: Giới hạn sàn và trần đầu tư để tránh tập trung cục bộ
        prob += pulp.lpSum(x[r][j] for j in items) >= m_r, f"C2_Min_Budget_{r}"
        prob += pulp.lpSum(x[r][j] for j in items) <= M_r, f"C3_Max_Budget_{r}"
        
        # Ràng buộc C5: Công bằng vùng miền (Linear hóa bằng biến chặn trên M_var)
        if f_on:
            prob += D0[r] + g * x[r]['D'] <= M_var, f"C5_Upper_Bound_{r}"
            prob += D0[r] + g * x[r]['D'] >= l * M_var, f"C5_Fairness_Constraint_{r}"

    # Thực thi giải thuật bằng solver CBC mặc định của PuLP
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Trích xuất dữ liệu sau khi tối ưu thành công
    status = pulp.LpStatus[prob.status]
    
    if status == 'Optimal':
        rows = []
        for r in regions:
            row_data = {'Vùng Kinh Tế - Xã Hội': r}
            total_allocated = 0
            for j in items:
                val = pulp.value(x[r][j])
                row_data[j] = round(val, 2)
                total_allocated += val
            row_data['Tổng Phân Bổ'] = round(total_allocated, 2)
            row_data['Chỉ số Số Hóa Mới'] = round(D0[r] + g * pulp.value(x[r]['D']), 2)
            rows.append(row_data)
            
        df_result = pd.DataFrame(rows)
        gdp_gain = pulp.value(prob.objective)
        
        with tab_result:
            st.success(f"🚀 **Trạng thái:** Tìm thấy phương án tối ưu thỏa mãn tất cả hệ ràng buộc toán học.")
            st.metric(label="Tổng GDP tăng thêm kỳ vọng (Z*)", value=f"{gdp_gain:,.2f} tỷ VND")
            
            st.markdown("#### Ma trận phân bổ chi tiết (Đơn vị: Tỷ VND)")
            st.dataframe(df_res = df_result.style.format({
                'I': '{:,.2f}', 'D': '{:,.2f}', 'AI': '{:,.2f}', 'H': '{:,.2f}', 
                'Tổng Phân Bổ': '{:,.2f}', 'Chỉ số Số Hóa Mới': '{:.2f}'
            }), use_container_width=True)
            
            # Tóm tắt phân bổ theo hạng mục đầu tư để đánh giá chính sách
            st.markdown("#### Cơ cấu đầu tư theo nhóm mục tiêu công nghệ")
            summary_items = {
                'Hạ tầng số (I)': sum(pulp.value(x[r]['I']) for r in regions),
                'Chuyển đổi số DN (D)': sum(pulp.value(x[r]['D']) for r in regions),
                'Năng lực AI': sum(pulp.value(x[r]['AI']) for r in regions),
                'Nhân lực số (H)': sum(pulp.value(x[r]['H']) for r in regions)
            }
            df_sum = pd.DataFrame(list(summary_items.items()), columns=['Hạng mục', 'Ngân sách (Tỷ VND)'])
            st.dataframe(df_sum.style.format({'Ngân sách (Tỷ VND)': '{:,.2f}'}), use_container_width=True)
            
        with tab_analysis:
            st.markdown("### 🗺️ Bản đồ nhiệt (Heatmap) ma trận phân bổ vốn tối ưu")
            df_heatmap = df_result.set_

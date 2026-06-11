# streamlit_app.py
import importlib
import streamlit as st


# =========================
# CẤU HÌNH TRANG
# =========================

st.set_page_config(
    page_title="AIDEOM-VN",
    page_icon="🇻🇳",
    layout="wide"
)


# =========================
# CSS GIAO DIỆN
# =========================

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #020617 100%);
            color: #f8fafc;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #111827 0%, #1e293b 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label {
            color: #f8fafc !important;
        }

        section[data-testid="stSidebar"] .stButton > button {
            width: 100%;
            justify-content: flex-start;
            text-align: left;
            border-radius: 12px;
            padding: 0.65rem 0.8rem;
            font-weight: 650;
            border: 1px solid rgba(148, 163, 184, 0.25);
        }

        section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
            background: rgba(15, 23, 42, 0.35);
            color: #f8fafc;
        }

        section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, #ef4444, #f97316);
            color: white;
            border: 1px solid rgba(248, 250, 252, 0.25);
        }

        h1, h2, h3 {
            color: #f8fafc !important;
            font-weight: 800 !important;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 100%;
        }

        .card {
            background: rgba(15, 23, 42, 0.86);
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 22px;
            padding: 24px;
            margin-bottom: 18px;
            box-shadow: 0 10px 35px rgba(0,0,0,0.28);
        }

        .pill {
            display: inline-block;
            padding: 6px 13px;
            border-radius: 999px;
            background: linear-gradient(90deg, #10b981, #06b6d4);
            color: white;
            font-weight: 700;
            font-size: 13px;
            margin-right: 8px;
            margin-bottom: 8px;
        }

        div[data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.95);
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 18px;
            padding: 18px;
        }

        .right-setting-title {
            background: rgba(15, 23, 42, 0.92);
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 18px;
            padding: 16px 16px 12px 16px;
            margin-bottom: 14px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.22);
        }

        .right-setting-title h3 {
            margin-top: 0;
            margin-bottom: 6px;
            font-size: 1.1rem;
        }

        .right-setting-title p {
            margin-bottom: 0;
            color: #cbd5e1;
            font-size: 0.88rem;
            line-height: 1.45;
        }

        div[data-testid="stVerticalBlock"] label {
            font-weight: 650 !important;
        }

        .difficulty-label {
            margin-top: 1rem;
            margin-bottom: 0.4rem;
            font-size: 0.92rem;
            font-weight: 800;
            color: #cbd5e1;
            letter-spacing: 0.02em;
        }

        .sidebar-subtitle {
            color: #cbd5e1;
            font-size: 0.92rem;
            line-height: 1.45;
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# DANH SÁCH MENU
# =========================

HOME_MENU = "🏠 Trang chủ"

BASIC_MENUS = [
    "🌱 Bài 1 — Cobb-Douglas + AI",
    "💰 Bài 2 — LP ngân sách số",
    "📊 Bài 3 — Priority 10 ngành",
    "🗺️ Bài 4 — LP ngành-vùng",
]

MEDIUM_MENUS = [
    "🎯 Bài 5 — MIP 15 dự án",
    "🏆 Bài 6 — TOPSIS 6 vùng",
    "🌐 Bài 7 — NSGA-II Pareto",
    "⏳ Bài 8 — Động 2026-2035",
]

ADVANCED_MENUS = [
    "👷 Bài 9 — Lao động & AI",
    "🎲 Bài 10 — Stochastic SP",
    "🤖 Bài 11 — Q-learning RL",
    "🧠 Bài 12 — AIDEOM tích hợp",
]


MODULE_ROUTES = {
    "🌱 Bài 1 — Cobb-Douglas + AI": [
        "modules.bai1_cobb_douglas",
        "bai01_cobb_douglas",
        "bai1_cobb_douglas",
    ],
    "💰 Bài 2 — LP ngân sách số": [
        "modules.bai2_lp_budget",
        "bai02_lp_phan_bo",
        "bai2_lp_budget",
    ],
    "📊 Bài 3 — Priority 10 ngành": [
        "modules.bai3_priority_sectors",
        "bai03_priority",
        "bai3_priority_sectors",
    ],
    "🗺️ Bài 4 — LP ngành-vùng": [
        "modules.bai4_lp_region_budget",
        "bai04_lp_nganh_vung",
        "bai4_lp_region_budget",
    ],
    "🎯 Bài 5 — MIP 15 dự án": [
        "modules.bai5_mip_project_selection",
        "bai05_mip_15_du_an",
        "bai5_mip_project_selection",
    ],
    "🏆 Bài 6 — TOPSIS 6 vùng": [
        "modules.bai6_topsis_ai_regions",
        "bai06_topsis_6_vung",
        "bai6_topsis_ai_regions",
    ],
    "🌐 Bài 7 — NSGA-II Pareto": [
        "modules.bai7_nsga2_pareto",
        "bai07_nsga2_pareto",
        "bai7_nsga2_pareto",
    ],
    "⏳ Bài 8 — Động 2026-2035": [
        "modules.bai8_dynamic_optimization",
        "bai08_dynamic_2026_2035",
        "bai8_dynamic_optimization",
    ],
    "👷 Bài 9 — Lao động & AI": [
        "modules.bai9_ai_labor_market",
        "bai09_lao_dong_ai",
        "bai9_ai_labor_market",
    ],
    "🎲 Bài 10 — Stochastic SP": [
        "modules.bai10_stochastic_programming",
        "bai10_stochastic_sp",
        "bai10_stochastic_programming",
    ],
    "🤖 Bài 11 — Q-learning RL": [
        "modules.bai11_qlearning_policy",
        "bai11_q_learning_rl",
        "bai11_qlearning_policy",
    ],
    "🧠 Bài 12 — AIDEOM tích hợp": [
        "modules.bai12_aideom_vn_integrated",
        "bai12_aideom_vn",
        "bai12_aideom_vn_integrated",
    ],
}


# =========================
# HÀM ĐỔI MENU
# =========================

def select_menu(menu_name: str):
    st.session_state["current_menu"] = menu_name


def sidebar_button(label: str, index: int):
    current_menu = st.session_state.get("current_menu", HOME_MENU)
    is_selected = current_menu == label

    st.sidebar.button(
        label,
        key=f"nav_button_{index}",
        use_container_width=True,
        type="primary" if is_selected else "secondary",
        on_click=select_menu,
        args=(label,),
    )


# =========================
# SIDEBAR CHỈ LÀ MENU
# =========================

if "current_menu" not in st.session_state:
    st.session_state["current_menu"] = HOME_MENU

st.sidebar.title("🇻🇳 AIDEOM-VN")
st.sidebar.markdown(
    """
    <div class="sidebar-subtitle">
        Mô hình ra quyết định phát triển kinh tế Việt Nam trong kỷ nguyên AI
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.divider()

st.sidebar.markdown("### Chọn bài")

sidebar_button(HOME_MENU, 0)

st.sidebar.markdown(
    '<div class="difficulty-label">🟢 Cơ bản — Bài 1 đến Bài 4</div>',
    unsafe_allow_html=True
)

for i, item in enumerate(BASIC_MENUS, start=1):
    sidebar_button(item, i)

st.sidebar.markdown(
    '<div class="difficulty-label">🟡 Trung bình — Bài 5 đến Bài 8</div>',
    unsafe_allow_html=True
)

for i, item in enumerate(MEDIUM_MENUS, start=10):
    sidebar_button(item, i)

st.sidebar.markdown(
    '<div class="difficulty-label">🔴 Nâng cao — Bài 9 đến Bài 12</div>',
    unsafe_allow_html=True
)

for i, item in enumerate(ADVANCED_MENUS, start=20):
    sidebar_button(item, i)

menu = st.session_state["current_menu"]


# =========================
# HÀM HIỂN THỊ PLACEHOLDER
# =========================

def module_dang_bo_sung(module_name: str):
    st.markdown(
        f"""
        <div class="card">
            <h1>🚧 {module_name}</h1>
            <p>
            Module này sẽ được bổ sung ở bước tiếp theo.
            Hiện tại app vẫn chạy bình thường và không bị lỗi thiếu file module.
            </p>
            <span class="pill">Đang phát triển</span>
            <span class="pill">Coming soon</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "Bạn có thể tiếp tục bổ sung file Python cho module này sau. "
        "Khi file module đã tồn tại, app sẽ tự gọi hàm render() hoặc run() trong module."
    )


# =========================
# HÀM GỌI MODULE AN TOÀN
# =========================

def render_module(module_paths, module_label: str):
    """
    module_paths: danh sách đường dẫn module có thể thử import.
    Ví dụ:
    ["modules.bai1_cobb_douglas", "bai01_cobb_douglas"]

    Điểm mới:
    - Sidebar trái chỉ dùng để chọn bài.
    - Các lệnh st.sidebar trong từng module sẽ được chuyển sang cột thiết lập bên phải.
    """

    main_col, setting_col = st.columns([4.6, 1.35], gap="large")

    with setting_col:
        st.markdown(
            """
            <div class="right-setting-title">
                <h3>⚙️ Thiết lập bài</h3>
                <p>
                    Các thông số, slider, lựa chọn và nút chạy của bài đang mở
                    sẽ hiển thị tại khu vực này.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    last_error = None
    original_sidebar = st.sidebar

    try:
        st.sidebar = setting_col

        with main_col:
            for module_path in module_paths:
                try:
                    module = importlib.import_module(module_path)

                    if hasattr(module, "render"):
                        module.render()
                        return

                    if hasattr(module, "run"):
                        module.run()
                        return

                    st.warning(
                        f"Đã tìm thấy module `{module_path}` nhưng chưa có hàm `render()` hoặc `run()`."
                    )
                    return

                except ModuleNotFoundError as e:
                    last_error = e
                    continue

                except Exception as e:
                    st.error(f"Module `{module_path}` có lỗi khi chạy.")
                    st.exception(e)
                    return

            module_dang_bo_sung(module_label)

            with st.expander("Chi tiết kỹ thuật"):
                st.write("App đã thử import các module sau nhưng chưa tìm thấy:")
                for p in module_paths:
                    st.code(p)
                if last_error:
                    st.caption(f"Lỗi cuối cùng: {last_error}")

    finally:
        st.sidebar = original_sidebar


# =========================
# TRANG CHỦ
# =========================

def render_home():
    st.markdown(
        """
        <div class="card">
            <h1>🇻🇳 VN AIDEOM-VN</h1>
            <h3>AI-Driven Decision Optimization Model for Vietnam</h3>
            <p>
            Web app giải 12 bài mô hình ra quyết định phát triển kinh tế Việt Nam
            trong kỷ nguyên AI, sử dụng Python và Streamlit.
            </p>
            <span class="pill">Decision Models</span>
            <span class="pill">Vietnam Economy</span>
            <span class="pill">AI</span>
            <span class="pill">Streamlit</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("GDP 2025", "514,0 tỷ USD", "+8,02%")
    col2.metric("Kinh tế số/GDP", "≈19,5%", "+1,2 điểm %")
    col3.metric("FDI 2025", "27,6 tỷ USD", "+8,9%")
    col4.metric("GDP/người 2025", "5.026 USD", "+6,9%")

    st.markdown("## 📌 Danh sách 12 module")

    st.markdown("### 🟢 Nhóm cơ bản")
    c1, c2, c3, c4 = st.columns(4)

    basic_cards = [
        ("🌱 Bài 1", "Cobb-Douglas mở rộng với AI và số hóa"),
        ("💰 Bài 2", "LP phân bổ ngân sách số"),
        ("📊 Bài 3", "Priority 10 ngành"),
        ("🗺️ Bài 4", "LP ngành-vùng"),
    ]

    for col, (title, desc) in zip([c1, c2, c3, c4], basic_cards):
        with col:
            st.markdown(
                f"""
                <div class="card">
                    <h3>{title}</h3>
                    <p>{desc}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("### 🟡 Nhóm trung bình")
    c1, c2, c3, c4 = st.columns(4)

    medium_cards = [
        ("🎯 Bài 5", "MIP lựa chọn 15 dự án"),
        ("🏆 Bài 6", "TOPSIS 6 vùng"),
        ("🌐 Bài 7", "NSGA-II Pareto"),
        ("⏳ Bài 8", "Tối ưu động 2026-2035"),
    ]

    for col, (title, desc) in zip([c1, c2, c3, c4], medium_cards):
        with col:
            st.markdown(
                f"""
                <div class="card">
                    <h3>{title}</h3>
                    <p>{desc}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("### 🔴 Nhóm nâng cao")
    c1, c2, c3, c4 = st.columns(4)

    advanced_cards = [
        ("👷 Bài 9", "Lao động và AI"),
        ("🎲 Bài 10", "Stochastic Programming"),
        ("🤖 Bài 11", "Q-learning RL"),
        ("🧠 Bài 12", "AIDEOM-VN tích hợp"),
    ]

    for col, (title, desc) in zip([c1, c2, c3, c4], advanced_cards):
        with col:
            st.markdown(
                f"""
                <div class="card">
                    <h3>{title}</h3>
                    <p>{desc}</p>
                </div>
                """,
                unsafe_allow_html=True
            )


# =========================
# ĐIỀU HƯỚNG TRANG
# =========================

if menu == HOME_MENU:
    render_home()

elif menu in MODULE_ROUTES:
    render_module(
        MODULE_ROUTES[menu],
        menu
    )

else:
    module_dang_bo_sung(menu)

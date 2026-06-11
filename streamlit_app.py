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
        /* =========================
           NỀN TỔNG THỂ
        ========================= */

        .stApp {
            background: #F8F1E3 !important;
            color: #053151 !important;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 100% !important;
        }

        /* =========================
           SIDEBAR TRÁI
        ========================= */

        section[data-testid="stSidebar"] {
            background: #053151 !important;
            border-right: 1px solid rgba(255,255,255,0.16);
        }

        section[data-testid="stSidebar"] * {
            color: #F8F1E3 !important;
        }

        section[data-testid="stSidebar"] hr {
            border-color: rgba(248, 241, 227, 0.25) !important;
        }

        section[data-testid="stSidebar"] .stButton > button {
            width: 100%;
            min-height: 48px;
            justify-content: flex-start;
            text-align: left;
            border-radius: 14px;
            padding: 0.65rem 0.9rem;
            font-weight: 700;
            font-size: 0.98rem;
            border: 1px solid rgba(248, 241, 227, 0.22);
            box-shadow: none;
            transition: all 0.18s ease;
        }

        section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
            background: rgba(248, 241, 227, 0.06) !important;
            color: #F8F1E3 !important;
        }

        section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
            background: rgba(228, 136, 55, 0.35) !important;
            border-color: #E48837 !important;
            transform: translateX(2px);
        }

        section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background: #E48837 !important;
            color: #ffffff !important;
            border: 1px solid #E48837 !important;
            box-shadow: 0 8px 18px rgba(228, 136, 55, 0.30);
        }

        .sidebar-subtitle {
            color: rgba(248, 241, 227, 0.82) !important;
            font-size: 0.92rem;
            line-height: 1.45;
            margin-bottom: 1rem;
        }

        .difficulty-label {
            margin-top: 1rem;
            margin-bottom: 0.45rem;
            font-size: 0.92rem;
            font-weight: 800;
            color: #F8F1E3 !important;
            letter-spacing: 0.02em;
        }

        /* =========================
           TEXT PHẦN GIẢI BÀI
        ========================= */

        h1, h2, h3, h4, h5, h6 {
            color: #053151 !important;
            font-weight: 850 !important;
        }

        p, li, span, label, div {
            color: inherit;
        }

        .stMarkdown, .stText, .stCaption {
            color: #053151 !important;
        }

        [data-testid="stMarkdownContainer"] {
            color: #053151 !important;
        }

        /* =========================
           CARD / KHỐI NỘI DUNG
        ========================= */

        .card {
            background: rgba(255, 255, 255, 0.58);
            border: 1px solid rgba(5, 49, 81, 0.16);
            border-radius: 22px;
            padding: 24px;
            margin-bottom: 18px;
            box-shadow: 0 10px 28px rgba(5, 49, 81, 0.10);
            color: #053151 !important;
        }

        .card h1,
        .card h2,
        .card h3,
        .card p {
            color: #053151 !important;
        }

        .pill {
            display: inline-block;
            padding: 6px 13px;
            border-radius: 999px;
            background: #E48837;
            color: white !important;
            font-weight: 700;
            font-size: 13px;
            margin-right: 8px;
            margin-bottom: 8px;
        }

        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.66);
            border: 1px solid rgba(5, 49, 81, 0.18);
            border-radius: 18px;
            padding: 18px;
            box-shadow: 0 8px 20px rgba(5, 49, 81, 0.08);
        }

        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] div {
            color: #053151 !important;
        }

        /* =========================
           TAB
        ========================= */

        button[data-baseweb="tab"] {
            color: #053151 !important;
            font-weight: 700 !important;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: #E48837 !important;
            border-bottom-color: #E48837 !important;
        }

        div[data-baseweb="tab-highlight"] {
            background-color: #E48837 !important;
        }

        /* =========================
           EXPANDER THAM SỐ TRONG BÀI
        ========================= */

        div[data-testid="stExpander"] {
            background: rgba(255, 255, 255, 0.58) !important;
            border: 1px solid rgba(5, 49, 81, 0.18) !important;
            border-radius: 18px !important;
            box-shadow: 0 8px 20px rgba(5, 49, 81, 0.08);
            margin-top: 12px;
            margin-bottom: 22px;
        }

        div[data-testid="stExpander"] summary {
            color: #053151 !important;
            font-weight: 800 !important;
        }

        div[data-testid="stExpander"] label,
        div[data-testid="stExpander"] p,
        div[data-testid="stExpander"] span,
        div[data-testid="stExpander"] div {
            color: #053151 !important;
        }

        /* =========================
           INPUT / SLIDER / SELECT
        ========================= */

        .stSlider label,
        .stNumberInput label,
        .stSelectbox label,
        .stCheckbox label,
        .stRadio label,
        .stTextInput label {
            color: #053151 !important;
            font-weight: 700 !important;
        }

        .stSlider [data-testid="stTickBar"] {
            color: #053151 !important;
        }

        div[data-baseweb="slider"] > div {
            color: #E48837 !important;
        }

        .stButton > button {
            background: #E48837 !important;
            color: #ffffff !important;
            border: 1px solid #E48837 !important;
            border-radius: 12px !important;
            font-weight: 750 !important;
        }

        .stButton > button:hover {
            background: #cf772d !important;
            border-color: #cf772d !important;
        }

        /* =========================
           DATAFRAME / TABLE
        ========================= */

        div[data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
        }

        table {
            color: #053151 !important;
        }

        /* =========================
           PLOTLY / CHART CONTAINER
        ========================= */

        div[data-testid="stPlotlyChart"],
        div[data-testid="stVegaLiteChart"],
        div[data-testid="stPyplot"] {
            background: rgba(255, 255, 255, 0.42);
            border-radius: 18px;
            padding: 8px;
            border: 1px solid rgba(5, 49, 81, 0.10);
        }

        /* =========================
           ALERT BOX
        ========================= */

        div[data-testid="stAlert"] {
            border-radius: 15px;
        }

        /* =========================
           MOBILE
        ========================= */

        @media (max-width: 900px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            h1 {
                font-size: 2rem !important;
                line-height: 1.22 !important;
            }

            h2 {
                font-size: 1.55rem !important;
            }

            h3 {
                font-size: 1.25rem !important;
            }
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

st.sidebar.title("VN AIDEOM-VN")
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

    Bản giao diện mới:
    - Sidebar trái chỉ còn menu.
    - Không còn cột tham số bên phải.
    - Nếu module cũ vẫn dùng st.sidebar.slider / st.sidebar.selectbox...
      thì app tự chuyển các widget đó vào một khối expander trong nội dung bài.
    """

    last_error = None
    original_sidebar = st.sidebar

    st.markdown(
        """
        <style>
            .legacy-param-note {
                background: rgba(228, 136, 55, 0.12);
                border: 1px solid rgba(228, 136, 55, 0.35);
                border-radius: 18px;
                padding: 14px 18px;
                margin-bottom: 18px;
                color: #053151 !important;
                font-weight: 650;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    legacy_param_box = st.expander(
        "⚙️ Tùy chỉnh tham số mô phỏng",
        expanded=False
    )

    try:
        # Chuyển toàn bộ st.sidebar trong các module cũ vào expander trên nội dung chính.
        # Nhờ vậy sidebar trái không bị lẫn slider/tham số nữa.
        st.sidebar = legacy_param_box

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

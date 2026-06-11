# streamlit_app.py
import importlib
import streamlit as st

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import plotly.io as pio
    pio.templates.default = "plotly_white"
except Exception:
    pio = None


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
        :root {
            --aideom-blue: #053151;
            --aideom-cream: #F8F1E3;
            --aideom-orange: #E48837;
            --aideom-white: #FFFFFF;
        }

        /* Nền app */
        .stApp {
            background: var(--aideom-cream) !important;
            color: var(--aideom-blue) !important;
        }

        /* Xóa phần header đen phía trên */
        header[data-testid="stHeader"] {
            background: var(--aideom-cream) !important;
            color: var(--aideom-blue) !important;
            box-shadow: none !important;
        }

        header[data-testid="stHeader"] * {
            color: var(--aideom-blue) !important;
        }

        div[data-testid="stToolbar"] {
            background: var(--aideom-cream) !important;
            color: var(--aideom-blue) !important;
        }

        div[data-testid="stToolbar"] * {
            color: var(--aideom-blue) !important;
        }

        div[data-testid="stDecoration"] {
            background: var(--aideom-cream) !important;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 100% !important;
        }

        /* Sidebar trái */
        section[data-testid="stSidebar"] {
            background: var(--aideom-blue) !important;
            border-right: 1px solid rgba(248, 241, 227, 0.24);
        }

        section[data-testid="stSidebar"] * {
            color: var(--aideom-cream) !important;
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
            font-weight: 750;
            font-size: 0.98rem;
            border: 1px solid rgba(248, 241, 227, 0.24);
            box-shadow: none;
            transition: all 0.18s ease;
        }

        section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
            background: rgba(248, 241, 227, 0.06) !important;
            color: var(--aideom-cream) !important;
        }

        section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
            background: rgba(228, 136, 55, 0.35) !important;
            border-color: var(--aideom-orange) !important;
            transform: translateX(2px);
        }

        section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background: var(--aideom-orange) !important;
            color: #ffffff !important;
            border: 1px solid var(--aideom-orange) !important;
            box-shadow: 0 8px 18px rgba(228, 136, 55, 0.32);
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
            font-weight: 850;
            color: var(--aideom-cream) !important;
            letter-spacing: 0.02em;
        }

        /* Text chính */
        section.main,
        section.main *,
        .main,
        .main *,
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] * {
            color: var(--aideom-blue) !important;
        }

        h1, h2, h3, h4, h5, h6 {
            color: var(--aideom-blue) !important;
            font-weight: 850 !important;
        }

        [data-testid="stMarkdownContainer"] {
            color: var(--aideom-blue) !important;
        }

        [data-testid="stCaptionContainer"] {
            color: rgba(5, 49, 81, 0.75) !important;
        }

        p, li, label, span {
            color: var(--aideom-blue) !important;
        }

        /* Highlight code như numpy, pandas, A_t */
        code {
            background: #111827 !important;
            color: #86efac !important;
            border-radius: 8px !important;
            padding: 0.15rem 0.42rem !important;
            font-weight: 750 !important;
        }

        pre,
        pre code {
            background: #111827 !important;
            color: #86efac !important;
            border-radius: 14px !important;
        }

        /* Card */
        .card {
            background: var(--aideom-white) !important;
            border: 1px solid rgba(5, 49, 81, 0.16);
            border-radius: 22px;
            padding: 24px;
            margin-bottom: 18px;
            box-shadow: 0 10px 28px rgba(5, 49, 81, 0.10);
            color: var(--aideom-blue) !important;
        }

        .card h1,
        .card h2,
        .card h3,
        .card p,
        .card span {
            color: var(--aideom-blue) !important;
        }

        .pill {
            display: inline-block;
            padding: 6px 13px;
            border-radius: 999px;
            background: var(--aideom-orange);
            color: white !important;
            font-weight: 750;
            font-size: 13px;
            margin-right: 8px;
            margin-bottom: 8px;
        }

        /* Metric */
        div[data-testid="stMetric"] {
            background: var(--aideom-white) !important;
            border: 1px solid rgba(5, 49, 81, 0.18);
            border-radius: 18px;
            padding: 18px;
            box-shadow: 0 8px 20px rgba(5, 49, 81, 0.08);
        }

        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] div {
            color: var(--aideom-blue) !important;
        }

        /* Tabs */
        button[data-baseweb="tab"] {
            color: var(--aideom-blue) !important;
            font-weight: 750 !important;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: var(--aideom-orange) !important;
            border-bottom-color: var(--aideom-orange) !important;
        }

        div[data-baseweb="tab-highlight"] {
            background-color: var(--aideom-orange) !important;
        }

        /* Bảng */
        div[data-testid="stDataFrame"],
        div[data-testid="stTable"],
        div[data-testid="stDataFrame"] div,
        div[data-testid="stTable"] div {
            background: var(--aideom-white) !important;
            color: var(--aideom-blue) !important;
        }

        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {
            border-radius: 18px !important;
            overflow: hidden !important;
            border: 1px solid rgba(5, 49, 81, 0.16) !important;
            box-shadow: 0 8px 20px rgba(5, 49, 81, 0.08);
            background: var(--aideom-white) !important;
        }

        table {
            background: var(--aideom-white) !important;
            color: var(--aideom-blue) !important;
            border-collapse: collapse !important;
        }

        thead,
        tbody,
        tr,
        th,
        td {
            background: var(--aideom-white) !important;
            color: var(--aideom-blue) !important;
            border-color: rgba(5, 49, 81, 0.16) !important;
        }

        th {
            font-weight: 850 !important;
            color: var(--aideom-blue) !important;
            background: var(--aideom-white) !important;
        }

        td {
            color: var(--aideom-blue) !important;
            background: var(--aideom-white) !important;
        }

        .stDataFrame,
        .stDataFrame * {
            background-color: var(--aideom-white) !important;
            color: var(--aideom-blue) !important;
        }

        /* Biểu đồ */
        div[data-testid="stPlotlyChart"],
        div[data-testid="stVegaLiteChart"],
        div[data-testid="stPyplot"] {
            background: var(--aideom-white) !important;
            border-radius: 18px;
            padding: 10px;
            border: 1px solid rgba(5, 49, 81, 0.12);
            box-shadow: 0 8px 20px rgba(5, 49, 81, 0.08);
        }

        div[data-testid="stPlotlyChart"] *,
        div[data-testid="stVegaLiteChart"] *,
        div[data-testid="stPyplot"] * {
            color: var(--aideom-blue) !important;
        }

        /* Khối tham số inline */
        div[data-testid="stExpander"] {
            background: var(--aideom-white) !important;
            border: 1px solid rgba(5, 49, 81, 0.18) !important;
            border-radius: 18px !important;
            box-shadow: 0 8px 20px rgba(5, 49, 81, 0.08);
            margin-top: 12px;
            margin-bottom: 22px;
        }

        div[data-testid="stExpander"] summary {
            color: var(--aideom-blue) !important;
            font-weight: 850 !important;
        }

        div[data-testid="stExpander"] label,
        div[data-testid="stExpander"] p,
        div[data-testid="stExpander"] span,
        div[data-testid="stExpander"] div {
            color: var(--aideom-blue) !important;
        }

        /* Input / slider / select */
        .stSlider label,
        .stNumberInput label,
        .stSelectbox label,
        .stCheckbox label,
        .stRadio label,
        .stTextInput label {
            color: var(--aideom-blue) !important;
            font-weight: 750 !important;
        }

        div[data-baseweb="input"],
        div[data-baseweb="select"] {
            background: var(--aideom-white) !important;
            color: var(--aideom-blue) !important;
            border-radius: 12px !important;
        }

        input {
            background: var(--aideom-white) !important;
            color: var(--aideom-blue) !important;
        }

        .stButton > button {
            background: var(--aideom-orange) !important;
            color: #ffffff !important;
            border: 1px solid var(--aideom-orange) !important;
            border-radius: 12px !important;
            font-weight: 800 !important;
        }

        .stButton > button * {
            color: #ffffff !important;
        }

        .stButton > button:hover {
            background: #cf772d !important;
            border-color: #cf772d !important;
        }

        div[data-testid="stAlert"] {
            border-radius: 15px;
            background: var(--aideom-white) !important;
            border: 1px solid rgba(5, 49, 81, 0.14) !important;
        }

        div[data-testid="stAlert"] * {
            color: var(--aideom-blue) !important;
        }

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
# STYLE BẢNG VÀ BIỂU ĐỒ
# =========================

def _style_dataframe_for_aideom(data):
    if pd is None:
        return data

    try:
        from pandas.io.formats.style import Styler

        if isinstance(data, Styler):
            return (
                data
                .set_properties(**{
                    "background-color": "#FFFFFF",
                    "color": "#053151",
                    "border-color": "rgba(5,49,81,0.16)",
                })
                .set_table_styles([
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#FFFFFF"),
                            ("color", "#053151"),
                            ("font-weight", "850"),
                            ("border-color", "rgba(5,49,81,0.16)"),
                        ],
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("background-color", "#FFFFFF"),
                            ("color", "#053151"),
                            ("border-color", "rgba(5,49,81,0.16)"),
                        ],
                    },
                    {
                        "selector": "table",
                        "props": [
                            ("background-color", "#FFFFFF"),
                            ("color", "#053151"),
                            ("border-collapse", "collapse"),
                        ],
                    },
                ])
            )

        if isinstance(data, pd.Series):
            data = data.to_frame()

        if isinstance(data, pd.DataFrame):
            return (
                data.style
                .set_properties(**{
                    "background-color": "#FFFFFF",
                    "color": "#053151",
                    "border-color": "rgba(5,49,81,0.16)",
                })
                .set_table_styles([
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#FFFFFF"),
                            ("color", "#053151"),
                            ("font-weight", "850"),
                            ("border-color", "rgba(5,49,81,0.16)"),
                        ],
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("background-color", "#FFFFFF"),
                            ("color", "#053151"),
                            ("border-color", "rgba(5,49,81,0.16)"),
                        ],
                    },
                    {
                        "selector": "table",
                        "props": [
                            ("background-color", "#FFFFFF"),
                            ("color", "#053151"),
                            ("border-collapse", "collapse"),
                        ],
                    },
                ])
            )

    except Exception:
        return data

    return data


_original_dataframe = st.dataframe
_original_table = st.table
_original_plotly_chart = st.plotly_chart


def _aideom_dataframe(data=None, *args, **kwargs):
    return _original_dataframe(_style_dataframe_for_aideom(data), *args, **kwargs)


def _aideom_table(data=None, *args, **kwargs):
    return _original_table(_style_dataframe_for_aideom(data), *args, **kwargs)


def _aideom_plotly_chart(figure_or_data=None, *args, **kwargs):
    fig = figure_or_data

    try:
        if hasattr(fig, "update_layout"):
            fig.update_layout(
                template="plotly_white",
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                font=dict(color="#053151"),
                title_font=dict(color="#053151"),
                legend=dict(
                    bgcolor="rgba(255,255,255,0)",
                    font=dict(color="#053151")
                ),
                margin=dict(l=40, r=30, t=60, b=40),
            )

            fig.update_xaxes(
                color="#053151",
                gridcolor="rgba(5,49,81,0.12)",
                zerolinecolor="rgba(5,49,81,0.18)",
                linecolor="rgba(5,49,81,0.30)",
            )

            fig.update_yaxes(
                color="#053151",
                gridcolor="rgba(5,49,81,0.12)",
                zerolinecolor="rgba(5,49,81,0.18)",
                linecolor="rgba(5,49,81,0.30)",
            )
    except Exception:
        pass

    return _original_plotly_chart(fig, *args, **kwargs)


st.dataframe = _aideom_dataframe
st.table = _aideom_table
st.plotly_chart = _aideom_plotly_chart


# =========================
# SIDEBAR INLINE PROXY
# =========================

class InlineSidebarProxy:
    """
    Chuyển các lệnh st.sidebar.xxx trong module con thành khối tham số inline.

    Ưu điểm:
    - Sidebar trái vẫn chỉ là menu.
    - Không tạo cột tham số bên phải.
    - Không tạo khối tham số to cố định trên đầu.
    - Widget sẽ xuất hiện tại vị trí module gọi st.sidebar.xxx.
    """

    def __init__(self):
        self._box = None

    def _ensure_box(self):
        if self._box is None:
            self._box = st.expander(
                "⚙️ Tùy chỉnh tham số cho phần này",
                expanded=False
            )
        return self._box

    def __enter__(self):
        box = self._ensure_box()
        return box.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        box = self._ensure_box()
        return box.__exit__(exc_type, exc_value, traceback)

    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            box = self._ensure_box()
            attr = getattr(box, name)
            return attr(*args, **kwargs)
        return wrapper


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
# PLACEHOLDER
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
# GỌI MODULE AN TOÀN
# =========================

def render_module(module_paths, module_label: str):
    last_error = None
    original_sidebar = st.sidebar

    try:
        st.sidebar = InlineSidebarProxy()

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

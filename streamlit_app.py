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
    page_title="VN AIDEOM-VN",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(
    """
    <style>
    /* Luôn giữ sidebar mở */
    section[data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
        transform: translateX(0px) !important;
        min-width: 320px !important;
        max-width: 320px !important;
        width: 320px !important;
        z-index: 999999 !important;
    }

    section[data-testid="stSidebar"] > div {
        width: 320px !important;
        min-width: 320px !important;
        max-width: 320px !important;
    }

    /* Ẩn nút thu gọn sidebar */
    button[title="Hide sidebar"],
    button[title="Show sidebar"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }

    /* Giữ nội dung chính không tràn sát sidebar */
    .block-container {
        padding-top: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# =========================
# MÀU CHỦ ĐẠO
# =========================
MAIN_TEXT = "#053151"
BG_COLOR = "#FFFAF1"
SIDEBAR_BG = "#053151"
SIDEBAR_TEXT = "#F8F1E3"
ACCENT = "#E48837"
WHITE = "#FFFFFF"


# =========================
# CSS GIAO DIỆN
# =========================
st.markdown(
    f"""
    <style>
        :root {{
            --main-text: {MAIN_TEXT};
            --bg-color: {BG_COLOR};
            --sidebar-bg: {SIDEBAR_BG};
            --sidebar-text: {SIDEBAR_TEXT};
            --accent: {ACCENT};
            --white: {WHITE};
        }}

        /* ===== NỀN APP ===== */
        .stApp {{
            background: var(--bg-color) !important;
            color: var(--main-text) !important;
        }}

        .block-container {{
            padding-top: 1.6rem !important;
            padding-bottom: 2.5rem !important;
            max-width: 100% !important;
        }}

        /* ===== HEADER TRÊN CÙNG ===== */
        header[data-testid="stHeader"] {{
            background: var(--bg-color) !important;
            color: var(--main-text) !important;
            box-shadow: none !important;
            border-bottom: none !important;
        }}

        header[data-testid="stHeader"] * {{
            color: var(--main-text) !important;
        }}

        div[data-testid="stToolbar"] {{
            background: var(--bg-color) !important;
        }}

        div[data-testid="stToolbar"] * {{
            color: var(--main-text) !important;
        }}

        div[data-testid="stDecoration"] {{
            background: var(--bg-color) !important;
        }}

        /* ===== SIDEBAR ===== */
        section[data-testid="stSidebar"] {{
            background: var(--sidebar-bg) !important;
            border-right: 1px solid rgba(248,241,227,0.20);
        }}

        section[data-testid="stSidebar"] * {{
            color: var(--sidebar-text) !important;
        }}

        section[data-testid="stSidebar"] .stButton > button {{
            width: 100%;
            min-height: 46px;
            justify-content: flex-start;
            text-align: left;
            border-radius: 12px;
            padding: 0.62rem 0.9rem;
            font-weight: 700;
            font-size: 0.97rem;
            border: 1px solid rgba(248,241,227,0.22);
            box-shadow: none;
            transition: all 0.18s ease;
            background: rgba(248,241,227,0.06) !important;
            color: var(--sidebar-text) !important;
        }}

        section[data-testid="stSidebar"] .stButton > button:hover {{
            background: rgba(228,136,55,0.22) !important;
            border-color: var(--accent) !important;
            transform: translateX(2px);
        }}

        section[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
            background: var(--accent) !important;
            color: #ffffff !important;
            border: 1px solid var(--accent) !important;
        }}

        .sidebar-subtitle {{
            color: rgba(248,241,227,0.85) !important;
            font-size: 0.92rem;
            line-height: 1.45;
            margin-bottom: 1rem;
        }}

        .difficulty-label {{
            margin-top: 1rem;
            margin-bottom: 0.45rem;
            font-size: 0.92rem;
            font-weight: 800;
            color: var(--sidebar-text) !important;
            letter-spacing: 0.02em;
        }}

        /* ===== TOÀN BỘ CHỮ ===== */
        html, body, p, div, span, label, li, h1, h2, h3, h4, h5, h6 {{
            color: var(--main-text);
        }}

        section.main,
        section.main *,
        .main,
        .main *,
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] * {{
            color: var(--main-text) !important;
        }}

        h1, h2, h3, h4, h5, h6 {{
            color: var(--main-text) !important;
            font-weight: 800 !important;
        }}

        [data-testid="stMarkdownContainer"] {{
            color: var(--main-text) !important;
        }}

        [data-testid="stCaptionContainer"] {{
            color: var(--main-text) !important;
            opacity: 0.9;
        }}

        /* ===== HIGHLIGHT / INLINE CODE =====
           Chỉ bôi đậm chữ, không có nền */
        code {{
            background: transparent !important;
            color: var(--main-text) !important;
            border-radius: 0 !important;
            padding: 0 !important;
            font-weight: 800 !important;
            border: none !important;
            box-shadow: none !important;
        }}

        pre {{
            background: transparent !important;
            color: var(--main-text) !important;
            border: 1px solid rgba(5,49,81,0.14) !important;
            border-radius: 12px !important;
            padding: 12px !important;
        }}

        pre code {{
            background: transparent !important;
            color: var(--main-text) !important;
            font-weight: 700 !important;
        }}

        /* ===== CARD TỐI GIẢN ===== */
        .card {{
            background: transparent !important;
            border: 1px solid rgba(5,49,81,0.14);
            border-radius: 18px;
            padding: 22px;
            margin-bottom: 18px;
            box-shadow: none !important;
        }}

        .card h1, .card h2, .card h3, .card p, .card span {{
            color: var(--main-text) !important;
        }}

        .pill {{
            display: none !important;
        }}

        /* ===== HOME CARD ĐỀU NHAU ===== */
        .home-module-card {{
            height: 200px !important;
            min-height: 200px !important;
            max-height: 200px !important;
            box-sizing: border-box !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: flex-start !important;
            align-items: flex-start !important;
            padding: 26px !important;
            background: rgba(255,255,255,0.35) !important;
        }}

        .home-module-card h3 {{
            font-size: 2rem !important;
            line-height: 1.15 !important;
            margin-top: 0 !important;
            margin-bottom: 1rem !important;
            white-space: nowrap !important;
            color: var(--main-text) !important;
        }}

        .home-module-card p {{
            font-size: 1.12rem !important;
            line-height: 1.55 !important;
            margin: 0 !important;
            color: var(--main-text) !important;
        }}

        /* ===== METRIC ===== */
        div[data-testid="stMetric"] {{
            background: transparent !important;
            border: 1px solid rgba(5,49,81,0.14);
            border-radius: 14px;
            padding: 16px;
            box-shadow: none !important;
        }}

        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"] {{
            color: var(--main-text) !important;
        }}

        [data-testid="stMetricValue"] > div {{
            color: var(--main-text) !important;
        }}

        /* ===== TAB ===== */
        button[data-baseweb="tab"] {{
            color: var(--main-text) !important;
            font-weight: 700 !important;
        }}

        button[data-baseweb="tab"][aria-selected="true"] {{
            color: var(--main-text) !important;
            border-bottom-color: var(--main-text) !important;
        }}

        div[data-baseweb="tab-highlight"] {{
            background-color: var(--main-text) !important;
        }}

        /* ===== DATAFRAME / TABLE ===== */
        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {{
            background: transparent !important;
            border: 1px solid rgba(5,49,81,0.14) !important;
            border-radius: 14px !important;
            overflow: hidden !important;
            box-shadow: none !important;
        }}

        table {{
            background: #ffffff !important;
            color: var(--main-text) !important;
            border-collapse: collapse !important;
        }}

        thead, tbody, tr, th, td {{
            background: #ffffff !important;
            color: var(--main-text) !important;
            border-color: rgba(5,49,81,0.14) !important;
        }}

        th {{
            font-weight: 800 !important;
            color: var(--main-text) !important;
        }}

        td {{
            font-weight: 500 !important;
            color: var(--main-text) !important;
        }}

        .stDataFrame,
        .stDataFrame * {{
            color: var(--main-text) !important;
        }}

        /* ===== CHART CONTAINER ===== */
        div[data-testid="stPlotlyChart"],
        div[data-testid="stVegaLiteChart"],
        div[data-testid="stPyplot"] {{
            background: transparent !important;
            border: 1px solid rgba(5,49,81,0.14);
            border-radius: 14px;
            padding: 8px;
            box-shadow: none !important;
        }}

        /* ===== EXPANDER THAM SỐ ===== */
        div[data-testid="stExpander"] {{
            background: transparent !important;
            border: 1px solid rgba(5,49,81,0.14) !important;
            border-radius: 14px !important;
            box-shadow: none !important;
            margin-top: 12px;
            margin-bottom: 18px;
        }}

        div[data-testid="stExpander"] summary {{
            color: var(--main-text) !important;
            font-weight: 800 !important;
        }}

        div[data-testid="stExpander"] * {{
            color: var(--main-text) !important;
        }}

        /* ===== INPUT / SLIDER ===== */
        .stSlider label,
        .stNumberInput label,
        .stSelectbox label,
        .stCheckbox label,
        .stRadio label,
        .stTextInput label {{
            color: var(--main-text) !important;
            font-weight: 700 !important;
        }}

        div[data-baseweb="input"],
        div[data-baseweb="select"] {{
            background: #ffffff !important;
            color: var(--main-text) !important;
            border-radius: 10px !important;
        }}

        input {{
            background: #ffffff !important;
            color: var(--main-text) !important;
        }}

        .stButton > button {{
            background: transparent !important;
            color: var(--main-text) !important;
            border: 1px solid rgba(5,49,81,0.22) !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
            box-shadow: none !important;
        }}

        .stButton > button:hover {{
            border-color: var(--main-text) !important;
            color: var(--main-text) !important;
        }}

        /* ===== ALERT / INFO / SUCCESS / WARNING =====
           Bỏ các nền màu */
        div[data-testid="stAlert"] {{
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
        }}

        div[data-testid="stAlert"] * {{
            color: var(--main-text) !important;
        }}

        /* ===== MOBILE ===== */
        @media (max-width: 900px) {{
            .block-container {{
                padding-left: 1rem;
                padding-right: 1rem;
            }}

            h1 {{
                font-size: 2rem !important;
                line-height: 1.2 !important;
            }}

            h2 {{
                font-size: 1.5rem !important;
            }}

            h3 {{
                font-size: 1.25rem !important;
            }}

            .home-module-card {{
                height: auto !important;
                min-height: 165px !important;
                max-height: none !important;
                padding: 22px !important;
            }}

            .home-module-card h3 {{
                font-size: 1.6rem !important;
                white-space: normal !important;
            }}

            .home-module-card p {{
                font-size: 1rem !important;
            }}
        }}
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# STYLE BẢNG
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
                    "color": MAIN_TEXT,
                    "border-color": "rgba(5,49,81,0.14)",
                })
                .set_table_styles([
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#FFFFFF"),
                            ("color", MAIN_TEXT),
                            ("font-weight", "800"),
                            ("border-color", "rgba(5,49,81,0.14)"),
                        ],
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("background-color", "#FFFFFF"),
                            ("color", MAIN_TEXT),
                            ("border-color", "rgba(5,49,81,0.14)"),
                        ],
                    },
                    {
                        "selector": "table",
                        "props": [
                            ("background-color", "#FFFFFF"),
                            ("color", MAIN_TEXT),
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
                    "color": MAIN_TEXT,
                    "border-color": "rgba(5,49,81,0.14)",
                })
                .set_table_styles([
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#FFFFFF"),
                            ("color", MAIN_TEXT),
                            ("font-weight", "800"),
                            ("border-color", "rgba(5,49,81,0.14)"),
                        ],
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("background-color", "#FFFFFF"),
                            ("color", MAIN_TEXT),
                            ("border-color", "rgba(5,49,81,0.14)"),
                        ],
                    },
                    {
                        "selector": "table",
                        "props": [
                            ("background-color", "#FFFFFF"),
                            ("color", MAIN_TEXT),
                            ("border-collapse", "collapse"),
                        ],
                    },
                ])
            )

    except Exception:
        return data

    return data


# =========================
# PATCH DATAFRAME / TABLE / CHART
# =========================
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
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=MAIN_TEXT),
                title_font=dict(color=MAIN_TEXT),
                legend=dict(
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(color=MAIN_TEXT)
                ),
                margin=dict(l=40, r=30, t=60, b=40),
            )

            fig.update_xaxes(
                color=MAIN_TEXT,
                tickfont=dict(color=MAIN_TEXT),
                title_font=dict(color=MAIN_TEXT),
                gridcolor="rgba(5,49,81,0.16)",
                zerolinecolor="rgba(5,49,81,0.20)",
                linecolor="rgba(5,49,81,0.30)",
            )

            fig.update_yaxes(
                color=MAIN_TEXT,
                tickfont=dict(color=MAIN_TEXT),
                title_font=dict(color=MAIN_TEXT),
                gridcolor="rgba(5,49,81,0.16)",
                zerolinecolor="rgba(5,49,81,0.20)",
                linecolor="rgba(5,49,81,0.30)",
            )

            if getattr(fig, "layout", None) and getattr(fig.layout, "annotations", None):
                for ann in fig.layout.annotations:
                    ann.font = dict(color=MAIN_TEXT)

        if hasattr(fig, "data"):
            for trace in fig.data:
                try:
                    trace.textfont = dict(color=MAIN_TEXT)
                except Exception:
                    pass

                try:
                    if getattr(trace, "type", "") == "table":
                        trace.header.font = dict(color=MAIN_TEXT)
                        trace.cells.font = dict(color=MAIN_TEXT)
                        trace.header.fill = dict(color="white")
                        trace.cells.fill = dict(color="white")
                    if getattr(trace, "type", "") == "indicator":
                        trace.number = dict(font=dict(color=MAIN_TEXT))
                        trace.title = dict(font=dict(color=MAIN_TEXT))
                except Exception:
                    pass
    except Exception:
        pass

    return _original_plotly_chart(fig, *args, **kwargs)


st.dataframe = _aideom_dataframe
st.table = _aideom_table
st.plotly_chart = _aideom_plotly_chart


# =========================
# INLINE SIDEBAR PROXY
# =========================
class InlineSidebarProxy:
    """
    Chuyển các lệnh st.sidebar.xxx trong module con
    thành khối tham số inline trong nội dung chính.
    """

    def __init__(self):
        self._box = None

    def _ensure_box(self):
        if self._box is None:
            self._box = st.expander(
                "Tùy chỉnh tham số",
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
# MENU
# =========================
HOME_MENU = "Trang chủ"

BASIC_MENUS = [
    "Bài 1 — Cobb-Douglas + AI",
    "Bài 2 — LP ngân sách số",
    "Bài 3 — Priority 10 ngành",
    "Bài 4 — LP ngành-vùng",
]

MEDIUM_MENUS = [
    "Bài 5 — MIP 15 dự án",
    "Bài 6 — TOPSIS 6 vùng",
    "Bài 7 — NSGA-II Pareto",
    "Bài 8 — Động 2026-2035",
]

ADVANCED_MENUS = [
    "Bài 9 — Lao động & AI",
    "Bài 10 — Stochastic SP",
    "Bài 11 — Q-learning RL",
    "Bài 12 — AIDEOM tích hợp",
]


MODULE_ROUTES = {
    "Bài 1 — Cobb-Douglas + AI": [
        "modules.bai1_cobb_douglas",
        "bai01_cobb_douglas",
        "bai1_cobb_douglas",
    ],
    "Bài 2 — LP ngân sách số": [
        "modules.bai2_lp_budget",
        "bai02_lp_phan_bo",
        "bai2_lp_budget",
    ],
    "Bài 3 — Priority 10 ngành": [
        "modules.bai3_priority_sectors",
        "bai03_priority",
        "bai3_priority_sectors",
    ],
    "Bài 4 — LP ngành-vùng": [
        "modules.bai4_lp_region_budget",
        "bai04_lp_nganh_vung",
        "bai4_lp_region_budget",
    ],
    "Bài 5 — MIP 15 dự án": [
        "modules.bai5_mip_project_selection",
        "bai05_mip_15_du_an",
        "bai5_mip_project_selection",
    ],
    "Bài 6 — TOPSIS 6 vùng": [
        "modules.bai6_topsis_ai_regions",
        "bai06_topsis_6_vung",
        "bai6_topsis_ai_regions",
    ],
    "Bài 7 — NSGA-II Pareto": [
        "modules.bai7_nsga2_pareto",
        "bai07_nsga2_pareto",
        "bai7_nsga2_pareto",
    ],
    "Bài 8 — Động 2026-2035": [
        "modules.bai8_dynamic_optimization",
        "bai08_dynamic_2026_2035",
        "bai8_dynamic_optimization",
    ],
    "Bài 9 — Lao động & AI": [
        "modules.bai9_ai_labor_market",
        "bai09_lao_dong_ai",
        "bai9_ai_labor_market",
    ],
    "Bài 10 — Stochastic SP": [
        "modules.bai10_stochastic_programming",
        "bai10_stochastic_sp",
        "bai10_stochastic_programming",
    ],
    "Bài 11 — Q-learning RL": [
        "modules.bai11_qlearning_policy",
        "bai11_q_learning_rl",
        "bai11_qlearning_policy",
    ],
    "Bài 12 — AIDEOM tích hợp": [
        "modules.bai12_aideom_vn_integrated",
        "bai12_aideom_vn",
        "bai12_aideom_vn_integrated",
    ],
}


# =========================
# HÀM CHỌN MENU
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
# SIDEBAR
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
    '<div class="difficulty-label">Cơ bản — Bài 1 đến Bài 4</div>',
    unsafe_allow_html=True
)
for i, item in enumerate(BASIC_MENUS, start=1):
    sidebar_button(item, i)

st.sidebar.markdown(
    '<div class="difficulty-label">Trung bình — Bài 5 đến Bài 8</div>',
    unsafe_allow_html=True
)
for i, item in enumerate(MEDIUM_MENUS, start=10):
    sidebar_button(item, i)

st.sidebar.markdown(
    '<div class="difficulty-label">Nâng cao — Bài 9 đến Bài 12</div>',
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
            <h1>{module_name}</h1>
            <p>
                Module này sẽ được bổ sung ở bước tiếp theo.
                Hiện tại app vẫn chạy bình thường và không bị lỗi thiếu file module.
            </p>
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
            <h1>VN AIDEOM-VN</h1>
            <h3>AI-Driven Decision Optimization Model for Vietnam</h3>
            <p>
                Web app giải 12 bài mô hình ra quyết định phát triển kinh tế Việt Nam
                trong kỷ nguyên AI, sử dụng Python và Streamlit.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("GDP 2025", "514,0 tỷ USD", "+8,02%")
    col2.metric("Kinh tế số/GDP", "≈19,5%", "+1,2 điểm %")
    col3.metric("FDI 2025", "27,6 tỷ USD", "+8,9%")
    col4.metric("GDP/người 2025", "5.026 USD", "+6,9%")

    st.markdown("## Danh sách 12 module")

    st.markdown("### Nhóm cơ bản")
    c1, c2, c3, c4 = st.columns(4)
    basic_cards = [
        ("Bài 1", "Cobb-Douglas mở rộng với AI và số hóa"),
        ("Bài 2", "LP phân bổ ngân sách số"),
        ("Bài 3", "Priority 10 ngành"),
        ("Bài 4", "LP ngành-vùng"),
    ]
    for col, (title, desc) in zip([c1, c2, c3, c4], basic_cards):
        with col:
            st.markdown(
                f"""
                <div class="card home-module-card">
                    <h3>{title}</h3>
                    <p>{desc}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("### Nhóm trung bình")
    c1, c2, c3, c4 = st.columns(4)
    medium_cards = [
        ("Bài 5", "MIP lựa chọn 15 dự án"),
        ("Bài 6", "TOPSIS 6 vùng"),
        ("Bài 7", "NSGA-II Pareto"),
        ("Bài 8", "Tối ưu động 2026-2035"),
    ]
    for col, (title, desc) in zip([c1, c2, c3, c4], medium_cards):
        with col:
            st.markdown(
                f"""
                <div class="card home-module-card">
                    <h3>{title}</h3>
                    <p>{desc}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("### Nhóm nâng cao")
    c1, c2, c3, c4 = st.columns(4)
    advanced_cards = [
        ("Bài 9", "Lao động và AI"),
        ("Bài 10", "Stochastic Programming"),
        ("Bài 11", "Q-learning RL"),
        ("Bài 12", "AIDEOM-VN tích hợp"),
    ]
    for col, (title, desc) in zip([c1, c2, c3, c4], advanced_cards):
        with col:
            st.markdown(
                f"""
                <div class="card home-module-card">
                    <h3>{title}</h3>
                    <p>{desc}</p>
                </div>
                """,
                unsafe_allow_html=True
            )


# =========================
# ĐIỀU HƯỚNG
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

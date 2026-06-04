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
           TOÀN BỘ NỀN APP
        ========================= */

        .stApp {
            background: #f6f8fb;
            color: #0f172a;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1280px;
        }

        /* =========================
           SIDEBAR
        ========================= */

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border-right: 1px solid #e5e7eb;
            box-shadow: 8px 0 24px rgba(15, 23, 42, 0.04);
        }

        section[data-testid="stSidebar"] * {
            color: #111827 !important;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #0f172a !important;
            font-weight: 800 !important;
        }

        section[data-testid="stSidebar"] .stCaption,
        section[data-testid="stSidebar"] small {
            color: #64748b !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 9px 11px;
            margin-bottom: 7px;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
            transition: all 0.18s ease;
        }

        section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
            border-color: #0f766e;
            background: #ecfdf5;
            transform: translateX(2px);
        }

        section[data-testid="stSidebar"] div[data-testid="stSelectbox"] {
            background: #ffffff;
            border-radius: 14px;
        }

        /* =========================
           TIÊU ĐỀ
        ========================= */

        h1 {
            color: #0f172a !important;
            font-weight: 850 !important;
            letter-spacing: -0.035em;
        }

        h2 {
            color: #1e293b !important;
            font-weight: 800 !important;
            letter-spacing: -0.02em;
        }

        h3 {
            color: #334155 !important;
            font-weight: 750 !important;
        }

        p, li, span, label {
            color: #334155;
        }

        /* =========================
           CARD CHÍNH
        ========================= */

        .card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 24px;
            padding: 26px;
            margin-bottom: 20px;
            box-shadow: 0 12px 35px rgba(15, 23, 42, 0.08);
        }

        .card h1,
        .card h2,
        .card h3 {
            color: #0f172a !important;
        }

        .card p {
            color: #475569 !important;
            font-size: 15.5px;
            line-height: 1.7;
        }

        /* =========================
           PILL / TAG
        ========================= */

        .pill {
            display: inline-block;
            padding: 7px 14px;
            border-radius: 999px;
            background: #e0f2fe;
            color: #0369a1 !important;
            font-weight: 750;
            font-size: 13px;
            margin-right: 8px;
            margin-bottom: 8px;
            border: 1px solid #bae6fd;
        }

        /* =========================
           METRIC BOX
        ========================= */

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 20px;
            padding: 18px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        }

        div[data-testid="stMetricLabel"] {
            color: #64748b !important;
            font-weight: 600;
        }

        div[data-testid="stMetricValue"] {
            color: #0f172a !important;
            font-weight: 850;
        }

        div[data-testid="stMetricDelta"] {
            color: #0f766e !important;
            font-weight: 700;
        }

        /* =========================
           TABS
        ========================= */

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 1px solid #e5e7eb;
        }

        .stTabs [data-baseweb="tab"] {
            background: #ffffff;
            border-radius: 999px;
            padding: 10px 18px;
            border: 1px solid #e5e7eb;
            color: #334155 !important;
            font-weight: 700;
        }

        .stTabs [aria-selected="true"] {
            background: #0f766e !important;
            color: #ffffff !important;
            border-color: #0f766e !important;
        }

        /* =========================
           BẢNG DATAFRAME
        ========================= */

        div[data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid #e5e7eb;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        }

        /* =========================
           BUTTON
        ========================= */

        .stButton > button {
            border-radius: 14px;
            border: 1px solid #d1d5db;
            background: #ffffff;
            color: #111827;
            font-weight: 700;
            padding: 0.65rem 1rem;
            transition: all 0.18s ease;
        }

        .stButton > button:hover {
            border-color: #0f766e;
            color: #0f766e;
            background: #ecfdf5;
            transform: translateY(-1px);
        }

        /* =========================
           INFO / WARNING / ERROR BOX
        ========================= */

        div[data-testid="stAlert"] {
            border-radius: 16px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
        }

        /* =========================
           INPUTS
        ========================= */

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {
            border-radius: 14px !important;
            border-color: #d1d5db !important;
            background-color: #ffffff !important;
        }

        input {
            color: #111827 !important;
        }

        /* =========================
           PLOTLY / CHART KHUNG
        ========================= */

        div[data-testid="stPlotlyChart"] {
            background: #ffffff;
            border-radius: 20px;
            padding: 10px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        }

        /* =========================
           DIVIDER
        ========================= */

        hr {
            border-color: #e5e7eb !important;
        }

    </style>
    """,
    unsafe_allow_html=True
)


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

    Hàm này giúp tránh lỗi ModuleNotFoundError.
    Nếu chưa có module, app vẫn chạy và hiện thông báo bổ sung sau.
    """

    last_error = None

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
        st.write(
            "App đã thử import các module sau nhưng chưa tìm thấy:"
        )
        for p in module_paths:
            st.code(p)
        if last_error:
            st.caption(f"Lỗi cuối cùng: {last_error}")


# =========================
# SIDEBAR
# =========================

# =========================
# SIDEBAR MENU HIỂN THỊ TOÀN BỘ THEO NHÓM
# =========================

MODULE_GROUPS = [
    {
        "title": "🟢 Nhóm Dễ",
        "subtitle": "Bài 1 đến Bài 3",
        "modules": [
            {
                "key": "bai01",
                "label": "🌱 Bài 1 — Cobb-Douglas + AI",
                "desc": "TFP, tăng trưởng, dự báo GDP 2030",
                "paths": [
                    "bai01_cobb_douglas",
                    "bai1_cobb_douglas",
                    "modules.bai1_cobb_douglas",
                ],
            },
            {
                "key": "bai02",
                "label": "💰 Bài 2 — LP ngân sách số",
                "desc": "Phân bổ ngân sách số bằng LP",
                "paths": [
                    "bai02_lp_phan_bo",
                    "bai2_lp_budget",
                    "modules.bai2_lp_budget",
                ],
            },
            {
                "key": "bai03",
                "label": "📊 Bài 3 — Priority 10 ngành",
                "desc": "Chỉ số ưu tiên ngành và phân tích trọng số",
                "paths": [
                    "bai03_priority",
                    "bai3_priority_sectors",
                    "modules.bai3_priority_sectors",
                ],
            },
        ],
    },
    {
        "title": "🟡 Nhóm Trung bình",
        "subtitle": "Bài 4 đến Bài 6",
        "modules": [
            {
                "key": "bai04",
                "label": "🗺️ Bài 4 — LP ngành-vùng",
                "desc": "Phân bổ ngân sách theo vùng và hạng mục",
                "paths": [
                    "bai04_lp_nganh_vung",
                    "bai4_lp_region_budget",
                    "modules.bai4_lp_region_budget",
                ],
            },
            {
                "key": "bai05",
                "label": "🎯 Bài 5 — MIP 15 dự án",
                "desc": "Lựa chọn dự án chuyển đổi số bằng MIP",
                "paths": [
                    "bai05_mip_15_du_an",
                    "bai5_mip_project_selection",
                    "modules.bai5_mip_project_selection",
                ],
            },
            {
                "key": "bai06",
                "label": "🏆 Bài 6 — TOPSIS 6 vùng",
                "desc": "Xếp hạng vùng ưu tiên đầu tư AI",
                "paths": [
                    "bai06_topsis_6_vung",
                    "bai6_topsis_ai_regions",
                    "modules.bai6_topsis_ai_regions",
                ],
            },
        ],
    },
    {
        "title": "🟠 Nhóm Khá khó",
        "subtitle": "Bài 7 đến Bài 9",
        "modules": [
            {
                "key": "bai07",
                "label": "🌐 Bài 7 — NSGA-II Pareto",
                "desc": "Tối ưu đa mục tiêu và đường biên Pareto",
                "paths": [
                    "bai07_nsga2_pareto",
                    "bai7_nsga2_pareto",
                    "modules.bai7_nsga2_pareto",
                ],
            },
            {
                "key": "bai08",
                "label": "⏳ Bài 8 — Động 2026-2035",
                "desc": "Tối ưu động phân bổ liên thời gian",
                "paths": [
                    "bai08_dynamic_2026_2035",
                    "bai8_dynamic_optimization",
                    "modules.bai8_dynamic_optimization",
                ],
            },
            {
                "key": "bai09",
                "label": "👷 Bài 9 — Lao động & AI",
                "desc": "Mô phỏng tác động AI tới thị trường lao động",
                "paths": [
                    "bai09_lao_dong_ai",
                    "bai9_ai_labor_market",
                    "modules.bai9_ai_labor_market",
                ],
            },
        ],
    },
    {
        "title": "🔴 Nhóm Khó",
        "subtitle": "Bài 10 đến Bài 12",
        "modules": [
            {
                "key": "bai10",
                "label": "🎲 Bài 10 — Stochastic SP",
                "desc": "Quy hoạch ngẫu nhiên hai giai đoạn",
                "paths": [
                    "bai10_stochastic_sp",
                    "bai10_stochastic_programming",
                    "modules.bai10_stochastic_programming",
                ],
            },
            {
                "key": "bai11",
                "label": "🤖 Bài 11 — Q-learning RL",
                "desc": "Học tăng cường cho chính sách thích nghi",
                "paths": [
                    "bai11_q_learning_rl",
                    "bai11_qlearning_policy",
                    "modules.bai11_qlearning_policy",
                ],
            },
            {
                "key": "bai12",
                "label": "🧠 Bài 12 — AIDEOM tích hợp",
                "desc": "Dashboard tích hợp AIDEOM-VN",
                "paths": [
                    "bai12_aideom_vn",
                    "bai12_aideom_vn_integrated",
                    "modules.bai12_aideom_vn_integrated",
                ],
            },
        ],
    },
]


# =========================
# KHỞI TẠO TRẠNG THÁI MENU
# =========================

if "selected_page" not in st.session_state:
    st.session_state.selected_page = "home"

if "selected_label" not in st.session_state:
    st.session_state.selected_label = "🏠 Trang chủ"

if "selected_paths" not in st.session_state:
    st.session_state.selected_paths = []


def set_page(page_key, label, paths):
    st.session_state.selected_page = page_key
    st.session_state.selected_label = label
    st.session_state.selected_paths = paths

# =========================
# TRANG CHỦ
# =========================

def show_home_page():
    st.markdown(
        """
        <div class="card">
            <h1>🇻🇳 VN AIDEOM-VN</h1>
            <h3>AI-Driven Decision Optimization Model for Vietnam</h3>
            <p>
            Web app giải 12 bài mô hình ra quyết định phát triển kinh tế Việt Nam
            trong kỷ nguyên AI, sử dụng Python, Streamlit và các mô hình tối ưu hóa.
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

    st.markdown("## 📚 Cấu trúc 12 bài tập")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            """
            <div class="card">
                <h3>🟢 Nhóm Dễ</h3>
                <h4>Bài 1–3</h4>
                <p>
                Hàm sản xuất Cobb-Douglas, quy hoạch tuyến tính đơn giản,
                và chỉ số ưu tiên ngành.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="card">
                <h3>🟡 Nhóm Trung bình</h3>
                <h4>Bài 4–6</h4>
                <p>
                LP ngành-vùng, MIP lựa chọn dự án và TOPSIS xếp hạng vùng.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class="card">
                <h3>🟠 Nhóm Khá khó</h3>
                <h4>Bài 7–9</h4>
                <p>
                NSGA-II Pareto, tối ưu động và mô phỏng thị trường lao động AI.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            """
            <div class="card">
                <h3>🔴 Nhóm Khó</h3>
                <h4>Bài 10–12</h4>
                <p>
                Stochastic Programming, Q-learning và dashboard tích hợp AIDEOM-VN.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("## 🧭 Hướng dẫn sử dụng")

    st.info(
        "Ở sidebar bên trái, chọn bài theo từng nhóm từ dễ đến khó. "
        "Các bài đã có file module `.py` sẽ hiển thị phần giải chi tiết. "
        "Các bài chưa có module sẽ hiện thông báo bổ sung sau."
    )
# =========================
# SIDEBAR HIỂN THỊ TOÀN BỘ BÀI
# =========================

with st.sidebar:
    st.markdown("## 🇻🇳 VN AIDEOM-VN")
    st.caption("Mô hình ra quyết định phát triển kinh tế Việt Nam trong kỷ nguyên AI")
    st.divider()

    if st.button("🏠 Trang chủ", key="nav_home"):
        set_page("home", "🏠 Trang chủ", [])

    st.markdown(
        """
        <div style="
            margin-top: 12px;
            margin-bottom: 12px;
            font-size: 13px;
            color: #64748b;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        ">
            Danh sách bài tập
        </div>
        """,
        unsafe_allow_html=True
    )

    for group in MODULE_GROUPS:
        st.markdown(
            f"""
            <div style="
                margin-top: 18px;
                margin-bottom: 6px;
                padding: 10px 12px;
                background: #f1f5f9;
                border: 1px solid #e2e8f0;
                border-radius: 14px;
            ">
                <div style="
                    font-size: 15px;
                    font-weight: 850;
                    color: #0f172a;
                ">
                    {group["title"]}
                </div>
                <div style="
                    font-size: 12.5px;
                    font-weight: 600;
                    color: #64748b;
                    margin-top: 2px;
                ">
                    {group["subtitle"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        for module in group["modules"]:
            is_active = st.session_state.selected_page == module["key"]

            button_label = module["label"]

            if is_active:
                st.markdown(
                    f"""
                    <div style="
                        margin: 6px 0 4px 0;
                        padding: 9px 11px;
                        background: #0f766e;
                        color: white;
                        border-radius: 13px;
                        font-weight: 800;
                        font-size: 14px;
                        border: 1px solid #0f766e;
                    ">
                        {button_label}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                if st.button(button_label, key=f"nav_{module['key']}"):
                    set_page(
                        module["key"],
                        module["label"],
                        module["paths"]
                    )

            st.caption(module["desc"])

    st.divider()

    st.markdown(
        f"""
        <div style="
            padding: 12px;
            background: #ecfdf5;
            border: 1px solid #bbf7d0;
            border-radius: 14px;
            font-size: 13px;
            color: #065f46;
            line-height: 1.5;
        ">
            <b>Đang chọn:</b><br>
            {st.session_state.selected_label}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption("Dữ liệu: NSO, MoST, MIC, MPI, WB, GII")


# =========================
# ĐIỀU HƯỚNG NỘI DUNG CHÍNH
# =========================

if st.session_state.selected_page == "home":
    show_home_page()
else:
    render_module(
        st.session_state.selected_paths,
        st.session_state.selected_label
    )

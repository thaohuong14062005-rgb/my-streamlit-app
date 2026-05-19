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

        h1, h2, h3 {
            color: #f8fafc !important;
            font-weight: 800 !important;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
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

st.sidebar.title("🇻🇳 AIDEOM-VN")
st.sidebar.caption(
    "Mô hình ra quyết định phát triển kinh tế Việt Nam trong kỷ nguyên AI"
)

menu = st.sidebar.radio(
    "Chọn bài",
    [
        "🏠 Trang chủ",
        "🌱 Bài 1 — Cobb-Douglas + AI",
        "💰 Bài 2 — LP ngân sách số",
        "📊 Bài 3 — Priority 10 ngành",
        "🗺️ Bài 4 — LP ngành-vùng",
        "🎯 Bài 5 — MIP 15 dự án",
        "🏆 Bài 6 — TOPSIS 6 vùng",
        "🌐 Bài 7 — NSGA-II Pareto",
        "⏳ Bài 8 — Động 2026-2035",
        "👷 Bài 9 — Lao động & AI",
        "🎲 Bài 10 — Stochastic SP",
        "🤖 Bài 11 — Q-learning RL",
        "🧠 Bài 12 — AIDEOM tích hợp",
    ]
)


# =========================
# TRANG CHỦ
# =========================

if menu == "🏠 Trang chủ":
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

    modules = [
        ("🌱 Bài 1", "Cobb-Douglas mở rộng với AI và số hóa"),
        ("💰 Bài 2", "LP phân bổ ngân sách số"),
        ("📊 Bài 3", "Priority 10 ngành"),
        ("🗺️ Bài 4", "LP ngành-vùng"),
        ("🎯 Bài 5", "MIP lựa chọn 15 dự án"),
        ("🏆 Bài 6", "TOPSIS 6 vùng"),
        ("🌐 Bài 7", "NSGA-II Pareto"),
        ("⏳ Bài 8", "Tối ưu động 2026-2035"),
        ("👷 Bài 9", "Lao động và AI"),
        ("🎲 Bài 10", "Stochastic Programming"),
        ("🤖 Bài 11", "Q-learning RL"),
        ("🧠 Bài 12", "AIDEOM-VN tích hợp"),
    ]

    c1, c2, c3 = st.columns(3)

    for i, (title, desc) in enumerate(modules):
        with [c1, c2, c3][i % 3]:
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
# GỌI MODULE AN TOÀN
# =========================

elif menu == "🌱 Bài 1 — Cobb-Douglas + AI":
    render_module(
        [
            "modules.bai1_cobb_douglas",
            "bai01_cobb_douglas",
            "bai1_cobb_douglas",
        ],
        menu
    )

elif menu == "💰 Bài 2 — LP ngân sách số":
    render_module(
        [
            "modules.bai2_lp_budget",
            "bai02_lp_phan_bo",
            "bai2_lp_budget",
        ],
        menu
    )

elif menu == "📊 Bài 3 — Priority 10 ngành":
    render_module(
        [
            "modules.bai3_priority_sectors",
            "bai03_priority",
            "bai3_priority_sectors",
        ],
        menu
    )

elif menu == "🗺️ Bài 4 — LP ngành-vùng":
    render_module(
        [
            "modules.bai4_lp_region_budget",
            "bai04_lp_nganh_vung",
            "bai4_lp_region_budget",
        ],
        menu
    )

elif menu == "🎯 Bài 5 — MIP 15 dự án":
    render_module(
        [
            "modules.bai5_mip_project_selection",
            "bai05_mip_15_du_an",
            "bai5_mip_project_selection",
        ],
        menu
    )

elif menu == "🏆 Bài 6 — TOPSIS 6 vùng":
    render_module(
        [
            "modules.bai6_topsis_ai_regions",
            "bai06_topsis_6_vung",
            "bai6_topsis_ai_regions",
        ],
        menu
    )

elif menu == "🌐 Bài 7 — NSGA-II Pareto":
    render_module(
        [
            "modules.bai7_nsga2_pareto",
            "bai07_nsga2_pareto",
            "bai7_nsga2_pareto",
        ],
        menu
    )

elif menu == "⏳ Bài 8 — Động 2026-2035":
    render_module(
        [
            "modules.bai8_dynamic_optimization",
            "bai08_dynamic_2026_2035",
            "bai8_dynamic_optimization",
        ],
        menu
    )

elif menu == "👷 Bài 9 — Lao động & AI":
    render_module(
        [
            "modules.bai9_ai_labor_market",
            "bai09_lao_dong_ai",
            "bai9_ai_labor_market",
        ],
        menu
    )

elif menu == "🎲 Bài 10 — Stochastic SP":
    render_module(
        [
            "modules.bai10_stochastic_programming",
            "bai10_stochastic_sp",
            "bai10_stochastic_programming",
        ],
        menu
    )

elif menu == "🤖 Bài 11 — Q-learning RL":
    render_module(
        [
            "modules.bai11_qlearning_policy",
            "bai11_q_learning_rl",
            "bai11_qlearning_policy",
        ],
        menu
    )

elif menu == "🧠 Bài 12 — AIDEOM tích hợp":
    render_module(
        [
            "modules.bai12_aideom_vn_integrated",
            "bai12_aideom_vn",
            "bai12_aideom_vn_integrated",
        ],
        menu
    )

else:
    module_dang_bo_sung(menu)

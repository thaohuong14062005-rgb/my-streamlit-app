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

# =========================
# SIDEBAR MENU CHIA NHÓM
# =========================

MODULE_GROUPS = {
    "🟢 Nhóm Dễ — Bài 1 đến Bài 3": {
        "bai01": {
            "label": "🌱 Bài 1 — Cobb-Douglas + AI",
            "desc": "Hàm sản xuất, TFP, dự báo GDP 2030",
            "paths": [
                "bai01_cobb_douglas",
                "bai1_cobb_douglas",
                "modules.bai1_cobb_douglas",
            ],
        },
        "bai02": {
            "label": "💰 Bài 2 — LP ngân sách số",
            "desc": "Phân bổ ngân sách số bằng Linear Programming",
            "paths": [
                "bai02_lp_phan_bo",
                "bai2_lp_budget",
                "modules.bai2_lp_budget",
            ],
        },
        "bai03": {
            "label": "📊 Bài 3 — Priority 10 ngành",
            "desc": "Chỉ số ưu tiên ngành, min-max, phân tích trọng số",
            "paths": [
                "bai03_priority",
                "bai3_priority_sectors",
                "modules.bai3_priority_sectors",
            ],
        },
    },

    "🟡 Nhóm Trung bình — Bài 4 đến Bài 6": {
        "bai04": {
            "label": "🗺️ Bài 4 — LP ngành-vùng",
            "desc": "Phân bổ ngân sách theo vùng và hạng mục đầu tư",
            "paths": [
                "bai04_lp_nganh_vung",
                "bai4_lp_region_budget",
                "modules.bai4_lp_region_budget",
            ],
        },
        "bai05": {
            "label": "🎯 Bài 5 — MIP 15 dự án",
            "desc": "Lựa chọn dự án chuyển đổi số bằng MIP",
            "paths": [
                "bai05_mip_15_du_an",
                "bai5_mip_project_selection",
                "modules.bai5_mip_project_selection",
            ],
        },
        "bai06": {
            "label": "🏆 Bài 6 — TOPSIS 6 vùng",
            "desc": "Xếp hạng vùng ưu tiên đầu tư AI",
            "paths": [
                "bai06_topsis_6_vung",
                "bai6_topsis_ai_regions",
                "modules.bai6_topsis_ai_regions",
            ],
        },
    },

    "🟠 Nhóm Khá khó — Bài 7 đến Bài 9": {
        "bai07": {
            "label": "🌐 Bài 7 — NSGA-II Pareto",
            "desc": "Tối ưu đa mục tiêu và đường biên Pareto",
            "paths": [
                "bai07_nsga2_pareto",
                "bai7_nsga2_pareto",
                "modules.bai7_nsga2_pareto",
            ],
        },
        "bai08": {
            "label": "⏳ Bài 8 — Động 2026-2035",
            "desc": "Tối ưu động phân bổ liên thời gian",
            "paths": [
                "bai08_dynamic_2026_2035",
                "bai8_dynamic_optimization",
                "modules.bai8_dynamic_optimization",
            ],
        },
        "bai09": {
            "label": "👷 Bài 9 — Lao động & AI",
            "desc": "Mô phỏng tác động AI tới thị trường lao động",
            "paths": [
                "bai09_lao_dong_ai",
                "bai9_ai_labor_market",
                "modules.bai9_ai_labor_market",
            ],
        },
    },

    "🔴 Nhóm Khó — Bài 10 đến Bài 12": {
        "bai10": {
            "label": "🎲 Bài 10 — Stochastic SP",
            "desc": "Quy hoạch ngẫu nhiên hai giai đoạn",
            "paths": [
                "bai10_stochastic_sp",
                "bai10_stochastic_programming",
                "modules.bai10_stochastic_programming",
            ],
        },
        "bai11": {
            "label": "🤖 Bài 11 — Q-learning RL",
            "desc": "Học tăng cường cho chính sách kinh tế thích nghi",
            "paths": [
                "bai11_q_learning_rl",
                "bai11_qlearning_policy",
                "modules.bai11_qlearning_policy",
            ],
        },
        "bai12": {
            "label": "🧠 Bài 12 — AIDEOM tích hợp",
            "desc": "Dashboard tích hợp toàn bộ mô hình AIDEOM-VN",
            "paths": [
                "bai12_aideom_vn",
                "bai12_aideom_vn_integrated",
                "modules.bai12_aideom_vn_integrated",
            ],
        },
    },
}


def show_home_page():
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

    st.markdown("## 📚 Cấu trúc 12 bài")

    groups_desc = [
        {
            "title": "🟢 Nhóm Dễ",
            "subtitle": "Bài 1–3",
            "content": "Cobb-Douglas, LP đơn giản, chỉ số Priority ngành.",
        },
        {
            "title": "🟡 Nhóm Trung bình",
            "subtitle": "Bài 4–6",
            "content": "LP ngành-vùng, MIP lựa chọn dự án, TOPSIS vùng.",
        },
        {
            "title": "🟠 Nhóm Khá khó",
            "subtitle": "Bài 7–9",
            "content": "NSGA-II Pareto, tối ưu động, mô phỏng lao động AI.",
        },
        {
            "title": "🔴 Nhóm Khó",
            "subtitle": "Bài 10–12",
            "content": "Stochastic Programming, Q-learning, dashboard tích hợp.",
        },
    ]

    cols = st.columns(4)

    for i, group in enumerate(groups_desc):
        with cols[i]:
            st.markdown(
                f"""
                <div class="card">
                    <h3>{group["title"]}</h3>
                    <h4>{group["subtitle"]}</h4>
                    <p>{group["content"]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )


# =========================
# HIỂN THỊ SIDEBAR
# =========================

with st.sidebar:
    st.markdown("## 🇻🇳 VN AIDEOM-VN")
    st.caption("Mô hình ra quyết định phát triển kinh tế Việt Nam trong kỷ nguyên AI")
    st.divider()

    main_page = st.radio(
        "Điều hướng",
        [
            "🏠 Trang chủ",
            "📚 Chọn nhóm bài",
        ],
        index=0
    )

    selected_key = "home"
    selected_label = "🏠 Trang chủ"
    selected_paths = []

    if main_page == "📚 Chọn nhóm bài":
        selected_group = st.selectbox(
            "Chọn cấp độ",
            list(MODULE_GROUPS.keys())
        )

        group_modules = MODULE_GROUPS[selected_group]

        selected_key = st.radio(
            "Chọn bài trong nhóm",
            list(group_modules.keys()),
            format_func=lambda key: group_modules[key]["label"]
        )

        selected_label = group_modules[selected_key]["label"]
        selected_paths = group_modules[selected_key]["paths"]

        st.divider()
        st.markdown("### 📌 Mô tả bài")
        st.info(group_modules[selected_key]["desc"])

    else:
        st.divider()
        st.info("Chọn “📚 Chọn nhóm bài” để mở danh sách 12 module.")

    st.divider()
    st.caption("Dữ liệu: NSO, MoST, MIC, MPI, WB, GII")


# =========================
# ĐIỀU HƯỚNG NỘI DUNG
# =========================

if selected_key == "home":
    show_home_page()

else:
    render_module(
        selected_paths,
        selected_label
    )

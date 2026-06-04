# bai05_mip_15_du_an.py
# Bài 5 — Quy hoạch nguyên hỗn hợp MIP lựa chọn 15 dự án chuyển đổi số
# Module dùng được với streamlit_app.py có cơ chế gọi module.render()

import itertools
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


try:
    import pulp
    PULP_AVAILABLE = True
except Exception:
    PULP_AVAILABLE = False


# =====================================================
# 1. DỮ LIỆU 15 DỰ ÁN THEO ĐỀ
# =====================================================

def build_projects_data() -> pd.DataFrame:
    return pd.DataFrame({
        "Mã": [f"P{i}" for i in range(1, 16)],
        "i": list(range(1, 16)),
        "Tên dự án": [
            "Trung tâm dữ liệu quốc gia Hòa Lạc",
            "Trung tâm dữ liệu quốc gia phía Nam",
            "Hệ thống 5G phủ sóng toàn quốc",
            "Hệ thống định danh điện tử VNeID 2.0",
            "Cổng dịch vụ công quốc gia v3",
            "Y tế số quốc gia",
            "Giáo dục số K-12 toàn quốc",
            "Trung tâm AI quốc gia + supercomputing",
            "Sandbox tài chính số",
            "Logistics thông minh + cảng biển số",
            "Nông nghiệp số ĐBSCL",
            "Đào tạo 50.000 kỹ sư AI/bán dẫn",
            "Khu CN bán dẫn Bắc Ninh - Bắc Giang",
            "An ninh mạng quốc gia SOC",
            "Open Data + dữ liệu mở quốc gia",
        ],
        "Lĩnh vực": [
            "Hạ tầng",
            "Hạ tầng",
            "Hạ tầng",
            "Chính phủ số",
            "Chính phủ số",
            "Y tế số",
            "Giáo dục",
            "AI",
            "Tài chính số",
            "Logistics",
            "Nông nghiệp",
            "Nhân lực",
            "Bán dẫn",
            "An ninh",
            "Dữ liệu",
        ],
        "Chi phí": [
            12000, 11500, 18000, 4500, 3200,
            5800, 6500, 15000, 2500, 7200,
            4800, 8500, 20000, 3800, 1500
        ],
        "NPV": [
            21500, 20800, 32500, 9200, 6800,
            11400, 12200, 28500, 5800, 13800,
            8500, 16200, 35000, 7500, 3800
        ],
        "Năm 1-2": [
            8500, 7500, 12000, 3500, 2500,
            4000, 4500, 9000, 1800, 5000,
            3500, 5500, 13000, 2800, 1200
        ],
        "Năm 3-5": [
            3500, 4000, 6000, 1000, 700,
            1800, 2000, 6000, 700, 2200,
            1300, 3000, 7000, 1000, 300
        ],
    })


def add_project_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["NPV/Chi phí"] = out["NPV"] / out["Chi phí"]
    out["Tỷ trọng Năm 1-2"] = out["Năm 1-2"] / out["Chi phí"]

    def p_complete(field):
        if field == "Hạ tầng":
            return 0.85
        if field in ["Chính phủ số"]:
            return 0.75
        if field in ["AI", "Bán dẫn"]:
            return 0.65
        return 0.80

    out["p hoàn thành"] = out["Lĩnh vực"].apply(p_complete)
    out["NPV kỳ vọng"] = out["NPV"] * out["p hoàn thành"]

    return out


# =====================================================
# 2. KIỂM TRA RÀNG BUỘC CHO MỘT TẬP DỰ ÁN
# =====================================================

def check_constraints(
    selected_ids,
    df,
    total_budget=80000,
    budget_year12=40000,
    min_projects=7,
    max_projects=11,
    force_p1_p2=False,
):
    selected_ids = set(selected_ids)
    temp = df[df["i"].isin(selected_ids)]

    total_cost = temp["Chi phí"].sum()
    cost_year12 = temp["Năm 1-2"].sum()
    n_projects = len(selected_ids)

    checks = {
        "C1_Ngan_sach_tong": total_cost <= total_budget,
        "C2_Ngan_sach_nam_1_2": cost_year12 <= budget_year12,
        "C3_Loai_tru_P1_P2": len({1, 2}.intersection(selected_ids)) <= 1,
        "C4_P8_can_P12": (8 not in selected_ids) or (12 in selected_ids),
        "C5_P13_can_P12": (13 not in selected_ids) or (12 in selected_ids),
        "C6_Chinh_phu_so_it_nhat_1": len({4, 5}.intersection(selected_ids)) >= 1,
        "C7_P14_bat_buoc": 14 in selected_ids,
        "C8_So_luong_du_an": min_projects <= n_projects <= max_projects,
    }

    if force_p1_p2:
        checks["C9_Bat_buoc_P1_va_P2"] = (1 in selected_ids) and (2 in selected_ids)

    feasible = all(checks.values())

    detail = {
        "Tổng chi phí": total_cost,
        "Chi phí năm 1-2": cost_year12,
        "Số dự án": n_projects,
        "Khả thi": feasible,
        **checks
    }

    return feasible, detail


# =====================================================
# 3. SOLVER PULP
# =====================================================

def solve_with_pulp(
    df,
    total_budget=80000,
    budget_year12=40000,
    min_projects=7,
    max_projects=11,
    force_p1_p2=False,
    use_expected_npv=False,
):
    if not PULP_AVAILABLE:
        return {
            "success": False,
            "message": "PuLP chưa được cài đặt. Hãy thêm `pulp` vào requirements.txt.",
            "selected": None,
            "objective": None,
            "status": None,
            "method": "PuLP"
        }

    projects = df["i"].tolist()

    C = dict(zip(df["i"], df["Chi phí"]))
    C12 = dict(zip(df["i"], df["Năm 1-2"]))

    if use_expected_npv:
        B = dict(zip(df["i"], df["NPV kỳ vọng"]))
    else:
        B = dict(zip(df["i"], df["NPV"]))

    model = pulp.LpProblem("VN_Project_Selection_MIP", pulp.LpMaximize)

    y = pulp.LpVariable.dicts(
        "y",
        projects,
        lowBound=0,
        upBound=1,
        cat="Binary"
    )

    # Hàm mục tiêu
    model += pulp.lpSum(B[i] * y[i] for i in projects), "Total_Benefit"

    # C1. Ngân sách tổng 5 năm
    model += pulp.lpSum(C[i] * y[i] for i in projects) <= total_budget, "C1_Ngan_sach_tong"

    # C2. Ngân sách năm 1-2
    model += pulp.lpSum(C12[i] * y[i] for i in projects) <= budget_year12, "C2_Ngan_sach_nam_1_2"

    # C3. Loại trừ trung tâm dữ liệu
    if force_p1_p2:
        # Trường hợp 5.4.3 yêu cầu bắt buộc có cả P1 và P2
        model += y[1] == 1, "C3a_Bat_buoc_P1"
        model += y[2] == 1, "C3b_Bat_buoc_P2"
    else:
        model += y[1] + y[2] <= 1, "C3_Loai_tru_P1_P2"

    # C4. Tiên quyết AI quốc gia cần đào tạo kỹ sư
    model += y[8] <= y[12], "C4_P8_can_P12"

    # C5. Tiên quyết bán dẫn cần đào tạo
    model += y[13] <= y[12], "C5_P13_can_P12"

    # C6. Ít nhất một chính phủ số
    model += y[4] + y[5] >= 1, "C6_Chinh_phu_so_it_nhat_1"

    # C7. An ninh mạng bắt buộc
    model += y[14] >= 1, "C7_P14_bat_buoc"

    # C8. Số lượng dự án tối thiểu/tối đa
    model += pulp.lpSum(y[i] for i in projects) >= min_projects, "C8_So_luong_toi_thieu"
    model += pulp.lpSum(y[i] for i in projects) <= max_projects, "C9_So_luong_toi_da"

    solver = pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)

    status = pulp.LpStatus[model.status]

    if status != "Optimal":
        return {
            "success": False,
            "message": f"PuLP không tìm được nghiệm tối ưu. Trạng thái: {status}",
            "selected": None,
            "objective": None,
            "status": status,
            "method": "PuLP"
        }

    selected = [i for i in projects if pulp.value(y[i]) > 0.5]
    objective = pulp.value(model.objective)

    return {
        "success": True,
        "message": "Optimal",
        "selected": selected,
        "objective": objective,
        "status": status,
        "method": "PuLP"
    }


# =====================================================
# 4. FALLBACK BRUTE FORCE NẾU PULP KHÔNG CÓ
# =====================================================

def solve_by_enumeration(
    df,
    total_budget=80000,
    budget_year12=40000,
    min_projects=7,
    max_projects=11,
    force_p1_p2=False,
    use_expected_npv=False,
):
    projects = df["i"].tolist()

    best_selected = None
    best_obj = -np.inf

    value_col = "NPV kỳ vọng" if use_expected_npv else "NPV"
    value_map = dict(zip(df["i"], df[value_col]))

    for bits in itertools.product([0, 1], repeat=len(projects)):
        selected = [projects[k] for k, bit in enumerate(bits) if bit == 1]

        feasible, detail = check_constraints(
            selected,
            df,
            total_budget=total_budget,
            budget_year12=budget_year12,
            min_projects=min_projects,
            max_projects=max_projects,
            force_p1_p2=force_p1_p2,
        )

        if not feasible:
            continue

        obj = sum(value_map[i] for i in selected)

        if obj > best_obj:
            best_obj = obj
            best_selected = selected

    if best_selected is None:
        return {
            "success": False,
            "message": "Enumeration không tìm được nghiệm khả thi.",
            "selected": None,
            "objective": None,
            "status": "Infeasible",
            "method": "Enumeration"
        }

    return {
        "success": True,
        "message": "Optimal",
        "selected": best_selected,
        "objective": best_obj,
        "status": "Optimal",
        "method": "Enumeration"
    }


def solve_mip(
    df,
    total_budget=80000,
    budget_year12=40000,
    min_projects=7,
    max_projects=11,
    force_p1_p2=False,
    use_expected_npv=False,
):
    pulp_result = solve_with_pulp(
        df=df,
        total_budget=total_budget,
        budget_year12=budget_year12,
        min_projects=min_projects,
        max_projects=max_projects,
        force_p1_p2=force_p1_p2,
        use_expected_npv=use_expected_npv,
    )

    if pulp_result["success"]:
        return pulp_result

    # Fallback để app vẫn chạy trên Streamlit dù chưa cài PuLP
    enum_result = solve_by_enumeration(
        df=df,
        total_budget=total_budget,
        budget_year12=budget_year12,
        min_projects=min_projects,
        max_projects=max_projects,
        force_p1_p2=force_p1_p2,
        use_expected_npv=use_expected_npv,
    )

    if enum_result["success"]:
        enum_result["message"] = (
            "Đã dùng Enumeration fallback vì PuLP không khả dụng hoặc PuLP không giải được."
        )

    return enum_result


# =====================================================
# 5. TÓM TẮT NGHIỆM
# =====================================================

def selected_projects_table(df, selected_ids, use_expected_npv=False):
    selected_ids = selected_ids or []

    temp = df.copy()
    temp["Chọn"] = temp["i"].isin(selected_ids)

    selected_df = temp[temp["Chọn"]].copy()

    if len(selected_df) == 0:
        summary = {
            "n_projects": 0,
            "total_cost": 0,
            "total_year12": 0,
            "total_year35": 0,
            "total_npv": 0,
            "expected_npv": 0,
            "npv_cost_ratio": np.nan,
        }
        return selected_df, summary, temp

    summary = {
        "n_projects": int(selected_df.shape[0]),
        "total_cost": float(selected_df["Chi phí"].sum()),
        "total_year12": float(selected_df["Năm 1-2"].sum()),
        "total_year35": float(selected_df["Năm 3-5"].sum()),
        "total_npv": float(selected_df["NPV"].sum()),
        "expected_npv": float(selected_df["NPV kỳ vọng"].sum()),
        "npv_cost_ratio": float(selected_df["NPV"].sum() / selected_df["Chi phí"].sum()),
    }

    return selected_df, summary, temp


def scenario_compare(df):
    scenarios = [
        {
            "Tên kịch bản": "Gốc: ngân sách 80.000",
            "total_budget": 80000,
            "budget_year12": 40000,
            "force_p1_p2": False,
            "use_expected_npv": False,
        },
        {
            "Tên kịch bản": "Nới ngân sách lên 100.000",
            "total_budget": 100000,
            "budget_year12": 40000,
            "force_p1_p2": False,
            "use_expected_npv": False,
        },
        {
            "Tên kịch bản": "Bắt buộc P1 và P2",
            "total_budget": 80000,
            "budget_year12": 40000,
            "force_p1_p2": True,
            "use_expected_npv": False,
        },
        {
            "Tên kịch bản": "Tối đa hóa NPV kỳ vọng",
            "total_budget": 80000,
            "budget_year12": 40000,
            "force_p1_p2": False,
            "use_expected_npv": True,
        },
    ]

    rows = []
    selected_map = {}

    for sc in scenarios:
        result = solve_mip(
            df,
            total_budget=sc["total_budget"],
            budget_year12=sc["budget_year12"],
            force_p1_p2=sc["force_p1_p2"],
            use_expected_npv=sc["use_expected_npv"],
        )

        if result["success"]:
            selected_df, summary, _ = selected_projects_table(df, result["selected"])
            selected_codes = selected_df["Mã"].tolist()

            rows.append({
                "Kịch bản": sc["Tên kịch bản"],
                "Trạng thái": result["status"],
                "Phương pháp": result["method"],
                "Số dự án": summary["n_projects"],
                "Tổng chi phí": summary["total_cost"],
                "Chi phí năm 1-2": summary["total_year12"],
                "Tổng NPV": summary["total_npv"],
                "Tổng NPV kỳ vọng": summary["expected_npv"],
                "Giá trị mục tiêu": result["objective"],
                "NPV/Chi phí": summary["npv_cost_ratio"],
                "Dự án chọn": ", ".join(selected_codes),
            })

            selected_map[sc["Tên kịch bản"]] = selected_codes

        else:
            rows.append({
                "Kịch bản": sc["Tên kịch bản"],
                "Trạng thái": result["status"],
                "Phương pháp": result["method"],
                "Số dự án": np.nan,
                "Tổng chi phí": np.nan,
                "Chi phí năm 1-2": np.nan,
                "Tổng NPV": np.nan,
                "Tổng NPV kỳ vọng": np.nan,
                "Giá trị mục tiêu": np.nan,
                "NPV/Chi phí": np.nan,
                "Dự án chọn": result["message"],
            })

            selected_map[sc["Tên kịch bản"]] = []

    return pd.DataFrame(rows), selected_map


# =====================================================
# 6. GIAO DIỆN STREAMLIT
# =====================================================

def render():
    st.markdown(
        """
        <div style="
            background: rgba(15, 23, 42, 0.86);
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 22px;
            padding: 24px;
            margin-bottom: 18px;
            box-shadow: 0 10px 35px rgba(0,0,0,0.28);
        ">
            <h1>🎯 Bài 5 — MIP lựa chọn 15 dự án chuyển đổi số</h1>
            <p>
            Module này giải bài toán quy hoạch nguyên hỗn hợp với biến nhị phân yᵢ,
            lựa chọn tập dự án tối ưu trong chương trình chuyển đổi số quốc gia giai đoạn 2026-2030.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    df = add_project_metrics(build_projects_data())

    st.info(
        "Mỗi biến yᵢ là biến nhị phân: yᵢ = 1 nếu chọn dự án i, yᵢ = 0 nếu không chọn. "
        "Mục tiêu gốc là tối đa hóa tổng lợi ích NPV trong giới hạn ngân sách và các ràng buộc chính sách."
    )

    with st.sidebar:
        st.markdown("### ⚙️ Tham số Bài 5")

        total_budget = st.number_input(
            "Ngân sách tổng 5 năm",
            min_value=30000,
            max_value=150000,
            value=80000,
            step=5000
        )

        budget_year12 = st.number_input(
            "Ngân sách năm 1-2",
            min_value=10000,
            max_value=100000,
            value=40000,
            step=5000
        )

        min_projects = st.number_input(
            "Số dự án tối thiểu",
            min_value=1,
            max_value=15,
            value=7,
            step=1
        )

        max_projects = st.number_input(
            "Số dự án tối đa",
            min_value=1,
            max_value=15,
            value=11,
            step=1
        )

        use_expected_npv = st.checkbox(
            "Dùng NPV kỳ vọng theo xác suất hoàn thành",
            value=False
        )

    if min_projects > max_projects:
        st.error("Số dự án tối thiểu không được lớn hơn số dự án tối đa.")
        st.stop()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📌 Mô hình",
        "5.4.1 MIP PuLP",
        "5.4.2 Ngân sách 100.000",
        "5.4.3 Bắt buộc P1 & P2",
        "5.4.4 Rủi ro dự án",
        "🧠 Thảo luận chính sách"
    ])

    # =====================================================
    # TAB 1 — MÔ HÌNH
    # =====================================================

    with tab1:
        st.header("1. Mô hình toán học")

        st.markdown("### Biến quyết định")

        st.latex(r"y_i \in \{0,1\},\quad i=1,\ldots,15")

        st.markdown("Trong đó: `yᵢ = 1` nếu chọn dự án `i`, ngược lại `yᵢ = 0`.")

        st.markdown("### Hàm mục tiêu")

        st.latex(r"\max Z = \sum_i B_i y_i")

        st.markdown("### Ràng buộc")

        constraints_df = pd.DataFrame({
            "Mã": ["C1", "C2", "C3", "C4", "C5", "C6", "C7"],
            "Ràng buộc": [
                r"Σ Cᵢyᵢ ≤ 80.000",
                r"Σ C₁ᵢyᵢ ≤ 40.000",
                r"y₁ + y₂ ≤ 1",
                r"y₈ ≤ y₁₂",
                r"y₁₃ ≤ y₁₂",
                r"y₄ + y₅ ≥ 1; y₁₄ ≥ 1",
                r"7 ≤ Σyᵢ ≤ 11",
            ],
            "Ý nghĩa": [
                "Ngân sách tổng 5 năm",
                "Ngân sách năm 1-2",
                "Chỉ chọn một trong hai trung tâm dữ liệu P1/P2",
                "Trung tâm AI cần dự án đào tạo kỹ sư",
                "Khu bán dẫn cần dự án đào tạo kỹ sư",
                "Có ít nhất một dự án chính phủ số và an ninh mạng bắt buộc",
                "Giới hạn số lượng dự án được chọn",
            ]
        })

        st.dataframe(constraints_df, use_container_width=True)

        st.markdown("### Danh mục 15 dự án ứng cử")

        st.dataframe(
            df[[
                "Mã", "Tên dự án", "Lĩnh vực", "Chi phí", "NPV",
                "Năm 1-2", "Năm 3-5", "NPV/Chi phí",
                "p hoàn thành", "NPV kỳ vọng"
            ]].round(4),
            use_container_width=True
        )

        fig_ratio = px.bar(
            df.sort_values("NPV/Chi phí", ascending=False),
            x="Mã",
            y="NPV/Chi phí",
            hover_name="Tên dự án",
            color="Lĩnh vực",
            title="Tỷ suất NPV/Chi phí của 15 dự án"
        )
        st.plotly_chart(fig_ratio, use_container_width=True)

    # =====================================================
    # TAB 2 — 5.4.1
    # =====================================================

    with tab2:
        st.header("Câu 5.4.1 — Cài đặt và giải bài toán bằng PuLP/CBC")

        result = solve_mip(
            df,
            total_budget=total_budget,
            budget_year12=budget_year12,
            min_projects=min_projects,
            max_projects=max_projects,
            force_p1_p2=False,
            use_expected_npv=use_expected_npv,
        )

        if not result["success"]:
            st.error(result["message"])
        else:
            selected_df, summary, all_df = selected_projects_table(
                df,
                result["selected"],
                use_expected_npv=use_expected_npv
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Trạng thái", result["status"])
            c2.metric("Giá trị mục tiêu Z*", f"{result['objective']:,.0f}")
            c3.metric("Số dự án chọn", f"{summary['n_projects']}")
            c4.metric("NPV/Chi phí", f"{summary['npv_cost_ratio']:.3f}")

            c5, c6, c7 = st.columns(3)

            c5.metric("Tổng chi phí", f"{summary['total_cost']:,.0f}")
            c6.metric("Chi phí năm 1-2", f"{summary['total_year12']:,.0f}")
            c7.metric("Tổng NPV", f"{summary['total_npv']:,.0f}")

            st.markdown("### Các dự án được chọn")
            st.dataframe(
                selected_df[[
                    "Mã", "Tên dự án", "Lĩnh vực", "Chi phí", "NPV",
                    "Năm 1-2", "Năm 3-5", "NPV/Chi phí",
                    "p hoàn thành", "NPV kỳ vọng"
                ]].round(4),
                use_container_width=True
            )

            feasible, detail = check_constraints(
                result["selected"],
                df,
                total_budget=total_budget,
                budget_year12=budget_year12,
                min_projects=min_projects,
                max_projects=max_projects,
                force_p1_p2=False
            )

            check_df = pd.DataFrame({
                "Chỉ tiêu": list(detail.keys()),
                "Giá trị": list(detail.values())
            })

            st.markdown("### Kiểm tra ràng buộc")
            st.dataframe(check_df, use_container_width=True)

            fig_selected = px.bar(
                selected_df,
                x="Mã",
                y="NPV",
                color="Lĩnh vực",
                hover_name="Tên dự án",
                text="NPV",
                title="NPV của các dự án được chọn"
            )
            st.plotly_chart(fig_selected, use_container_width=True)

            fig_budget = px.pie(
                selected_df,
                names="Mã",
                values="Chi phí",
                hover_name="Tên dự án",
                title="Cơ cấu chi phí của các dự án được chọn"
            )
            st.plotly_chart(fig_budget, use_container_width=True)

            if result["method"] != "PuLP":
                st.warning(
                    "App đang dùng Enumeration fallback vì PuLP chưa khả dụng. "
                    "Để đúng yêu cầu đề bài, hãy thêm `pulp` vào requirements.txt."
                )

    # =====================================================
    # TAB 3 — 5.4.2
    # =====================================================

    with tab3:
        st.header("Câu 5.4.2 — Phân tích khi nới ngân sách lên 100.000 tỷ")

        base = solve_mip(
            df,
            total_budget=80000,
            budget_year12=40000,
            min_projects=7,
            max_projects=11,
            force_p1_p2=False,
            use_expected_npv=False,
        )

        relaxed = solve_mip(
            df,
            total_budget=100000,
            budget_year12=40000,
            min_projects=7,
            max_projects=11,
            force_p1_p2=False,
            use_expected_npv=False,
        )

        if not base["success"] or not relaxed["success"]:
            st.error("Không giải được một trong hai kịch bản.")
            st.write("Gốc:", base["message"])
            st.write("Nới ngân sách:", relaxed["message"])
        else:
            base_df, base_summary, _ = selected_projects_table(df, base["selected"])
            relaxed_df, relaxed_summary, _ = selected_projects_table(df, relaxed["selected"])

            compare = pd.DataFrame({
                "Kịch bản": ["Ngân sách 80.000", "Ngân sách 100.000"],
                "Z*": [base["objective"], relaxed["objective"]],
                "Số dự án": [base_summary["n_projects"], relaxed_summary["n_projects"]],
                "Tổng chi phí": [base_summary["total_cost"], relaxed_summary["total_cost"]],
                "Chi phí năm 1-2": [base_summary["total_year12"], relaxed_summary["total_year12"]],
                "Tổng NPV": [base_summary["total_npv"], relaxed_summary["total_npv"]],
                "NPV/Chi phí": [base_summary["npv_cost_ratio"], relaxed_summary["npv_cost_ratio"]],
                "Dự án chọn": [
                    ", ".join(base_df["Mã"].tolist()),
                    ", ".join(relaxed_df["Mã"].tolist())
                ]
            })

            st.dataframe(compare.round(4), use_container_width=True)

            added = sorted(list(set(relaxed["selected"]) - set(base["selected"])))
            removed = sorted(list(set(base["selected"]) - set(relaxed["selected"])))

            c1, c2, c3 = st.columns(3)
            c1.metric("Z* gốc", f"{base['objective']:,.0f}")
            c2.metric("Z* 100.000", f"{relaxed['objective']:,.0f}")
            c3.metric("Tăng Z*", f"{relaxed['objective'] - base['objective']:,.0f}")

            st.markdown("### Dự án thay đổi")

            change_df = pd.DataFrame({
                "Loại thay đổi": ["Thêm khi nới ngân sách", "Bị loại khi nới ngân sách"],
                "Dự án": [
                    ", ".join([f"P{i}" for i in added]) if added else "Không có",
                    ", ".join([f"P{i}" for i in removed]) if removed else "Không có"
                ]
            })

            st.dataframe(change_df, use_container_width=True)

            both = pd.concat([
                base_df.assign(Kịch_bản="80.000"),
                relaxed_df.assign(Kịch_bản="100.000")
            ])

            fig = px.bar(
                both,
                x="Mã",
                y="NPV",
                color="Kịch_bản",
                barmode="group",
                hover_name="Tên dự án",
                title="So sánh dự án được chọn khi ngân sách tăng"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.success(
                "Khi nới ngân sách, mô hình có thêm không gian chọn các dự án NPV cao hơn, "
                "nhưng vẫn bị khống chế bởi ngân sách năm 1-2, ràng buộc tiên quyết và số lượng dự án tối đa."
            )

    # =====================================================
    # TAB 4 — 5.4.3
    # =====================================================

    with tab4:
        st.header("Câu 5.4.3 — Quốc hội yêu cầu phải có cả P1 và P2")

        st.warning(
            "Lưu ý: yêu cầu bắt buộc cả P1 và P2 mâu thuẫn trực tiếp với ràng buộc gốc y1 + y2 ≤ 1. "
            "Trong kịch bản này, app thay ràng buộc loại trừ bằng y1 = 1 và y2 = 1 để kiểm tra tính khả thi mới."
        )

        base = solve_mip(
            df,
            total_budget=80000,
            budget_year12=40000,
            min_projects=7,
            max_projects=11,
            force_p1_p2=False,
            use_expected_npv=False,
        )

        forced = solve_mip(
            df,
            total_budget=80000,
            budget_year12=40000,
            min_projects=7,
            max_projects=11,
            force_p1_p2=True,
            use_expected_npv=False,
        )

        if not base["success"]:
            st.error("Kịch bản gốc không giải được.")
            st.write(base["message"])
        elif not forced["success"]:
            st.error("Kịch bản bắt buộc P1 và P2 không khả thi.")
            st.write(forced["message"])

            base_df, base_summary, _ = selected_projects_table(df, base["selected"])

            st.info(
                "Nguyên nhân thường gặp: P1 và P2 đều có chi phí lớn, đồng thời tiêu tốn ngân sách năm 1-2. "
                "Khi bắt buộc cả hai, mô hình có thể không còn đủ ngân sách để thỏa mãn các ràng buộc số lượng, "
                "dự án bắt buộc P14, chính phủ số và tiên quyết."
            )
        else:
            base_df, base_summary, _ = selected_projects_table(df, base["selected"])
            forced_df, forced_summary, _ = selected_projects_table(df, forced["selected"])

            compare = pd.DataFrame({
                "Kịch bản": ["Gốc", "Bắt buộc P1 và P2"],
                "Khả thi": ["Có", "Có"],
                "Z*": [base["objective"], forced["objective"]],
                "Thay đổi Z*": [0, forced["objective"] - base["objective"]],
                "Số dự án": [base_summary["n_projects"], forced_summary["n_projects"]],
                "Tổng chi phí": [base_summary["total_cost"], forced_summary["total_cost"]],
                "Chi phí năm 1-2": [base_summary["total_year12"], forced_summary["total_year12"]],
                "Dự án chọn": [
                    ", ".join(base_df["Mã"].tolist()),
                    ", ".join(forced_df["Mã"].tolist())
                ]
            })

            st.dataframe(compare.round(4), use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Z* gốc", f"{base['objective']:,.0f}")
            c2.metric("Z* bắt buộc P1/P2", f"{forced['objective']:,.0f}")
            c3.metric("Thay đổi Z*", f"{forced['objective'] - base['objective']:,.0f}")

            fig = px.bar(
                compare,
                x="Kịch bản",
                y="Z*",
                text=compare["Z*"].round(0),
                title="So sánh Z* khi bắt buộc cả P1 và P2"
            )
            st.plotly_chart(fig, use_container_width=True)

            if forced["objective"] < base["objective"]:
                st.warning(
                    "Việc bắt buộc cả P1 và P2 làm giảm Z*. "
                    "Đây là chi phí kinh tế của mục tiêu redundancy/an toàn hệ thống."
                )
            else:
                st.success(
                    "Trong cấu hình hiện tại, yêu cầu P1 và P2 không làm giảm Z*. "
                    "Điều này có thể xảy ra nếu hai dự án vẫn nằm trong tổ hợp tối ưu sau khi xét các ràng buộc khác."
                )

    # =====================================================
    # TAB 5 — 5.4.4
    # =====================================================

    with tab5:
        st.header("Câu 5.4.4 — Mở rộng: thêm rủi ro dự án và tối đa hóa NPV kỳ vọng")

        st.markdown(
            """
            Xác suất hoàn thành đúng tiến độ giả định:
            - Hạ tầng: 0,85  
            - Chính phủ số: 0,75  
            - AI/Bán dẫn: 0,65  
            - Các lĩnh vực còn lại: 0,80  
            """
        )

        risk_df = df[[
            "Mã", "Tên dự án", "Lĩnh vực", "NPV",
            "p hoàn thành", "NPV kỳ vọng", "Chi phí", "NPV/Chi phí"
        ]].copy()

        st.dataframe(risk_df.round(4), use_container_width=True)

        normal = solve_mip(
            df,
            total_budget=80000,
            budget_year12=40000,
            min_projects=7,
            max_projects=11,
            force_p1_p2=False,
            use_expected_npv=False,
        )

        expected = solve_mip(
            df,
            total_budget=80000,
            budget_year12=40000,
            min_projects=7,
            max_projects=11,
            force_p1_p2=False,
            use_expected_npv=True,
        )

        if not normal["success"] or not expected["success"]:
            st.error("Không giải được một trong hai mô hình.")
            st.write("Mô hình NPV gốc:", normal["message"])
            st.write("Mô hình NPV kỳ vọng:", expected["message"])
        else:
            normal_df, normal_summary, _ = selected_projects_table(df, normal["selected"])
            expected_df, expected_summary, _ = selected_projects_table(df, expected["selected"])

            compare = pd.DataFrame({
                "Mô hình": ["Tối đa hóa NPV gốc", "Tối đa hóa NPV kỳ vọng"],
                "Giá trị mục tiêu": [normal["objective"], expected["objective"]],
                "Tổng NPV gốc": [normal_summary["total_npv"], expected_summary["total_npv"]],
                "Tổng NPV kỳ vọng": [normal_summary["expected_npv"], expected_summary["expected_npv"]],
                "Tổng chi phí": [normal_summary["total_cost"], expected_summary["total_cost"]],
                "Chi phí năm 1-2": [normal_summary["total_year12"], expected_summary["total_year12"]],
                "Dự án chọn": [
                    ", ".join(normal_df["Mã"].tolist()),
                    ", ".join(expected_df["Mã"].tolist())
                ]
            })

            st.dataframe(compare.round(4), use_container_width=True)

            fig = px.bar(
                compare,
                x="Mô hình",
                y=["Tổng NPV gốc", "Tổng NPV kỳ vọng"],
                barmode="group",
                title="So sánh tối ưu theo NPV gốc và NPV kỳ vọng"
            )
            st.plotly_chart(fig, use_container_width=True)

            changed_added = sorted(list(set(expected["selected"]) - set(normal["selected"])))
            changed_removed = sorted(list(set(normal["selected"]) - set(expected["selected"])))

            st.markdown("### Thay đổi tập dự án khi xét rủi ro")

            st.dataframe(
                pd.DataFrame({
                    "Loại thay đổi": ["Thêm vào", "Loại ra"],
                    "Dự án": [
                        ", ".join([f"P{i}" for i in changed_added]) if changed_added else "Không có",
                        ", ".join([f"P{i}" for i in changed_removed]) if changed_removed else "Không có",
                    ]
                }),
                use_container_width=True
            )

            st.info(
                "Khi tối đa hóa NPV kỳ vọng, các dự án có NPV danh nghĩa cao nhưng xác suất hoàn thành thấp "
                "có thể bị giảm sức hấp dẫn. Đây là cách đưa rủi ro triển khai vào mô hình lựa chọn dự án."
            )

    # =====================================================
    # TAB 6 — THẢO LUẬN
    # =====================================================

    with tab6:
        st.header("5.5 — Câu hỏi thảo luận chính sách")

        result = solve_mip(
            df,
            total_budget=80000,
            budget_year12=40000,
            min_projects=7,
            max_projects=11,
            force_p1_p2=False,
            use_expected_npv=False,
        )

        if result["success"]:
            selected_df, summary, all_df = selected_projects_table(df, result["selected"])

            selected_codes = set(selected_df["Mã"].tolist())

            st.markdown("### a) Vì sao mô hình có thể bỏ qua P15 dù P15 có tỷ suất lợi ích/chi phí cao?")

            if "P15" in selected_codes:
                st.success(
                    "Trong nghiệm hiện tại, P15 được chọn. Điều này hợp lý vì P15 có chi phí thấp và NPV/Chi phí cao."
                )
            else:
                st.warning(
                    "Nếu P15 bị bỏ qua, nguyên nhân không nhất thiết là P15 kém hiệu quả. "
                    "MIP chọn tổ hợp tối ưu toàn cục dưới nhiều ràng buộc đồng thời: ngân sách năm 1-2, "
                    "số lượng dự án, dự án bắt buộc, ràng buộc tiên quyết và loại trừ. "
                    "Một dự án có NPV/Chi phí cao vẫn có thể bị loại nếu nó làm mất chỗ của tổ hợp dự án có tổng NPV lớn hơn."
                )

            st.markdown("### b) Ràng buộc bắt buộc P14 có làm giảm Z* không? Có hợp lý không?")

            # So sánh với mô hình bỏ bắt buộc P14 bằng cách brute-force riêng
            st.write(
                "P14 là dự án an ninh mạng quốc gia. Về mặt kinh tế thuần túy, ràng buộc bắt buộc có thể làm giảm Z* "
                "nếu P14 không nằm trong tổ hợp tối đa hóa NPV. Tuy nhiên, về chính sách công, an ninh mạng là điều kiện nền "
                "để các dự án dữ liệu, AI, định danh điện tử và dịch vụ công vận hành an toàn. Do đó, việc bắt buộc P14 là hợp lý "
                "như một ràng buộc an toàn hệ thống, không chỉ là lựa chọn tài chính."
            )

            st.markdown("### c) Mô hình hóa hiệu ứng cộng hưởng giữa P8 và P13 như thế nào?")

            st.write(
                "Mô hình hiện tại giả định lợi ích các dự án độc lập, tức tổng lợi ích bằng tổng NPV riêng lẻ. "
                "Nếu P8 trung tâm AI và P13 khu công nghiệp bán dẫn có cộng hưởng, có thể thêm biến nhị phân phụ z_8_13 với: "
                "`z_8_13 ≤ y8`, `z_8_13 ≤ y13`, `z_8_13 ≥ y8 + y13 - 1`, rồi cộng thêm `S_8_13 * z_8_13` vào hàm mục tiêu. "
                "Khi đó mô hình sẽ ghi nhận lợi ích tăng thêm nếu cả hai dự án cùng được chọn."
            )

            st.markdown("### Kết luận ngắn")

            st.success(
                f"Nghiệm cơ sở chọn {summary['n_projects']} dự án, tổng chi phí {summary['total_cost']:,.0f} tỷ VND, "
                f"tổng NPV {summary['total_npv']:,.0f} tỷ VND, tỷ suất NPV/chi phí {summary['npv_cost_ratio']:.3f}. "
                "Kết quả cần được hiểu như một đề xuất định lượng ban đầu; quyết định cuối cùng vẫn cần xét yếu tố chiến lược, "
                "an ninh, vùng miền và năng lực triển khai."
            )

        else:
            st.error(result["message"])


def run():
    render()

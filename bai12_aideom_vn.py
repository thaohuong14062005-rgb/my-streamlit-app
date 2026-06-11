# bai12_aideom_dashboard.py
# Bài 12 - Đồ án tích hợp AIDEOM-VN
# Module Streamlit: có hàm render() và run()

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    px = None
    go = None
    PLOTLY_AVAILABLE = False


# ============================================================
# 1. Cấu hình chung
# ============================================================

ITEMS = ["K", "D", "AI", "H"]

ITEM_LABELS = {
    "K": "Vốn / hạ tầng",
    "D": "Chuyển đổi số",
    "AI": "Trí tuệ nhân tạo",
    "H": "Nhân lực số",
}

SCENARIO_NAMES = {
    "S1": "S1 - Truyền thống",
    "S2": "S2 - Số hóa nhanh",
    "S3": "S3 - AI dẫn dắt",
    "S4": "S4 - Bao trùm số",
    "S5": "S5 - Tối ưu cân bằng",
}

SCENARIO_DESCRIPTIONS = {
    "S1": "Tập trung vốn vật chất, FDI, hạ tầng truyền thống, xuất khẩu.",
    "S2": "Tăng đầu tư chính phủ số, doanh nghiệp số, thanh toán số.",
    "S3": "Ưu tiên AI, dữ liệu lớn, bán dẫn, trung tâm dữ liệu.",
    "S4": "Ưu tiên vùng yếu, SME, giáo dục số, nông nghiệp số.",
    "S5": "Kịch bản tối ưu cân bằng do mô hình tự tìm theo trọng số chính sách.",
}

BASE_SCENARIO_SHARES = {
    "S1": np.array([0.70, 0.10, 0.10, 0.10]),
    "S2": np.array([0.25, 0.45, 0.15, 0.15]),
    "S3": np.array([0.20, 0.20, 0.45, 0.15]),
    "S4": np.array([0.30, 0.20, 0.10, 0.40]),
}

YEARS = list(range(2026, 2031))


# ============================================================
# 2. Dữ liệu mẫu Việt Nam
# ============================================================

def get_regions_data():
    """
    Dữ liệu 6 vùng kinh tế - xã hội Việt Nam.
    Dùng để minh họa module M2 và M3.
    """
    return pd.DataFrame({
        "region": ["NMM", "RRD", "NCC", "CH", "SE", "MD"],
        "region_name": [
            "Trung du miền núi phía Bắc",
            "Đồng bằng sông Hồng",
            "Bắc Trung Bộ + DH Trung Bộ",
            "Tây Nguyên",
            "Đông Nam Bộ",
            "Đồng bằng sông Cửu Long",
        ],
        "grdp_per_capita": [57.0, 152.3, 87.5, 68.9, 158.9, 80.5],
        "fdi": [3.5, 20.0, 8.2, 0.8, 18.5, 2.1],
        "digital_index": [38, 78, 55, 32, 82, 48],
        "ai_readiness": [22, 68, 40, 18, 75, 30],
        "trained_labor": [21.5, 36.8, 27.5, 18.2, 42.5, 16.8],
        "rd_intensity": [0.18, 0.85, 0.32, 0.15, 0.78, 0.22],
        "internet": [72, 92, 84, 68, 94, 78],
        "gini": [0.405, 0.358, 0.372, 0.412, 0.385, 0.392],
        "emission_coef": [0.42, 0.55, 0.48, 0.32, 0.62, 0.38],
        "data_risk_coef": [0.18, 0.45, 0.28, 0.12, 0.52, 0.22],
        "human_risk_reduction": [0.32, 0.28, 0.30, 0.35, 0.25, 0.30],
    })


def get_sector_labor_data():
    """
    Dữ liệu 8 ngành cho module M4 - mô phỏng lao động.
    """
    return pd.DataFrame({
        "sector": [
            "Nông-Lâm-Thủy sản",
            "CN chế biến chế tạo",
            "Xây dựng",
            "Bán buôn-bán lẻ",
            "Tài chính-Ngân hàng",
            "Logistics-Vận tải",
            "CNTT-Truyền thông",
            "Giáo dục-Đào tạo",
        ],
        "labor_million": [13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15],
        "risk_pct": [18, 42, 25, 38, 52, 35, 28, 22],
        "a1_ai_job": [8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5],
        "b1_upgrade": [45.0, 28.0, 35.0, 32.0, 22.0, 30.0, 20.0, 55.0],
        "c1_displace": [5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5],
        "d1_retrain": [50.0, 32.0, 42.0, 38.0, 26.0, 36.0, 24.0, 62.0],
    })


def get_module_structure():
    """
    Cấu trúc 6 module AIDEOM-VN theo Bài 12.
    """
    return pd.DataFrame({
        "Module": ["M1", "M2", "M3", "M4", "M5", "M6"],
        "Tên module": [
            "Dự báo kinh tế",
            "Đánh giá sẵn sàng số",
            "Tối ưu phân bổ",
            "Mô phỏng lao động",
            "Đánh giá rủi ro",
            "Dashboard ra quyết định",
        ],
        "Đầu vào": [
            "Macro 2020-2025",
            "Sectors, Regions",
            "Budget, beta-matrix",
            "AI, H plans",
            "Risk parameters",
            "Outputs M1-M5",
        ],
        "Đầu ra": [
            "GDP, TFP, lao động 2026-2030",
            "Digital Index + AI Readiness",
            "Phân bổ ngành-vùng-thời gian",
            "NetJob từng ngành",
            "Cyber, environmental, dependency risk",
            "Kịch bản, cảnh báo, khuyến nghị",
        ],
        "Kỹ thuật": [
            "Cobb-Douglas",
            "TOPSIS + entropy",
            "LP + dynamic allocation",
            "NetJob simulation",
            "Multi-objective + stochastic logic",
            "Streamlit dashboard",
        ],
    })


# ============================================================
# 3. Hàm tính toán chung
# ============================================================

def minmax_good(x):
    x = np.asarray(x, dtype=float)
    denom = x.max() - x.min()
    if denom == 0:
        return np.ones_like(x) * 0.5
    return (x - x.min()) / denom


def minmax_bad(x):
    x = np.asarray(x, dtype=float)
    denom = x.max() - x.min()
    if denom == 0:
        return np.ones_like(x) * 0.5
    return (x.max() - x) / denom


def clamp(value, low, high):
    return float(max(low, min(high, value)))


def production_y(A, K, L, D, AI, H):
    """
    Hàm sản xuất Cobb-Douglas mở rộng:
    Y = A*K^0.33*L^0.42*D^0.10*AI^0.08*H^0.07
    """
    K = max(float(K), 1e-9)
    L = max(float(L), 1e-9)
    D = max(float(D), 1e-9)
    AI = max(float(AI), 1e-9)
    H = max(float(H), 1e-9)

    return (
        A
        * (K ** 0.33)
        * (L ** 0.42)
        * (D ** 0.10)
        * (AI ** 0.08)
        * (H ** 0.07)
    )


def calibrate_A0():
    """
    Hiệu chỉnh A0 để GDP ban đầu xấp xỉ GDP 2025.
    """
    Y0 = 12847.6
    K0 = 27500.0
    L0 = 53.9
    D0 = 20.3
    AI0 = 86.0
    H0 = 30.0

    denom = (
        (K0 ** 0.33)
        * (L0 ** 0.42)
        * (D0 ** 0.10)
        * (AI0 ** 0.08)
        * (H0 ** 0.07)
    )
    return Y0 / denom


def normalize_shares(shares):
    shares = np.asarray(shares, dtype=float)
    total = shares.sum()
    if total <= 0:
        return np.array([0.25, 0.25, 0.25, 0.25])
    return shares / total


def get_candidate_allocations(step=0.05):
    """
    Sinh các phương án phân bổ K, D, AI, H có tổng bằng 1.
    Dùng cho S5 - tối ưu cân bằng.
    """
    values = np.arange(0.05, 0.91, step)
    candidates = []

    for k in values:
        for d in values:
            for ai in values:
                h = 1.0 - k - d - ai
                if h >= 0.05 and h <= 0.90:
                    shares = np.array([k, d, ai, h])
                    if abs(shares.sum() - 1.0) < 1e-8:
                        candidates.append(shares)

    return np.array(candidates)


# ============================================================
# 4. M1 - Dự báo kinh tế
# ============================================================

def simulate_macro_path(
    shares,
    annual_budget=1000.0,
    years=None,
    scenario_code="S1",
):
    """
    Mô phỏng đường đi vĩ mô 2026-2030 cho một kịch bản phân bổ.
    """
    if years is None:
        years = YEARS

    shares = normalize_shares(shares)

    K = 27500.0
    L = 53.9
    D = 20.3
    AI = 86.0
    H = 30.0
    A = calibrate_A0()

    previous_y = 12847.6
    rows = []

    for year in years:
        invest_k = shares[0] * annual_budget
        invest_d = shares[1] * annual_budget
        invest_ai = shares[2] * annual_budget
        invest_h = shares[3] * annual_budget

        K = (1.0 - 0.05) * K + invest_k

        D = (1.0 - 0.08) * D + 0.010 * invest_d + 0.018 * H
        D = clamp(D, 5.0, 45.0)

        AI = (1.0 - 0.10) * AI + 0.035 * invest_ai + 0.080 * D
        AI = clamp(AI, 30.0, 250.0)

        H = (1.0 - 0.015) * H + 0.006 * invest_h
        H = clamp(H, 20.0, 60.0)

        tfp_growth = (
            0.003 * (D / 100.0)
            + 0.002 * (AI / 150.0)
            + 0.004 * (H / 100.0)
        )
        A = A * (1.0 + tfp_growth)

        Y = production_y(A, K, L, D, AI, H)
        growth = (Y / previous_y - 1.0) * 100.0

        rows.append({
            "scenario": scenario_code,
            "year": year,
            "K": K,
            "D": D,
            "AI": AI,
            "H": H,
            "A": A,
            "GDP": Y,
            "GDP_growth_pct": growth,
            "digital_gdp_share_pct": D,
            "AI_capacity": AI,
            "human_capital_pct": H,
            "invest_K": invest_k,
            "invest_D": invest_d,
            "invest_AI": invest_ai,
            "invest_H": invest_h,
        })

        previous_y = Y

    return pd.DataFrame(rows)


# ============================================================
# 5. M2 - Đánh giá sẵn sàng số bằng TOPSIS
# ============================================================

def compute_region_topsis():
    """
    Tính TOPSIS cho 6 vùng theo readiness AI.
    """
    df = get_regions_data()

    criteria = [
        "grdp_per_capita",
        "fdi",
        "digital_index",
        "ai_readiness",
        "trained_labor",
        "rd_intensity",
        "internet",
        "gini",
    ]

    is_benefit = np.array([True, True, True, True, True, True, True, False])
    weights = np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10])

    X = df[criteria].to_numpy(dtype=float)
    denom = np.sqrt((X ** 2).sum(axis=0))
    denom[denom == 0] = 1.0
    R = X / denom
    V = R * weights

    ideal = np.where(is_benefit, V.max(axis=0), V.min(axis=0))
    anti = np.where(is_benefit, V.min(axis=0), V.max(axis=0))

    dist_ideal = np.sqrt(((V - ideal) ** 2).sum(axis=1))
    dist_anti = np.sqrt(((V - anti) ** 2).sum(axis=1))

    score = dist_anti / (dist_ideal + dist_anti + 1e-12)

    out = df.copy()
    out["topsis_score"] = score
    out["readiness_rank"] = out["topsis_score"].rank(ascending=False, method="dense").astype(int)

    return out.sort_values("readiness_rank")


# ============================================================
# 6. M3 - Phân bổ vùng/hạng mục
# ============================================================

def region_priority_weights(scenario_code):
    """
    Tạo trọng số phân bổ ngân sách cho 6 vùng theo từng kịch bản.
    """
    regions = get_regions_data()

    readiness = minmax_good(regions["topsis_score"] if "topsis_score" in regions else regions["ai_readiness"])
    digital = minmax_good(regions["digital_index"])
    fdi = minmax_good(regions["fdi"])
    inverse_digital = minmax_bad(regions["digital_index"])
    inverse_trained = minmax_bad(regions["trained_labor"])
    gini_need = minmax_good(regions["gini"])

    if scenario_code == "S1":
        raw = 0.50 * fdi + 0.30 * minmax_good(regions["grdp_per_capita"]) + 0.20
    elif scenario_code == "S2":
        raw = 0.55 * digital + 0.25 * readiness + 0.20
    elif scenario_code == "S3":
        raw = 0.65 * minmax_good(regions["ai_readiness"]) + 0.25 * minmax_good(regions["rd_intensity"]) + 0.10
    elif scenario_code == "S4":
        raw = 0.45 * inverse_digital + 0.30 * inverse_trained + 0.15 * gini_need + 0.10
    else:
        raw = (
            0.25 * readiness
            + 0.25 * inverse_digital
            + 0.20 * minmax_good(regions["ai_readiness"])
            + 0.15 * minmax_good(regions["grdp_per_capita"])
            + 0.15
        )

    raw = np.maximum(raw, 1e-9)
    weights = raw / raw.sum()

    return regions[["region", "region_name"]].assign(region_weight=weights)


def allocate_budget_by_region(scenario_code, shares, annual_budget=1000.0, years=None):
    """
    Phân bổ ngân sách tổng theo vùng và hạng mục.
    """
    if years is None:
        years = YEARS

    region_w = region_priority_weights(scenario_code)
    shares = normalize_shares(shares)
    total_budget = annual_budget * len(years)

    rows = []
    for _, row in region_w.iterrows():
        region_budget = total_budget * row["region_weight"]
        for item_idx, item in enumerate(ITEMS):
            rows.append({
                "scenario": scenario_code,
                "region": row["region"],
                "region_name": row["region_name"],
                "item": item,
                "item_name": ITEM_LABELS[item],
                "budget": region_budget * shares[item_idx],
                "region_weight": row["region_weight"],
            })

    return pd.DataFrame(rows)


# ============================================================
# 7. M4 - Mô phỏng lao động
# ============================================================

def compute_labor_impact(shares, annual_budget=1000.0, years=None):
    """
    Tính NetJob theo 8 ngành từ tổng đầu tư AI và H.
    """
    if years is None:
        years = YEARS

    shares = normalize_shares(shares)
    sectors = get_sector_labor_data()

    total_budget = annual_budget * len(years)
    total_ai_budget = shares[2] * total_budget
    total_h_budget = shares[3] * total_budget

    risk = sectors["risk_pct"].to_numpy(dtype=float) / 100.0
    labor = sectors["labor_million"].to_numpy(dtype=float)

    # Trọng số phân bổ AI ưu tiên ngành có readiness/job creation cao.
    ai_raw = sectors["a1_ai_job"].to_numpy(dtype=float) * (1 + risk)
    ai_weights = ai_raw / ai_raw.sum()

    # Trọng số đào tạo ưu tiên ngành lao động lớn và rủi ro cao.
    h_raw = labor * (1 + risk)
    h_weights = h_raw / h_raw.sum()

    x_ai = total_ai_budget * ai_weights
    x_h = total_h_budget * h_weights

    new_job = sectors["a1_ai_job"].to_numpy(dtype=float) * x_ai
    upgrade_job = sectors["b1_upgrade"].to_numpy(dtype=float) * x_h
    displaced = sectors["c1_displace"].to_numpy(dtype=float) * risk * x_ai
    retrain_capacity = sectors["d1_retrain"].to_numpy(dtype=float) * x_h
    net_job = new_job + upgrade_job - displaced

    out = sectors.copy()
    out["x_AI"] = x_ai
    out["x_H"] = x_h
    out["NewJob_AI"] = new_job
    out["UpgradeJob"] = upgrade_job
    out["DisplacedJob"] = displaced
    out["RetrainingCapacity"] = retrain_capacity
    out["NetJob"] = net_job
    out["LossCap_5pct_labor"] = labor * 1_000_000 * 0.05
    out["Alert_NetJob_Negative"] = out["NetJob"] < 0
    out["Alert_Displaced_Over_Capacity"] = out["DisplacedJob"] > out["RetrainingCapacity"]
    out["Alert_Displaced_Over_5pct"] = out["DisplacedJob"] > out["LossCap_5pct_labor"]

    return out


# ============================================================
# 8. M5 - Rủi ro
# ============================================================

def compute_risk_metrics(scenario_code, shares, macro_df, region_alloc_df, labor_df):
    """
    Tính các rủi ro chính: cyber, environmental, dependency, labor.
    """
    shares = normalize_shares(shares)
    final = macro_df.sort_values("year").iloc[-1]

    regions = get_regions_data()
    reg_sum = region_alloc_df.pivot_table(
        index="region",
        columns="item",
        values="budget",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()

    merged = regions.merge(reg_sum, on="region", how="left")
    for col in ITEMS:
        if col not in merged.columns:
            merged[col] = 0.0

    total_budget = merged[ITEMS].sum().sum()
    if total_budget <= 0:
        total_budget = 1.0

    cyber_risk = (
        25
        + 45 * shares[2]
        + 0.08 * final["AI_capacity"]
        - 0.30 * final["human_capital_pct"]
        - 0.10 * final["digital_gdp_share_pct"]
    )
    cyber_risk = clamp(cyber_risk, 0, 100)

    environmental_risk = (
        20
        + 55 * shares[0]
        + 35 * shares[2]
        + 100 * (
            (merged["emission_coef"] * (merged["K"] + merged["AI"])).sum()
            / total_budget
        )
    )
    environmental_risk = clamp(environmental_risk / 2.0, 0, 100)

    dependency_risk = (
        30
        + 35 * shares[2]
        + 15 * shares[1]
        - 20 * shares[3]
        - 0.05 * final["human_capital_pct"]
    )
    dependency_risk = clamp(dependency_risk, 0, 100)

    labor_risk = 0
    if labor_df["Alert_NetJob_Negative"].any():
        labor_risk += 35
    if labor_df["Alert_Displaced_Over_Capacity"].any():
        labor_risk += 35
    if labor_df["Alert_Displaced_Over_5pct"].any():
        labor_risk += 30
    labor_risk = clamp(labor_risk, 0, 100)

    total_risk = (
        0.30 * cyber_risk
        + 0.25 * environmental_risk
        + 0.20 * dependency_risk
        + 0.25 * labor_risk
    )

    return pd.DataFrame({
        "scenario": [scenario_code],
        "Cyber risk": [cyber_risk],
        "Environmental risk": [environmental_risk],
        "Dependency risk": [dependency_risk],
        "Labor transition risk": [labor_risk],
        "Total risk": [total_risk],
    })


def make_alerts(risk_df, labor_df, risk_threshold=60.0):
    """
    Sinh bảng cảnh báo từ risk và lao động.
    """
    alerts = []

    row = risk_df.iloc[0]
    for metric in ["Cyber risk", "Environmental risk", "Dependency risk", "Labor transition risk", "Total risk"]:
        value = float(row[metric])
        if value >= risk_threshold:
            alerts.append({
                "Mức": "Cao",
                "Nhóm rủi ro": metric,
                "Giá trị": value,
                "Cảnh báo": f"{metric} vượt ngưỡng {risk_threshold:.0f}.",
                "Khuyến nghị": "Điều chỉnh giảm AI/K quá nhanh hoặc tăng H, D và cơ chế quản trị rủi ro.",
            })
        elif value >= risk_threshold * 0.75:
            alerts.append({
                "Mức": "Trung bình",
                "Nhóm rủi ro": metric,
                "Giá trị": value,
                "Cảnh báo": f"{metric} đang tiến gần ngưỡng cảnh báo.",
                "Khuyến nghị": "Theo dõi thêm và bổ sung biện pháp giảm thiểu.",
            })

    bad_net = labor_df[labor_df["Alert_NetJob_Negative"]]
    for _, sec in bad_net.iterrows():
        alerts.append({
            "Mức": "Cao",
            "Nhóm rủi ro": "NetJob ngành",
            "Giá trị": float(sec["NetJob"]),
            "Cảnh báo": f"Ngành {sec['sector']} có NetJob âm.",
            "Khuyến nghị": "Tăng ngân sách đào tạo lại H hoặc giảm tốc độ tự động hóa trong ngành.",
        })

    over_capacity = labor_df[labor_df["Alert_Displaced_Over_Capacity"]]
    for _, sec in over_capacity.iterrows():
        alerts.append({
            "Mức": "Cao",
            "Nhóm rủi ro": "Đào tạo lại",
            "Giá trị": float(sec["DisplacedJob"] - sec["RetrainingCapacity"]),
            "Cảnh báo": f"Ngành {sec['sector']} có lao động dịch chuyển vượt năng lực đào tạo lại.",
            "Khuyến nghị": "Tăng x_H hoặc xây dựng chương trình reskilling riêng cho ngành.",
        })

    if not alerts:
        alerts.append({
            "Mức": "Ổn định",
            "Nhóm rủi ro": "Tổng hợp",
            "Giá trị": float(row["Total risk"]),
            "Cảnh báo": "Chưa có rủi ro nào vượt ngưỡng.",
            "Khuyến nghị": "Có thể duy trì kịch bản, đồng thời tiếp tục giám sát định kỳ.",
        })

    return pd.DataFrame(alerts)


# ============================================================
# 9. S5 - Tối ưu cân bằng
# ============================================================

def evaluate_candidate_for_s5(shares, annual_budget, policy_weights):
    """
    Đánh giá một phương án phân bổ để chọn S5.
    """
    macro = simulate_macro_path(shares, annual_budget=annual_budget, scenario_code="candidate")
    labor = compute_labor_impact(shares, annual_budget=annual_budget)
    region_alloc = allocate_budget_by_region("S5", shares, annual_budget=annual_budget)
    risk = compute_risk_metrics("candidate", shares, macro, region_alloc, labor)

    final = macro.sort_values("year").iloc[-1]

    return {
        "GDP_2030": float(final["GDP"]),
        "Digital_2030": float(final["digital_gdp_share_pct"]),
        "AI_2030": float(final["AI_capacity"]),
        "H_2030": float(final["human_capital_pct"]),
        "Total_NetJob": float(labor["NetJob"].sum()),
        "Total_Risk": float(risk["Total risk"].iloc[0]),
        "Cyber_Risk": float(risk["Cyber risk"].iloc[0]),
        "Environmental_Risk": float(risk["Environmental risk"].iloc[0]),
        "Dependency_Risk": float(risk["Dependency risk"].iloc[0]),
        "Labor_Risk": float(risk["Labor transition risk"].iloc[0]),
    }


def optimize_s5_allocation(annual_budget, policy_weights):
    """
    Tìm phân bổ S5 bằng cách quét lưới và chấm điểm đa tiêu chí.
    """
    candidates = get_candidate_allocations(step=0.05)

    rows = []
    for idx, shares in enumerate(candidates):
        metrics = evaluate_candidate_for_s5(shares, annual_budget, policy_weights)
        row = {
            "candidate_id": idx,
            "K_share": shares[0],
            "D_share": shares[1],
            "AI_share": shares[2],
            "H_share": shares[3],
            **metrics,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    df["score_growth"] = minmax_good(df["GDP_2030"])
    df["score_digital"] = minmax_good(df["Digital_2030"])
    df["score_jobs"] = minmax_good(df["Total_NetJob"])
    df["score_risk"] = minmax_bad(df["Total_Risk"])
    df["score_green"] = minmax_bad(df["Environmental_Risk"])

    w_growth = policy_weights["growth"]
    w_digital = policy_weights["digital"]
    w_jobs = policy_weights["jobs"]
    w_risk = policy_weights["risk"]
    w_green = policy_weights["green"]

    total_w = w_growth + w_digital + w_jobs + w_risk + w_green
    if total_w <= 0:
        total_w = 1.0

    df["balanced_score"] = (
        w_growth * df["score_growth"]
        + w_digital * df["score_digital"]
        + w_jobs * df["score_jobs"]
        + w_risk * df["score_risk"]
        + w_green * df["score_green"]
    ) / total_w

    best = df.sort_values("balanced_score", ascending=False).iloc[0]
    best_shares = np.array([
        best["K_share"],
        best["D_share"],
        best["AI_share"],
        best["H_share"],
    ])

    return best_shares, df.sort_values("balanced_score", ascending=False)


# ============================================================
# 10. Pipeline tích hợp M1-M5
# ============================================================

def run_aideom_pipeline(annual_budget=1000.0, policy_weights=None):
    """
    Chạy toàn bộ pipeline AIDEOM-VN cho S1-S5.
    """
    if policy_weights is None:
        policy_weights = {
            "growth": 0.30,
            "digital": 0.20,
            "jobs": 0.20,
            "risk": 0.20,
            "green": 0.10,
        }

    s5_shares, s5_candidates = optimize_s5_allocation(annual_budget, policy_weights)

    scenario_shares = {
        **BASE_SCENARIO_SHARES,
        "S5": s5_shares,
    }

    macro_all = []
    region_all = []
    labor_all = []
    risk_all = []
    alerts_all = []

    for scenario_code, shares in scenario_shares.items():
        macro = simulate_macro_path(
            shares=shares,
            annual_budget=annual_budget,
            scenario_code=scenario_code,
        )
        region_alloc = allocate_budget_by_region(
            scenario_code=scenario_code,
            shares=shares,
            annual_budget=annual_budget,
        )
        labor = compute_labor_impact(
            shares=shares,
            annual_budget=annual_budget,
        )
        labor["scenario"] = scenario_code

        risk = compute_risk_metrics(
            scenario_code=scenario_code,
            shares=shares,
            macro_df=macro,
            region_alloc_df=region_alloc,
            labor_df=labor,
        )

        alerts = make_alerts(risk, labor)
        alerts["scenario"] = scenario_code

        macro_all.append(macro)
        region_all.append(region_alloc)
        labor_all.append(labor)
        risk_all.append(risk)
        alerts_all.append(alerts)

    macro_df = pd.concat(macro_all, ignore_index=True)
    region_df = pd.concat(region_all, ignore_index=True)
    labor_df = pd.concat(labor_all, ignore_index=True)
    risk_df = pd.concat(risk_all, ignore_index=True)
    alerts_df = pd.concat(alerts_all, ignore_index=True)

    shares_df = pd.DataFrame([
        {
            "scenario": s,
            "scenario_name": SCENARIO_NAMES[s],
            "K": v[0],
            "D": v[1],
            "AI": v[2],
            "H": v[3],
        }
        for s, v in scenario_shares.items()
    ])

    return {
        "shares": shares_df,
        "macro": macro_df,
        "region_alloc": region_df,
        "labor": labor_df,
        "risk": risk_df,
        "alerts": alerts_df,
        "s5_candidates": s5_candidates,
        "region_readiness": compute_region_topsis(),
        "module_structure": get_module_structure(),
    }


def make_summary_table(results):
    """
    Tạo bảng KPI 2030 cho các kịch bản.
    """
    macro = results["macro"]
    labor = results["labor"]
    risk = results["risk"]
    shares = results["shares"]

    final_2030 = macro[macro["year"] == macro["year"].max()].copy()

    labor_sum = labor.groupby("scenario", as_index=False).agg({
        "NetJob": "sum",
        "DisplacedJob": "sum",
        "RetrainingCapacity": "sum",
    })

    summary = final_2030.merge(labor_sum, on="scenario", how="left")
    summary = summary.merge(risk, on="scenario", how="left")
    summary = summary.merge(shares, on="scenario", how="left", suffixes=("", "_share"))

    summary["scenario_name"] = summary["scenario"].map(SCENARIO_NAMES)

    cols = [
        "scenario",
        "scenario_name",
        "GDP",
        "GDP_growth_pct",
        "digital_gdp_share_pct",
        "AI_capacity",
        "human_capital_pct",
        "NetJob",
        "DisplacedJob",
        "RetrainingCapacity",
        "Cyber risk",
        "Environmental risk",
        "Dependency risk",
        "Labor transition risk",
        "Total risk",
        "K_share",
        "D_share",
        "AI_share",
        "H_share",
    ]

    rename_map = {
        "GDP": "GDP 2030",
        "GDP_growth_pct": "Tăng trưởng GDP 2030 (%)",
        "digital_gdp_share_pct": "Kinh tế số/GDP 2030 (%)",
        "AI_capacity": "Năng lực AI 2030",
        "human_capital_pct": "Nhân lực số 2030 (%)",
        "NetJob": "NetJob tổng",
        "DisplacedJob": "Việc làm dịch chuyển",
        "RetrainingCapacity": "Năng lực đào tạo lại",
        "Total risk": "Rủi ro tổng",
    }

    summary = summary[cols].rename(columns=rename_map)

    return summary.sort_values("scenario")


# ============================================================
# 11. Hàm vẽ biểu đồ
# ============================================================

def plot_line(df, x, y, color, title):
    if PLOTLY_AVAILABLE:
        fig = px.line(df, x=x, y=y, color=color, markers=True, title=title)
        fig.update_layout(height=430)
        st.plotly_chart(fig, use_container_width=True)
    else:
        pivot = df.pivot_table(index=x, columns=color, values=y, aggfunc="mean")
        st.line_chart(pivot)


def plot_bar(df, x, y, color, title):
    if PLOTLY_AVAILABLE:
        fig = px.bar(df, x=x, y=y, color=color, barmode="group", title=title)
        fig.update_layout(height=430)
        st.plotly_chart(fig, use_container_width=True)
    else:
        pivot = df.pivot_table(index=x, columns=color, values=y, aggfunc="sum")
        st.bar_chart(pivot)


def plot_heatmap(pivot_df, title):
    if PLOTLY_AVAILABLE:
        fig = px.imshow(
            pivot_df,
            text_auto=".1f",
            aspect="auto",
            title=title,
            labels=dict(x="Hạng mục", y="Vùng", color="Ngân sách"),
        )
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(pivot_df, use_container_width=True)


def plot_radar_for_scenarios(summary_df):
    """
    Vẽ radar đơn giản cho các KPI đã chuẩn hóa.
    """
    if not PLOTLY_AVAILABLE:
        st.info("Chưa có plotly, bỏ qua biểu đồ radar.")
        return

    df = summary_df.copy()

    metrics = [
        "GDP 2030",
        "Kinh tế số/GDP 2030 (%)",
        "Năng lực AI 2030",
        "Nhân lực số 2030 (%)",
        "NetJob tổng",
        "Rủi ro tổng",
    ]

    norm = pd.DataFrame()
    norm["scenario_name"] = df["scenario_name"]

    for m in metrics:
        if m == "Rủi ro tổng":
            norm[m] = minmax_bad(df[m])
        else:
            norm[m] = minmax_good(df[m])

    fig = go.Figure()

    for _, row in norm.iterrows():
        values = [row[m] for m in metrics]
        values.append(values[0])
        theta = metrics + [metrics[0]]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=theta,
            fill="toself",
            name=row["scenario_name"],
        ))

    fig.update_layout(
        title="So sánh đa tiêu chí giữa các kịch bản",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=560,
    )
    st.plotly_chart(fig, use_container_width=True)


def style_numeric(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return df.style.format({col: "{:,.2f}" for col in numeric_cols})


# ============================================================
# 12. Streamlit dashboard
# ============================================================

def render():
    st.title("Bài 12. Đồ án tích hợp - Nguyên mẫu AIDEOM-VN")
    st.caption("Dashboard M6 tích hợp M1-M5: dự báo kinh tế, sẵn sàng số, phân bổ, lao động và rủi ro.")

    st.markdown(
        """
        Module này là bản **dashboard tích hợp** cho Bài 12.  
        App mô phỏng 5 kịch bản chính sách S1-S5, so sánh kết quả đến năm 2030,
        hiển thị phân bổ ngân sách theo vùng/hạng mục và tạo cảnh báo rủi ro.
        """
    )

    st.sidebar.header("Thiết lập Bài 12")

    annual_budget = st.sidebar.number_input(
        "Ngân sách mỗi năm, nghìn tỷ VND",
        min_value=100.0,
        max_value=5000.0,
        value=1000.0,
        step=100.0,
    )

    st.sidebar.subheader("Trọng số chọn S5 - Tối ưu cân bằng")

    w_growth = st.sidebar.slider("Tăng trưởng GDP", 0.0, 1.0, 0.30, 0.05)
    w_digital = st.sidebar.slider("Chuyển đổi số", 0.0, 1.0, 0.20, 0.05)
    w_jobs = st.sidebar.slider("Việc làm / bao trùm", 0.0, 1.0, 0.20, 0.05)
    w_risk = st.sidebar.slider("Giảm rủi ro", 0.0, 1.0, 0.20, 0.05)
    w_green = st.sidebar.slider("Môi trường xanh", 0.0, 1.0, 0.10, 0.05)

    risk_threshold = st.sidebar.slider(
        "Ngưỡng cảnh báo rủi ro",
        min_value=30.0,
        max_value=90.0,
        value=60.0,
        step=5.0,
    )

    policy_weights = {
        "growth": w_growth,
        "digital": w_digital,
        "jobs": w_jobs,
        "risk": w_risk,
        "green": w_green,
    }

    with st.spinner("Đang chạy pipeline tích hợp AIDEOM-VN..."):
        results = run_aideom_pipeline(
            annual_budget=float(annual_budget),
            policy_weights=policy_weights,
        )

    summary_df = make_summary_table(results)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Tổng quan",
        "Phân bổ",
        "Kịch bản so sánh",
        "Cảnh báo rủi ro",
        "Kỹ thuật & bàn giao",
    ])

    # ========================================================
    # Tab 1 - Tổng quan
    # ========================================================
    with tab1:
        st.subheader("1. Tổng quan kết quả 2030")

        s5_row = summary_df[summary_df["scenario"] == "S5"].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("GDP 2030 - S5", f"{s5_row['GDP 2030']:,.1f}")
        c2.metric("Kinh tế số/GDP - S5", f"{s5_row['Kinh tế số/GDP 2030 (%)']:,.2f}%")
        c3.metric("NetJob tổng - S5", f"{s5_row['NetJob tổng']:,.0f}")
        c4.metric("Rủi ro tổng - S5", f"{s5_row['Rủi ro tổng']:,.2f}")

        st.markdown("### Bảng KPI 2030 cho 5 kịch bản")
        st.dataframe(style_numeric(summary_df), use_container_width=True)

        st.markdown("### Cấu trúc 6 module AIDEOM-VN")
        st.dataframe(results["module_structure"], use_container_width=True)

        st.markdown("### Đường đi GDP 2026-2030")
        plot_line(
            results["macro"],
            x="year",
            y="GDP",
            color="scenario",
            title="GDP mô phỏng theo kịch bản",
        )

        st.markdown("### Tỷ lệ phân bổ của từng kịch bản")
        share_long = results["shares"].melt(
            id_vars=["scenario", "scenario_name"],
            value_vars=ITEMS,
            var_name="Hạng mục",
            value_name="Tỷ trọng",
        )
        share_long["Tên hạng mục"] = share_long["Hạng mục"].map(ITEM_LABELS)
        plot_bar(
            share_long,
            x="scenario_name",
            y="Tỷ trọng",
            color="Tên hạng mục",
            title="Cơ cấu phân bổ ngân sách theo kịch bản",
        )

    # ========================================================
    # Tab 2 - Phân bổ
    # ========================================================
    with tab2:
        st.subheader("2. Phân bổ ngân sách vùng/hạng mục")

        selected_scenario = st.selectbox(
            "Chọn kịch bản để xem phân bổ",
            options=list(SCENARIO_NAMES.keys()),
            format_func=lambda x: SCENARIO_NAMES[x],
        )

        st.info(SCENARIO_DESCRIPTIONS[selected_scenario])

        region_alloc = results["region_alloc"]
        region_selected = region_alloc[region_alloc["scenario"] == selected_scenario].copy()

        pivot = region_selected.pivot_table(
            index="region_name",
            columns="item_name",
            values="budget",
            aggfunc="sum",
            fill_value=0.0,
        )

        st.markdown("### Heatmap phân bổ ngân sách")
        plot_heatmap(pivot, f"Phân bổ ngân sách theo vùng và hạng mục - {SCENARIO_NAMES[selected_scenario]}")

        st.markdown("### Bảng phân bổ chi tiết")
        st.dataframe(style_numeric(region_selected), use_container_width=True)

        st.markdown("### Xếp hạng sẵn sàng AI vùng theo TOPSIS")
        readiness = results["region_readiness"]
        st.dataframe(
            style_numeric(readiness[[
                "readiness_rank",
                "region_name",
                "digital_index",
                "ai_readiness",
                "trained_labor",
                "rd_intensity",
                "gini",
                "topsis_score",
            ]]),
            use_container_width=True,
        )

        plot_bar(
            readiness,
            x="region_name",
            y="topsis_score",
            color="region_name",
            title="TOPSIS score - mức độ sẵn sàng AI theo vùng",
        )

    # ========================================================
    # Tab 3 - Kịch bản so sánh
    # ========================================================
    with tab3:
        st.subheader("3. So sánh 5 kịch bản chính sách")

        st.markdown("### Bảng so sánh tổng hợp")
        st.dataframe(style_numeric(summary_df), use_container_width=True)

        metric = st.selectbox(
            "Chọn chỉ tiêu để vẽ theo thời gian",
            options=[
                "GDP",
                "GDP_growth_pct",
                "digital_gdp_share_pct",
                "AI_capacity",
                "human_capital_pct",
            ],
            format_func=lambda x: {
                "GDP": "GDP",
                "GDP_growth_pct": "Tăng trưởng GDP (%)",
                "digital_gdp_share_pct": "Kinh tế số/GDP (%)",
                "AI_capacity": "Năng lực AI",
                "human_capital_pct": "Nhân lực số (%)",
            }[x],
        )

        plot_line(
            results["macro"],
            x="year",
            y=metric,
            color="scenario",
            title=f"So sánh {metric} theo thời gian",
        )

        st.markdown("### Radar đa tiêu chí")
        plot_radar_for_scenarios(summary_df)

        st.markdown("### Top ứng viên S5")
        candidates = results["s5_candidates"].head(10).copy()
        st.dataframe(style_numeric(candidates), use_container_width=True)

        best_s5 = results["shares"][results["shares"]["scenario"] == "S5"].iloc[0]
        st.success(
            "Phân bổ S5 tối ưu cân bằng hiện tại: "
            f"K={best_s5['K']:.0%}, D={best_s5['D']:.0%}, "
            f"AI={best_s5['AI']:.0%}, H={best_s5['H']:.0%}."
        )

    # ========================================================
    # Tab 4 - Cảnh báo rủi ro
    # ========================================================
    with tab4:
        st.subheader("4. Cảnh báo rủi ro")

        risk_df = results["risk"].copy()
        risk_df["scenario_name"] = risk_df["scenario"].map(SCENARIO_NAMES)

        st.markdown("### Bảng rủi ro theo kịch bản")
        st.dataframe(style_numeric(risk_df), use_container_width=True)

        risk_long = risk_df.melt(
            id_vars=["scenario", "scenario_name"],
            value_vars=[
                "Cyber risk",
                "Environmental risk",
                "Dependency risk",
                "Labor transition risk",
                "Total risk",
            ],
            var_name="Nhóm rủi ro",
            value_name="Điểm rủi ro",
        )

        plot_bar(
            risk_long,
            x="scenario_name",
            y="Điểm rủi ro",
            color="Nhóm rủi ro",
            title="So sánh rủi ro theo kịch bản",
        )

        selected_alert_scenario = st.selectbox(
            "Chọn kịch bản để xem cảnh báo chi tiết",
            options=list(SCENARIO_NAMES.keys()),
            format_func=lambda x: SCENARIO_NAMES[x],
            key="alert_scenario",
        )

        selected_risk = risk_df[risk_df["scenario"] == selected_alert_scenario]
        selected_labor = results["labor"][results["labor"]["scenario"] == selected_alert_scenario]
        alert_table = make_alerts(
            selected_risk,
            selected_labor,
            risk_threshold=float(risk_threshold),
        )

        st.markdown("### Cảnh báo tự động")
        st.dataframe(style_numeric(alert_table), use_container_width=True)

        st.markdown("### Mô phỏng lao động theo ngành")
        st.dataframe(
            style_numeric(selected_labor[[
                "sector",
                "labor_million",
                "risk_pct",
                "x_AI",
                "x_H",
                "NewJob_AI",
                "UpgradeJob",
                "DisplacedJob",
                "RetrainingCapacity",
                "NetJob",
                "Alert_NetJob_Negative",
                "Alert_Displaced_Over_Capacity",
                "Alert_Displaced_Over_5pct",
            ]]),
            use_container_width=True,
        )

        labor_chart = selected_labor[["sector", "NetJob", "DisplacedJob", "RetrainingCapacity"]]
        plot_bar(
            labor_chart.melt(id_vars="sector", var_name="Chỉ tiêu", value_name="Số việc làm"),
            x="sector",
            y="Số việc làm",
            color="Chỉ tiêu",
            title=f"Lao động theo ngành - {SCENARIO_NAMES[selected_alert_scenario]}",
        )

    # ========================================================
    # Tab 5 - Kỹ thuật & bàn giao
    # ========================================================
    with tab5:
        st.subheader("5. Kỹ thuật triển khai và sản phẩm bàn giao")

        st.markdown(
            """
            **Module này đáp ứng các yêu cầu chính của Bài 12:**

            - Có dashboard Streamlit tích hợp M1-M5.
            - Có tối thiểu 4 tab: Tổng quan, Phân bổ, Kịch bản so sánh, Cảnh báo rủi ro.
            - Có 5 kịch bản chính sách S1-S5.
            - Có bảng so sánh kết quả 2030.
            - Có cảnh báo rủi ro và khuyến nghị chính sách.
            - Có các hàm Python độc lập, có docstring, có thể viết unit test bằng pytest.

            **Gợi ý kiểm thử nhanh bằng pytest:**

            ```python
            from bai12_aideom_dashboard import (
                simulate_macro_path,
                compute_labor_impact,
                run_aideom_pipeline
            )

            def test_macro_path_has_2030():
                df = simulate_macro_path([0.25, 0.25, 0.25, 0.25])
                assert 2030 in df["year"].values

            def test_labor_netjob_column_exists():
                df = compute_labor_impact([0.25, 0.25, 0.25, 0.25])
                assert "NetJob" in df.columns

            def test_pipeline_has_five_scenarios():
                result = run_aideom_pipeline()
                assert result["shares"]["scenario"].nunique() == 5
            ```
            """
        )

        st.markdown("### Gợi ý requirements.txt")
        st.code(
            """
numpy
pandas
streamlit
plotly
            """.strip(),
            language="text",
        )

        st.markdown("### Gợi ý gọi trong streamlit_app.py")
        st.code(
            """
import bai12_aideom_dashboard

# Thêm vào menu:
"Bài 12 - AIDEOM-VN Dashboard": bai12_aideom_dashboard
            """.strip(),
            language="python",
        )

        st.warning(
            "Nếu file của bạn đang có tên khác, ví dụ bai12_nsga2_topsis.py, "
            "thì trong streamlit_app.py phải import đúng tên file đó."
        )


def run():
    """
    App chính có thể gọi run() thay vì render().
    """
    render()

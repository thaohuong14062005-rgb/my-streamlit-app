"S3. AI dẫn dắt": "#2A9D8F",
"S4. Bao trùm số": "#F4A261",
"S5. Tối ưu cân bằng": "#8E44AD",
    "K": "#053151",
    "D": "#E76F51",
    "AI": "#2A9D8F",
    "H": "#F4A261",
"GDP": "#053151",
"Digital Index": "#E76F51",
"AI Readiness": "#2A9D8F",
"NetJob": "#F4A261",
    "Risk": "#BC4749",
    "Risk score": "#BC4749",
}


# =========================================================
# BÀI 12 — AIDEOM-VN INTEGRATED DASHBOARD
# =========================================================


YEARS = np.arange(2026, 2031)

MODULES = [
@@ -75,7 +74,8 @@
},
]

SCENARIOS = [

BASE_SCENARIOS = [
{
"Kịch bản": "S1. Truyền thống",
"Mô tả ngắn": "Tập trung vốn vật chất, FDI, hạ tầng truyền thống, xuất khẩu",
@@ -84,6 +84,7 @@
"D": 0.10,
"AI": 0.10,
"H": 0.10,
        "Loại": "Cố định",
},
{
"Kịch bản": "S2. Số hóa nhanh",
@@ -93,6 +94,7 @@
"D": 0.45,
"AI": 0.15,
"H": 0.15,
        "Loại": "Cố định",
},
{
"Kịch bản": "S3. AI dẫn dắt",
@@ -102,6 +104,7 @@
"D": 0.20,
"AI": 0.45,
"H": 0.15,
        "Loại": "Cố định",
},
{
"Kịch bản": "S4. Bao trùm số",
@@ -111,21 +114,13 @@
"D": 0.20,
"AI": 0.10,
"H": 0.40,
    },
    {
        "Kịch bản": "S5. Tối ưu cân bằng",
        "Mô tả ngắn": "Kết quả mô hình AIDEOM-VN tự chạy ra",
        "Đặc điểm phân bổ": "Tối ưu cân bằng K/D/AI/H",
        "K": 0.35,
        "D": 0.25,
        "AI": 0.22,
        "H": 0.18,
        "Loại": "Cố định",
},
]


def find_file(filename):
    possible_paths = [
    paths = [
filename,
f"./{filename}",
f"data/{filename}",
@@ -134,7 +129,7 @@ def find_file(filename):
f"./datasets/{filename}",
]

    for path in possible_paths:
    for path in paths:
if os.path.exists(path):
return path

@@ -197,87 +192,106 @@ def load_macro_data():
path = find_file("vietnam_macro_2020_2025.csv")

if path is None:
        return fallback_macro_data()
        return fallback_macro_data(), "Fallback"

try:
df = pd.read_csv(path)

        needed = ["year", "GDP_trillion_VND", "GDP_growth_pct"]
        required = ["year", "GDP_trillion_VND", "GDP_growth_pct"]

        for col in needed:
            if col not in df.columns:
                return fallback_macro_data()
        if any(col not in df.columns for col in required):
            return fallback_macro_data(), "Fallback"

for col in df.columns:
if col != "year":
df[col] = pd.to_numeric(df[col], errors="coerce")

        return df.dropna(subset=needed).reset_index(drop=True)
        df = df.dropna(subset=required).reset_index(drop=True)

        if len(df) == 0:
            return fallback_macro_data(), "Fallback"

        return df, path

except Exception:
        return fallback_macro_data()
        return fallback_macro_data(), "Fallback"


def load_region_data():
path = find_file("vietnam_regions_2024.csv")

if path is None:
        return fallback_region_data()
        return fallback_region_data(), "Fallback"

try:
df = pd.read_csv(path)

        needed = ["region_name_vi", "digital_index_0_100", "ai_readiness_0_100"]
        required = ["region_name_vi", "digital_index_0_100", "ai_readiness_0_100"]

        for col in needed:
            if col not in df.columns:
                return fallback_region_data()
        if any(col not in df.columns for col in required):
            return fallback_region_data(), "Fallback"

for col in df.columns:
            if col != "region_name_vi" and df[col].dtype != "object":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            if col != "region_name_vi":
                df[col] = pd.to_numeric(df[col], errors="ignore")

        df = df.dropna(subset=required).reset_index(drop=True)

        return df.dropna(subset=needed).reset_index(drop=True)
        if len(df) == 0:
            return fallback_region_data(), "Fallback"

        return df, path

except Exception:
        return fallback_region_data()
        return fallback_region_data(), "Fallback"


def load_sector_data():
path = find_file("vietnam_sectors_2024.csv")

if path is None:
        return fallback_sector_data()
        return fallback_sector_data(), "Fallback"

try:
df = pd.read_csv(path)

        needed = ["sector_name_vi", "ai_readiness_0_100"]
        required = ["sector_name_vi", "ai_readiness_0_100"]

        for col in needed:
            if col not in df.columns:
                return fallback_sector_data()
        if any(col not in df.columns for col in required):
            return fallback_sector_data(), "Fallback"

for col in df.columns:
            if col != "sector_name_vi" and df[col].dtype != "object":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            if col != "sector_name_vi":
                df[col] = pd.to_numeric(df[col], errors="ignore")

        return df.dropna(subset=needed).reset_index(drop=True)
        df = df.dropna(subset=required).reset_index(drop=True)

    except Exception:
        return fallback_sector_data()
        if len(df) == 0:
            return fallback_sector_data(), "Fallback"

        return df, path

def scenario_df():
    return pd.DataFrame(SCENARIOS)
    except Exception:
        return fallback_sector_data(), "Fallback"


def modules_df():
return pd.DataFrame(MODULES)


def normalize_weights(weights):
    total = sum(weights.values())

    if total <= 0:
        n = len(weights)
        return {k: 1.0 / n for k in weights}

    return {k: v / total for k, v in weights.items()}


def base_values_from_data(macro, regions, sectors):
    last = macro.sort_values("year").iloc[-1]
    macro = macro.sort_values("year").reset_index(drop=True)
    last = macro.iloc[-1]

gdp0 = float(last.get("GDP_trillion_VND", 8550))
growth0 = float(last.get("GDP_growth_pct", 6.5))
@@ -292,19 +306,31 @@ def base_values_from_data(macro, regions, sectors):
else:
productivity0 = 198.0

    digital_index0 = float(regions["digital_index_0_100"].mean())
    ai_ready0 = float(regions["ai_readiness_0_100"].mean())
    digital_index0 = float(pd.to_numeric(regions["digital_index_0_100"], errors="coerce").mean())
    ai_ready0 = float(pd.to_numeric(regions["ai_readiness_0_100"], errors="coerce").mean())

if "trained_labor_pct" in regions.columns:
        trained0 = float(regions["trained_labor_pct"].mean())
        trained0 = float(pd.to_numeric(regions["trained_labor_pct"], errors="coerce").mean())
else:
trained0 = 32.0

if "automation_risk_pct" in sectors.columns:
        automation_risk0 = float(sectors["automation_risk_pct"].mean())
        automation_risk0 = float(pd.to_numeric(sectors["automation_risk_pct"], errors="coerce").mean())
else:
automation_risk0 = 32.5

    if np.isnan(digital_index0):
        digital_index0 = 60.0

    if np.isnan(ai_ready0):
        ai_ready0 = 55.0

    if np.isnan(trained0):
        trained0 = 32.0

    if np.isnan(automation_risk0):
        automation_risk0 = 32.5

return {
"gdp0": gdp0,
"growth0": growth0,
@@ -317,103 +343,252 @@ def base_values_from_data(macro, regions, sectors):
}


def simulate_scenarios(macro, regions, sectors, annual_budget=1000):
    base = base_values_from_data(macro, regions, sectors)

def simulate_one_scenario(scenario, base, annual_budget=1000):
rows = []

    for sc in SCENARIOS:
        gdp = base["gdp0"]
        digital_index = base["digital_index0"]
        ai_ready = base["ai_ready0"]
        trained = base["trained0"]
        risk = base["automation_risk0"]
        digital_share = base["digital_share0"]

        for idx, year in enumerate(YEARS):
            k = sc["K"]
            d = sc["D"]
            ai = sc["AI"]
            h = sc["H"]

            growth = (
                4.6
                + 1.00 * k
                + 1.45 * d
                + 1.70 * ai
                + 0.95 * h
                + 0.018 * (digital_index - 50)
                + 0.012 * (ai_ready - 50)
            )

            inclusion_bonus = 1.8 * h + 0.8 * d
            environment_pressure = 14.0 * k + 18.0 * ai - 10.0 * d - 8.0 * h
            cyber_pressure = 20.0 * ai + 10.0 * d - 6.0 * h
            dependency_pressure = 16.0 * ai + 8.0 * d - 4.0 * h

            displaced = risk * (0.50 * ai + 0.20 * d) * 100
            upgrade = trained * h * 18
            new_job = 550 * ai + 260 * d + 180 * h
            net_job = new_job + upgrade - displaced

            risk_score = np.clip(
                35
                + 0.35 * environment_pressure
                + 0.32 * cyber_pressure
                + 0.22 * dependency_pressure
                - 0.20 * inclusion_bonus,
                0,
                100,
            )

            gdp = gdp * (1 + growth / 100)
            digital_index = min(100, digital_index + 3.8 * d + 1.4 * k + 1.1 * h)
            ai_ready = min(100, ai_ready + 4.2 * ai + 1.4 * d + 1.7 * h)
            trained = min(100, trained + 3.6 * h + 0.8 * d)
            digital_share = min(45, digital_share + 1.3 * d + 1.0 * ai + 0.4 * h)

            rows.append(
    gdp = base["gdp0"]
    digital_index = base["digital_index0"]
    ai_ready = base["ai_ready0"]
    trained = base["trained0"]
    digital_share = base["digital_share0"]
    automation_risk = base["automation_risk0"]

    for year in YEARS:
        k = float(scenario["K"])
        d = float(scenario["D"])
        ai = float(scenario["AI"])
        h = float(scenario["H"])

        growth = (
            4.40
            + 1.05 * k
            + 1.45 * d
            + 1.75 * ai
            + 1.05 * h
            + 0.018 * (digital_index - 55)
            + 0.013 * (ai_ready - 55)
        )

        cyber_pressure = 20.0 * ai + 9.0 * d - 8.0 * h
        environmental_pressure = 13.0 * k + 18.0 * ai - 8.0 * d - 7.0 * h
        dependency_pressure = 18.0 * ai + 8.0 * d - 5.0 * h
        labor_pressure = automation_risk * (0.50 * ai + 0.22 * d + 0.10 * k)

        new_job = 540 * ai + 280 * d + 120 * k
        upgrade_job = trained * h * 15
        displaced_job = labor_pressure * 18
        net_job = new_job + upgrade_job - displaced_job

        risk_score = np.clip(
            34
            + 0.32 * cyber_pressure
            + 0.28 * environmental_pressure
            + 0.25 * dependency_pressure
            + 0.15 * labor_pressure
            - 0.18 * trained * h,
            0,
            100,
        )

        gdp = gdp * (1 + growth / 100)
        digital_index = min(100, digital_index + 3.7 * d + 1.4 * k + 1.1 * h)
        ai_ready = min(100, ai_ready + 4.5 * ai + 1.2 * d + 1.5 * h)
        trained = min(100, trained + 3.7 * h + 0.7 * d)
        digital_share = min(45, digital_share + 1.35 * d + 1.05 * ai + 0.45 * h)

        rows.append(
            {
                "Năm": int(year),
                "Kịch bản": scenario["Kịch bản"],
                "GDP": gdp,
                "GDP growth (%)": growth,
                "Digital Index": digital_index,
                "AI Readiness": ai_ready,
                "Lao động đào tạo (%)": trained,
                "Kinh tế số/GDP (%)": digital_share,
                "NetJob": net_job,
                "DisplacedJob": displaced_job,
                "Risk score": risk_score,
                "Cyber risk": np.clip(35 + cyber_pressure, 0, 100),
                "Environmental risk": np.clip(35 + environmental_pressure, 0, 100),
                "Dependency risk": np.clip(35 + dependency_pressure, 0, 100),
                "Labor transition risk": np.clip(35 + labor_pressure - 0.25 * trained * h, 0, 100),
                "K": k * annual_budget,
                "D": d * annual_budget,
                "AI": ai * annual_budget,
                "H": h * annual_budget,
                "Tổng ngân sách": annual_budget,
            }
        )

    return pd.DataFrame(rows)


def absolute_candidate_score(row, score_weights):
    gdp_growth_score = np.clip((row["GDP growth (%)"] - 4.0) / 4.5, 0, 1)
    digital_score = np.clip(row["Digital Index"] / 100, 0, 1)
    ai_score = np.clip(row["AI Readiness"] / 100, 0, 1)
    netjob_score = np.clip((row["NetJob"] + 200) / 1400, 0, 1)
    risk_score = np.clip(1 - row["Risk score"] / 100, 0, 1)
    training_score = np.clip(row["Lao động đào tạo (%)"] / 100, 0, 1)

    score = (
        score_weights["GDP"] * gdp_growth_score
        + score_weights["Digital"] * digital_score
        + score_weights["AI"] * ai_score
        + score_weights["NetJob"] * netjob_score
        + score_weights["Risk"] * risk_score
        + score_weights["Training"] * training_score
    )

    return float(score)


def optimize_s5_allocation(base, score_weights, annual_budget=1000, step=0.05):
    candidates = []

    grid = np.arange(0.10, 0.501, step)

    for k in grid:
        for d in grid:
            for ai in grid:
                h = 1.0 - k - d - ai

                if h < 0.10 or h > 0.50:
                    continue

                if k < 0.15 or d < 0.15 or ai < 0.10:
                    continue

                if not np.isclose(k + d + ai + h, 1.0, atol=1e-8):
                    continue

                scenario = {
                    "Kịch bản": "Candidate",
                    "K": k,
                    "D": d,
                    "AI": ai,
                    "H": h,
                }

                path = simulate_one_scenario(scenario, base, annual_budget=annual_budget)
                terminal = path[path["Năm"] == 2030].iloc[0]
                score = absolute_candidate_score(terminal, score_weights)

                balance_penalty = 0.035 * (
                    abs(k - 0.25) + abs(d - 0.25) + abs(ai - 0.25) + abs(h - 0.25)
                )

                final_score = score - balance_penalty

                candidates.append(
                    {
                        "K": k,
                        "D": d,
                        "AI": ai,
                        "H": h,
                        "Raw score": score,
                        "Balance penalty": balance_penalty,
                        "Optimized score": final_score,
                        "GDP 2030": terminal["GDP"],
                        "Digital Index 2030": terminal["Digital Index"],
                        "AI Readiness 2030": terminal["AI Readiness"],
                        "NetJob 2030": terminal["NetJob"],
                        "Risk score 2030": terminal["Risk score"],
                    }
                )

    candidate_df = pd.DataFrame(candidates)

    if len(candidate_df) == 0:
        fallback = {
            "K": 0.35,
            "D": 0.25,
            "AI": 0.22,
            "H": 0.18,
        }

        candidate_df = pd.DataFrame(
            [
{
                    "Năm": int(year),
                    "Kịch bản": sc["Kịch bản"],
                    "GDP": gdp,
                    "GDP growth (%)": growth,
                    "Digital Index": digital_index,
                    "AI Readiness": ai_ready,
                    "Lao động đào tạo (%)": trained,
                    "Kinh tế số/GDP (%)": digital_share,
                    "NetJob": net_job,
                    "Risk score": risk_score,
                    "Cyber risk": np.clip(35 + cyber_pressure, 0, 100),
                    "Environmental risk": np.clip(35 + environment_pressure, 0, 100),
                    "Dependency risk": np.clip(35 + dependency_pressure, 0, 100),
                    "K": k * annual_budget,
                    "D": d * annual_budget,
                    "AI": ai * annual_budget,
                    "H": h * annual_budget,
                    "Tổng ngân sách": annual_budget,
                    **fallback,
                    "Raw score": 0.0,
                    "Balance penalty": 0.0,
                    "Optimized score": 0.0,
                    "GDP 2030": np.nan,
                    "Digital Index 2030": np.nan,
                    "AI Readiness 2030": np.nan,
                    "NetJob 2030": np.nan,
                    "Risk score 2030": np.nan,
}
            )
            ]
        )

    return pd.DataFrame(rows)
    candidate_df = candidate_df.sort_values("Optimized score", ascending=False).reset_index(drop=True)
    best = candidate_df.iloc[0].to_dict()

    s5 = {
        "Kịch bản": "S5. Tối ưu cân bằng",
        "Mô tả ngắn": "Kết quả mô hình AIDEOM-VN tự tối ưu theo trọng số KPI",
        "Đặc điểm phân bổ": (
            f"{best['K']:.0%} K + {best['D']:.0%} D + "
            f"{best['AI']:.0%} AI + {best['H']:.0%} H"
        ),
        "K": float(best["K"]),
        "D": float(best["D"]),
        "AI": float(best["AI"]),
        "H": float(best["H"]),
        "Loại": "Tối ưu tự động",
    }

def kpi_2030_table(sim_df):
    df = sim_df[sim_df["Năm"] == 2030].copy()
    return s5, candidate_df

    df["Xếp hạng GDP"] = df["GDP"].rank(ascending=False, method="min").astype(int)
    df["Xếp hạng AI"] = df["AI Readiness"].rank(ascending=False, method="min").astype(int)
    df["Xếp hạng Risk"] = df["Risk score"].rank(ascending=True, method="min").astype(int)

    score = (
        0.30 * normalize_series(df["GDP"])
        + 0.20 * normalize_series(df["Digital Index"])
        + 0.20 * normalize_series(df["AI Readiness"])
        + 0.15 * normalize_series(df["NetJob"])
        + 0.15 * normalize_series(-df["Risk score"])
def build_scenarios(s5):
    return BASE_SCENARIOS + [s5]


def simulate_scenarios(scenarios, macro, regions, sectors, annual_budget=1000):
    base = base_values_from_data(macro, regions, sectors)

    frames = []

    for scenario in scenarios:
        frames.append(simulate_one_scenario(scenario, base, annual_budget=annual_budget))

    return pd.concat(frames, ignore_index=True)


def normalize_series(x):
    arr = np.asarray(x, dtype=float)
    min_v = np.nanmin(arr)
    max_v = np.nanmax(arr)

    if np.isclose(max_v, min_v):
        return np.ones_like(arr) * 0.5

    return (arr - min_v) / (max_v - min_v)


def kpi_2030_table(sim_df, score_weights):
    df = sim_df[sim_df["Năm"] == 2030].copy().reset_index(drop=True)

    df["Score GDP"] = normalize_series(df["GDP"])
    df["Score Digital"] = normalize_series(df["Digital Index"])
    df["Score AI"] = normalize_series(df["AI Readiness"])
    df["Score NetJob"] = normalize_series(df["NetJob"])
    df["Score Risk"] = normalize_series(-df["Risk score"])
    df["Score Training"] = normalize_series(df["Lao động đào tạo (%)"])

    df["AIDEOM score"] = (
        score_weights["GDP"] * df["Score GDP"]
        + score_weights["Digital"] * df["Score Digital"]
        + score_weights["AI"] * df["Score AI"]
        + score_weights["NetJob"] * df["Score NetJob"]
        + score_weights["Risk"] * df["Score Risk"]
        + score_weights["Training"] * df["Score Training"]
)

    df["AIDEOM score"] = score
df["Xếp hạng tổng hợp"] = df["AIDEOM score"].rank(ascending=False, method="min").astype(int)

cols = [
@@ -425,6 +600,7 @@ def kpi_2030_table(sim_df):
"Lao động đào tạo (%)",
"Kinh tế số/GDP (%)",
"NetJob",
        "DisplacedJob",
"Risk score",
"AIDEOM score",
"Xếp hạng tổng hợp",
@@ -433,19 +609,8 @@ def kpi_2030_table(sim_df):
return df[cols].sort_values("Xếp hạng tổng hợp").reset_index(drop=True)


def normalize_series(x):
    arr = np.asarray(x, dtype=float)
    min_v = np.nanmin(arr)
    max_v = np.nanmax(arr)

    if np.isclose(max_v, min_v):
        return np.ones_like(arr) * 0.5

    return (arr - min_v) / (max_v - min_v)


def risk_alert_table(sim_df):
    df = sim_df[sim_df["Năm"] == 2030].copy()
    df = sim_df[sim_df["Năm"] == 2030].copy().reset_index(drop=True)

rows = []

@@ -456,13 +621,16 @@ def risk_alert_table(sim_df):
alerts.append("Rủi ro tổng hợp cao")

if row["Cyber risk"] >= 70:
            alerts.append("Rủi ro an ninh mạng cao")
            alerts.append("An ninh mạng cao")

if row["Environmental risk"] >= 70:
alerts.append("Áp lực môi trường cao")

if row["Dependency risk"] >= 70:
            alerts.append("Rủi ro phụ thuộc công nghệ cao")
            alerts.append("Phụ thuộc công nghệ cao")

        if row["Labor transition risk"] >= 70:
            alerts.append("Dịch chuyển lao động cao")

if row["NetJob"] < 0:
alerts.append("NetJob âm")
@@ -471,23 +639,28 @@ def risk_alert_table(sim_df):
alerts.append("AI nhanh hơn năng lực nhân lực")

if len(alerts) == 0:
            level = "Thấp"
            level = "Xanh"
            action = "Theo dõi định kỳ"
alert_text = "Không có cảnh báo lớn"
elif row["Risk score"] >= 70 or row["NetJob"] < 0:
            level = "Cao"
            level = "Đỏ"
            action = "Cần can thiệp chính sách"
alert_text = "; ".join(alerts)
else:
            level = "Trung bình"
            level = "Vàng"
            action = "Cần theo dõi sát"
alert_text = "; ".join(alerts)

rows.append(
{
"Kịch bản": row["Kịch bản"],
"Mức cảnh báo": level,
                "Hành động quản trị": action,
"Risk score": row["Risk score"],
"Cyber risk": row["Cyber risk"],
"Environmental risk": row["Environmental risk"],
"Dependency risk": row["Dependency risk"],
                "Labor transition risk": row["Labor transition risk"],
"NetJob": row["NetJob"],
"Cảnh báo": alert_text,
}
@@ -497,33 +670,38 @@ def risk_alert_table(sim_df):


def recommendation_table(kpi_df, risk_df):
    merged = kpi_df.merge(risk_df[["Kịch bản", "Mức cảnh báo", "Cảnh báo"]], on="Kịch bản", how="left")
    merged = kpi_df.merge(
        risk_df[["Kịch bản", "Mức cảnh báo", "Cảnh báo"]],
        on="Kịch bản",
        how="left",
    )

rows = []

for _, row in merged.iterrows():
scenario = row["Kịch bản"]

if scenario.startswith("S1"):
            rec = "Không nên chọn làm chiến lược trung tâm; chỉ giữ vai trò nền tảng hạ tầng vì chuyển đổi số và AI readiness thấp."
priority = "Giảm tỷ trọng K, tăng D và H"
            rec = "Không nên chọn làm trục trung tâm. Phù hợp làm kịch bản đối chứng vì tăng trưởng truyền thống nhưng chuyển đổi số chậm."
elif scenario.startswith("S2"):
            rec = "Phù hợp để tăng tốc chính phủ số và kinh tế số; cần bổ sung an ninh dữ liệu và đào tạo kỹ năng số."
priority = "Tăng cyber governance và kỹ năng số"
            rec = "Phù hợp để tăng tốc chính phủ số và kinh tế số; cần kiểm soát rủi ro dữ liệu và đào tạo kỹ năng số."
elif scenario.startswith("S3"):
            rec = "Tạo tăng trưởng và AI readiness cao nhưng cần kiểm soát rủi ro tự động hóa, phụ thuộc công nghệ và an ninh mạng."
priority = "AI đi kèm H, an ninh mạng và chuẩn dữ liệu"
            rec = "Tạo AI readiness cao, nhưng không nên triển khai nguyên trạng nếu rủi ro lao động và phụ thuộc công nghệ tăng mạnh."
elif scenario.startswith("S4"):
            rec = "Phù hợp mục tiêu bao trùm và an sinh; tốc độ tăng trưởng có thể thấp hơn nhưng giúp giảm phân hóa vùng và lao động."
priority = "Ưu tiên vùng yếu, SME, giáo dục số"
            rec = "Phù hợp mục tiêu bao trùm, giảm phân hóa vùng và hỗ trợ lao động; tốc độ tăng trưởng có thể thấp hơn."
else:
            rec = "Khuyến nghị chọn làm kịch bản chính vì cân bằng giữa tăng trưởng, số hóa, AI, lao động và rủi ro."
priority = "Triển khai làm baseline chính sách"
            rec = "Khuyến nghị chọn làm kịch bản nền vì cân bằng giữa tăng trưởng, số hóa, AI, lao động và rủi ro."

rows.append(
{
"Kịch bản": scenario,
"Xếp hạng tổng hợp": row["Xếp hạng tổng hợp"],
                "AIDEOM score": row["AIDEOM score"],
"Mức cảnh báo": row["Mức cảnh báo"],
"Ưu tiên điều chỉnh": priority,
"Khuyến nghị": rec,
@@ -533,14 +711,125 @@ def recommendation_table(kpi_df, risk_df):
return pd.DataFrame(rows).sort_values("Xếp hạng tổng hợp").reset_index(drop=True)


def ai_policy_analysis(kpi_df, risk_df, rec_df, s5, score_weights):
    best = kpi_df.iloc[0]
    worst_risk = risk_df.sort_values("Risk score", ascending=False).iloc[0]
    best_netjob = kpi_df.sort_values("NetJob", ascending=False).iloc[0]
    best_ai = kpi_df.sort_values("AI Readiness", ascending=False).iloc[0]

    analysis = f"""
### Tóm tắt điều hành

Kết quả AIDEOM-VN cho thấy kịch bản có điểm tổng hợp cao nhất là **{best['Kịch bản']}** với AIDEOM score **{best['AIDEOM score']:.3f}**. 
Kịch bản này đạt sự cân bằng tương đối giữa tăng trưởng, số hóa, năng lực AI, việc làm ròng và kiểm soát rủi ro.

### Diễn giải mô hình

S5 được mô hình tự tối ưu với cấu trúc phân bổ: **{s5['Đặc điểm phân bổ']}**. 
Điều này cho thấy dashboard không chỉ so sánh các kịch bản cố định, mà còn có khả năng đề xuất một phương án cân bằng dựa trên trọng số chính sách do người dùng chọn.

### Điểm nổi bật

- Kịch bản có **AI Readiness cao nhất** là **{best_ai['Kịch bản']}**.
- Kịch bản có **NetJob cao nhất** là **{best_netjob['Kịch bản']}**.
- Kịch bản có **rủi ro tổng hợp cao nhất** là **{worst_risk['Kịch bản']}** với Risk score **{worst_risk['Risk score']:.2f}**.

### Hàm ý chính sách

Nếu mục tiêu là tăng trưởng nhanh và xây dựng năng lực AI, chính sách có thể nghiêng về S2 hoặc S3. 
Nếu mục tiêu là ổn định xã hội, giảm chênh lệch vùng và hạn chế cú sốc lao động, S4 có vai trò quan trọng.
Nếu mục tiêu là phương án điều hành quốc gia giai đoạn 2026-2030, **S5 nên được xem là baseline chính sách**.

### Cảnh báo

AIDEOM-VN không thay thế quyết định chính trị - xã hội. Mô hình chỉ cung cấp bằng chứng định lượng, cảnh báo sớm và khuyến nghị kỹ thuật.
Quyết định cuối cùng vẫn cần xét đến ngân sách, năng lực thể chế, địa - chính trị, an sinh xã hội và tính chính danh.
"""

    weights_df = pd.DataFrame(
        {
            "Nhóm KPI": list(score_weights.keys()),
            "Trọng số đang dùng": list(score_weights.values()),
        }
    )

    return analysis, weights_df


def validation_checks(scenarios, sim_df, kpi_df, risk_df, candidate_df):
    scenario_df = pd.DataFrame(scenarios)

    checks = []

    checks.append(
        {
            "Yêu cầu": "Có đủ 6 module M1-M6",
            "Trạng thái": "Đạt" if len(MODULES) == 6 else "Chưa đạt",
            "Ghi chú": f"Số module = {len(MODULES)}",
        }
    )

    checks.append(
        {
            "Yêu cầu": "Có đủ 5 kịch bản S1-S5",
            "Trạng thái": "Đạt" if len(scenarios) == 5 else "Chưa đạt",
            "Ghi chú": f"Số kịch bản = {len(scenarios)}",
        }
    )

    share_ok = np.allclose(scenario_df[["K", "D", "AI", "H"]].sum(axis=1), 1.0)

    checks.append(
        {
            "Yêu cầu": "Tổng tỷ trọng K/D/AI/H của mỗi kịch bản bằng 1",
            "Trạng thái": "Đạt" if share_ok else "Chưa đạt",
            "Ghi chú": "Kiểm tra tổng share từng dòng",
        }
    )

    checks.append(
        {
            "Yêu cầu": "S5 được tối ưu tự động",
            "Trạng thái": "Đạt" if len(candidate_df) > 1 else "Chưa đạt",
            "Ghi chú": f"Số phương án ứng viên đã quét = {len(candidate_df)}",
        }
    )

    checks.append(
        {
            "Yêu cầu": "Có dữ liệu đường kịch bản 2026-2030",
            "Trạng thái": "Đạt" if set(YEARS).issubset(set(sim_df["Năm"])) else "Chưa đạt",
            "Ghi chú": f"Số dòng mô phỏng = {len(sim_df)}",
        }
    )

    checks.append(
        {
            "Yêu cầu": "Có KPI năm 2030 và xếp hạng tổng hợp",
            "Trạng thái": "Đạt" if "AIDEOM score" in kpi_df.columns else "Chưa đạt",
            "Ghi chú": f"Số kịch bản trong KPI = {len(kpi_df)}",
        }
    )

    checks.append(
        {
            "Yêu cầu": "Có cảnh báo rủi ro xanh/vàng/đỏ",
            "Trạng thái": "Đạt" if "Mức cảnh báo" in risk_df.columns else "Chưa đạt",
            "Ghi chú": ", ".join(risk_df["Mức cảnh báo"].unique().tolist()),
        }
    )

    return pd.DataFrame(checks)


def make_styled_table(df, decimals=3):
show_df = df.copy()

format_dict = {}

for col in show_df.columns:
if pd.api.types.is_numeric_dtype(show_df[col]):
            if str(col).lower() in ["năm", "xếp hạng tổng hợp", "xếp hạng gdp", "xếp hạng ai", "xếp hạng risk"]:
            if str(col).lower() in ["năm", "xếp hạng tổng hợp"]:
format_dict[col] = "{:.0f}"
else:
format_dict[col] = "{:." + str(decimals) + "f}"
@@ -633,17 +922,62 @@ def style_base_fig(fig, height=430):


def render():
    st.title("Bài 12. AIDEOM-VN tích hợp")
    st.caption("Dashboard ra quyết định chính sách: mô-đun, kịch bản, KPI 2030, cảnh báo rủi ro và khuyến nghị")
    st.title("Bài 12. AIDEOM-VN tích hợp nâng cao")
    st.caption("Dashboard ra quyết định: M1-M6, kịch bản, S5 tự tối ưu, KPI 2030, cảnh báo rủi ro và khuyến nghị chính sách")

    macro, macro_source = load_macro_data()
    regions, region_source = load_region_data()
    sectors, sector_source = load_sector_data()

    with st.expander("Cấu hình mô phỏng và trọng số AIDEOM score", expanded=False):
        st.markdown("Điều chỉnh trọng số để xem S5 tối ưu thay đổi như thế nào.")

        c1, c2, c3 = st.columns(3)
        w_gdp = c1.slider("Trọng số tăng trưởng/GDP", 0.05, 0.50, 0.25, 0.05)
        w_digital = c2.slider("Trọng số số hóa", 0.05, 0.40, 0.20, 0.05)
        w_ai = c3.slider("Trọng số AI readiness", 0.05, 0.40, 0.20, 0.05)

        c4, c5, c6 = st.columns(3)
        w_netjob = c4.slider("Trọng số việc làm ròng", 0.05, 0.40, 0.15, 0.05)
        w_risk = c5.slider("Trọng số giảm rủi ro", 0.05, 0.40, 0.15, 0.05)
        w_training = c6.slider("Trọng số nhân lực số", 0.05, 0.35, 0.05, 0.05)

    macro = load_macro_data()
    regions = load_region_data()
    sectors = load_sector_data()
        annual_budget = st.slider("Ngân sách mô phỏng hằng năm", 500, 3000, 1000, 100)

    sim_df = simulate_scenarios(macro, regions, sectors, annual_budget=1000)
    kpi_df = kpi_2030_table(sim_df)
    score_weights = normalize_weights(
        {
            "GDP": w_gdp,
            "Digital": w_digital,
            "AI": w_ai,
            "NetJob": w_netjob,
            "Risk": w_risk,
            "Training": w_training,
        }
    )

    base = base_values_from_data(macro, regions, sectors)
    s5, candidate_df = optimize_s5_allocation(
        base=base,
        score_weights=score_weights,
        annual_budget=annual_budget,
        step=0.05,
    )

    scenarios = build_scenarios(s5)
    scenarios_df = pd.DataFrame(scenarios)

    sim_df = simulate_scenarios(
        scenarios=scenarios,
        macro=macro,
        regions=regions,
        sectors=sectors,
        annual_budget=annual_budget,
    )

    kpi_df = kpi_2030_table(sim_df, score_weights)
risk_df = risk_alert_table(sim_df)
rec_df = recommendation_table(kpi_df, risk_df)
    validation_df = validation_checks(scenarios, sim_df, kpi_df, risk_df, candidate_df)

best_scenario = kpi_df.iloc[0]["Kịch bản"]
best_score = kpi_df.iloc[0]["AIDEOM score"]
@@ -653,40 +987,42 @@ def render():
"Tổng quan AIDEOM-VN",
"Đường kịch bản",
"KPI năm 2030",
            "S5 tối ưu tự động",
"Cảnh báo rủi ro",
            "Thảo luận chính sách",
            "Tác nhân AI phân tích",
"Khuyến nghị chính sách",
            "Kiểm định & tải dữ liệu",
]
)

    # =====================================================
    # 12.1 + 12.2
    # =====================================================
with tabs[0]:
st.header("12.1. Yêu cầu chức năng")

st.markdown(
"""
           Mô hình **AIDEOM-VN** gồm 6 module liên kết theo cấu trúc Mục 14 của bài báo nguồn.
            Mục tiêu là tích hợp các kết quả từ Bài 1 đến Bài 11 thành một dashboard ra quyết định chính sách.
            Bài 12 đóng vai trò tích hợp kết quả từ các bài trước thành dashboard hỗ trợ ra quyết định.
           """
)

show_table(modules_df(), decimals=3)

st.header("12.2. Năm kịch bản chính sách")

        scenario_display = scenario_df()[
            ["Kịch bản", "Mô tả ngắn", "Đặc điểm phân bổ", "K", "D", "AI", "H"]
        ]
        show_table(scenario_display, decimals=3)
        show_table(
            scenarios_df[
                ["Kịch bản", "Mô tả ngắn", "Đặc điểm phân bổ", "K", "D", "AI", "H", "Loại"]
            ],
            decimals=3,
        )

        c1, c2, c3 = st.columns(3)
        c1, c2, c3, c4 = st.columns(4)
c1.metric("Số module", "6")
c2.metric("Số kịch bản", "5")
c3.metric("Kịch bản tốt nhất", best_scenario)
        c4.metric("AIDEOM score", f"{best_score:.3f}")

        alloc_long = scenario_df().melt(
        alloc_long = scenarios_df.melt(
id_vars=["Kịch bản"],
value_vars=["K", "D", "AI", "H"],
var_name="Hạng mục",
@@ -699,6 +1035,7 @@ def render():
y="Tỷ trọng",
color="Hạng mục",
title="Cấu trúc phân bổ K/D/AI/H theo 5 kịch bản",
            color_discrete_map=MULTI_COLORS,
)
fig_alloc.update_traces(marker_line_color="white", marker_line_width=1)
fig_alloc.update_layout(
@@ -709,18 +1046,13 @@ def render():
style_base_fig(fig_alloc, height=470)
st.plotly_chart(fig_alloc, use_container_width=True)

        st.success(
            "Dashboard tích hợp M1-M6 và mô phỏng 5 kịch bản chính sách đến năm 2030."
        )
        st.success("Bản nâng cấp đã có S5 tự tối ưu theo trọng số KPI, không còn là tỷ trọng cố định.")

    # =====================================================
    # Đường kịch bản
    # =====================================================
with tabs[1]:
st.header("Đường kịch bản 2026-2030")

metric_choice = st.selectbox(
            "Chọn chỉ tiêu để vẽ đường kịch bản",
            "Chọn chỉ tiêu",
[
"GDP",
"GDP growth (%)",
@@ -729,16 +1061,12 @@ def render():
"Lao động đào tạo (%)",
"Kinh tế số/GDP (%)",
"NetJob",
                "DisplacedJob",
"Risk score",
],
index=0,
)

        c1, c2, c3 = st.columns(3)
        c1.metric("Kịch bản tốt nhất", best_scenario)
        c2.metric("AIDEOM score", f"{best_score:.3f}")
        c3.metric("Năm đích", "2030")

fig_line = px.line(
sim_df,
x="Năm",
@@ -749,26 +1077,27 @@ def render():
color_discrete_map=MULTI_COLORS,
)
fig_line.update_traces(line=dict(width=4), marker=dict(size=8))
        fig_line.update_layout(
            xaxis_title="Năm",
            yaxis_title=metric_choice,
        )
        fig_line.update_layout(xaxis_title="Năm", yaxis_title=metric_choice)
style_base_fig(fig_line, height=500)
st.plotly_chart(fig_line, use_container_width=True)

        st.subheader("Bảng dữ liệu mô phỏng")
        st.subheader("Dữ liệu mô phỏng 2026-2030")
show_table(sim_df, decimals=3)

        st.success(
            "Đường kịch bản giúp so sánh động thái của tăng trưởng, số hóa, AI readiness, việc làm và rủi ro theo thời gian."
        )

    # =====================================================
    # KPI 2030
    # =====================================================
with tabs[2]:
st.header("KPI năm 2030")

        weight_df = pd.DataFrame(
            {
                "Nhóm KPI": list(score_weights.keys()),
                "Trọng số chuẩn hóa": list(score_weights.values()),
            }
        )

        st.subheader("Trọng số AIDEOM score")
        show_table(weight_df, decimals=3)

        st.subheader("Bảng KPI 2030")
show_table(kpi_df, decimals=3)

kpi_long = kpi_df[
@@ -794,44 +1123,93 @@ def render():
y="Giá trị",
color="KPI",
barmode="group",
            title="So sánh KPI năm 2030 theo kịch bản",
            title="So sánh KPI năm 2030",
)
fig_kpi.update_traces(marker_line_color="white", marker_line_width=1)
        fig_kpi.update_layout(
            xaxis_title="Kịch bản",
            yaxis_title="Giá trị KPI",
        )
        fig_kpi.update_layout(xaxis_title="Kịch bản", yaxis_title="Giá trị KPI")
style_base_fig(fig_kpi, height=540)
st.plotly_chart(fig_kpi, use_container_width=True)

        score_plot = kpi_df.sort_values("AIDEOM score", ascending=True)

fig_score = px.bar(
            kpi_df.sort_values("AIDEOM score", ascending=True),
            score_plot,
x="AIDEOM score",
y="Kịch bản",
orientation="h",
            text=kpi_df.sort_values("AIDEOM score", ascending=True)["AIDEOM score"].round(3),
            title="Xếp hạng tổng hợp AIDEOM score năm 2030",
            text=score_plot["AIDEOM score"].round(3),
            title="Xếp hạng AIDEOM score năm 2030",
)
fig_score.update_traces(
marker_color=BRAND,
textposition="outside",
textfont=dict(color=BRAND),
)
        fig_score.update_layout(
            xaxis_title="AIDEOM score",
            yaxis_title="Kịch bản",
        )
        fig_score.update_layout(xaxis_title="AIDEOM score", yaxis_title="Kịch bản")
style_base_fig(fig_score, height=430)
st.plotly_chart(fig_score, use_container_width=True)

        st.success(
            f"Kịch bản có điểm tổng hợp cao nhất năm 2030 là {best_scenario}."
        st.download_button(
            "Tải KPI 2030 CSV",
            data=kpi_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="bai12_kpi_2030.csv",
            mime="text/csv",
            key="download_kpi_2030",
)

    # =====================================================
    # Cảnh báo rủi ro
    # =====================================================
with tabs[3]:
        st.header("S5 tối ưu tự động")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("K", f"{s5['K']:.0%}")
        c2.metric("D", f"{s5['D']:.0%}")
        c3.metric("AI", f"{s5['AI']:.0%}")
        c4.metric("H", f"{s5['H']:.0%}")

        st.success(f"S5 được chọn tự động: **{s5['Đặc điểm phân bổ']}**.")

        st.subheader("Top 15 phương án ứng viên")
        show_table(candidate_df.head(15), decimals=4)

        best_candidates = candidate_df.head(20).copy()
        best_candidates["Phương án"] = ["C" + str(i + 1) for i in range(len(best_candidates))]

        fig_candidates = px.scatter(
            best_candidates,
            x="AI",
            y="H",
            size="Optimized score",
            color="K",
            hover_data=["D", "Raw score", "Balance penalty", "Risk score 2030", "NetJob 2030"],
            title="Không gian ứng viên S5: AI - H - K - điểm tối ưu",
            color_continuous_scale="Viridis",
        )
        fig_candidates.update_layout(xaxis_title="Tỷ trọng AI", yaxis_title="Tỷ trọng H")
        style_base_fig(fig_candidates, height=500)
        st.plotly_chart(fig_candidates, use_container_width=True)

        alloc_s5_df = pd.DataFrame(
            {
                "Hạng mục": ["K", "D", "AI", "H"],
                "Tỷ trọng": [s5["K"], s5["D"], s5["AI"], s5["H"]],
            }
        )

        fig_s5 = px.pie(
            alloc_s5_df,
            names="Hạng mục",
            values="Tỷ trọng",
            title="Cấu trúc phân bổ S5 tối ưu cân bằng",
        )
        fig_s5.update_layout(
            height=430,
            paper_bgcolor="white",
            font=dict(color=BRAND, size=15),
            title_font=dict(color=BRAND, size=20),
        )
        st.plotly_chart(fig_s5, use_container_width=True)

    with tabs[4]:
st.header("Cảnh báo rủi ro")

show_table(risk_df, decimals=3)
@@ -843,6 +1221,7 @@ def render():
"Cyber risk",
"Environmental risk",
"Dependency risk",
                "Labor transition risk",
]
].melt(
id_vars="Kịch bản",
@@ -856,124 +1235,72 @@ def render():
y="Điểm rủi ro",
color="Loại rủi ro",
barmode="group",
            title="Bản đồ rủi ro theo kịch bản năm 2030",
            title="Bản đồ rủi ro năm 2030",
)
fig_risk.update_traces(marker_line_color="white", marker_line_width=1)
        fig_risk.update_layout(
            xaxis_title="Kịch bản",
            yaxis_title="Điểm rủi ro 0-100",
        )
        style_base_fig(fig_risk, height=500)
        fig_risk.update_layout(xaxis_title="Kịch bản", yaxis_title="Điểm rủi ro 0-100")
        style_base_fig(fig_risk, height=520)
st.plotly_chart(fig_risk, use_container_width=True)

        high_alerts = risk_df[risk_df["Mức cảnh báo"] == "Cao"]
        red_alert = risk_df[risk_df["Mức cảnh báo"] == "Đỏ"]

        if len(high_alerts) > 0:
            st.warning(
                "Có kịch bản cảnh báo cao: "
                + ", ".join(high_alerts["Kịch bản"].tolist())
            )
        if len(red_alert) > 0:
            st.warning("Có kịch bản mức Đỏ: " + ", ".join(red_alert["Kịch bản"].tolist()))
else:
            st.success("Không có kịch bản nào ở mức cảnh báo cao theo ngưỡng hiện tại.")

        st.markdown(
            """
            Cảnh báo rủi ro trong AIDEOM-VN không chỉ phản ánh mức AI cao,
            mà còn xét sự mất cân đối giữa AI, nhân lực số, dữ liệu, an ninh mạng và môi trường.
            """
        )

    # =====================================================
    # Thảo luận chính sách
    # =====================================================
    with tabs[4]:
        st.header("Thảo luận chính sách")

        st.markdown("### 1. AIDEOM-VN tích hợp các bài trước như thế nào?")

        st.markdown(
            """
            AIDEOM-VN hoạt động như một khung tích hợp. Bài 1 cung cấp nền dự báo tăng trưởng;
            Bài 4, 7, 8 cung cấp tối ưu phân bổ tĩnh - động - đa mục tiêu;
            Bài 6 đánh giá sẵn sàng vùng; Bài 9 mô phỏng lao động;
            Bài 10 đưa bất định vào quyết định; Bài 11 bổ sung tư duy học chính sách thích ứng.
            """
        )

        st.markdown("### 2. Tại sao cần so sánh nhiều kịch bản?")

        st.markdown(
            """
            Một kịch bản đơn lẻ dễ tạo cảm giác chắc chắn giả tạo.
            Trong thực tế, chuyển đổi số Việt Nam chịu ảnh hưởng của tăng trưởng, FDI, kỹ năng lao động,
            rủi ro tự động hóa, an ninh dữ liệu, thiên tai, biến động chuỗi cung ứng và thay đổi công nghệ.
            Vì vậy, dashboard cần so sánh nhiều đường kịch bản để thấy đánh đổi giữa tăng trưởng, bao trùm và rủi ro.
            """
            st.success("Không có kịch bản nào ở mức Đỏ theo ngưỡng hiện tại.")

        st.download_button(
            "Tải cảnh báo rủi ro CSV",
            data=risk_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="bai12_risk_alerts.csv",
            mime="text/csv",
            key="download_risk_alerts",
)

        st.markdown("### 3. Kịch bản nào nên là trục chính sách?")
    with tabs[5]:
        st.header("Tác nhân AI phân tích kết quả")

        st.success(
            f"Theo AIDEOM score năm 2030, kịch bản nên ưu tiên là **{best_scenario}**."
        )
        analysis_text, weights_df = ai_policy_analysis(kpi_df, risk_df, rec_df, s5, score_weights)

        st.markdown(
            """
            Tuy nhiên, kết quả mô hình không nên được hiểu là mệnh lệnh tự động.
            Đây là cơ sở kỹ thuật để hỗ trợ ra quyết định. Nhà hoạch định chính sách vẫn cần cân nhắc:
            năng lực ngân sách, địa - chính trị, công bằng vùng, an sinh lao động,
            an ninh dữ liệu và tính khả thi thể chế.
            """
        )
        st.markdown(analysis_text)

        st.markdown("### 4. AI có nên được ưu tiên tuyệt đối không?")
        st.subheader("Trọng số tác nhân đang sử dụng")
        show_table(weights_df, decimals=3)

        st.markdown(
            """
            AI là động lực tăng năng suất, nhưng nếu AI tăng nhanh hơn dữ liệu, nhân lực và quản trị rủi ro,
            nền kinh tế có thể đối mặt với phụ thuộc công nghệ, rủi ro an ninh mạng và dịch chuyển lao động.
            Vì vậy, AI nên đi cùng ba điều kiện: dữ liệu tốt, nhân lực đủ và khung quản trị an toàn.
            """
        st.info(
            "Tác nhân này là phân tích offline/rule-based để tránh lỗi API key khi deploy. "
            "Nội dung vẫn dựa trên kết quả định lượng của dashboard."
)

    # =====================================================
    # Khuyến nghị chính sách
    # =====================================================
    with tabs[5]:
    with tabs[6]:
st.header("Khuyến nghị chính sách")

show_table(rec_df, decimals=3)

st.markdown("### Khuyến nghị trung tâm")

        st.success(
            f"Chọn **{best_scenario}** làm kịch bản chính sách nền cho giai đoạn 2026-2030."
        )
        st.success(f"Chọn **{best_scenario}** làm kịch bản nền cho giai đoạn 2026-2030.")

st.markdown(
"""
            **Một là, triển khai kịch bản tối ưu cân bằng làm baseline.**
            Không nên cực đoan theo hướng chỉ đầu tư hạ tầng truyền thống hoặc chỉ chạy theo AI.
            Chính sách cần cân bằng giữa K, D, AI và H.
            **Một là, dùng S5 tối ưu cân bằng làm baseline.**  
            Chính sách không nên cực đoan theo hướng chỉ đầu tư hạ tầng truyền thống hoặc chỉ chạy theo AI.

            **Hai là, đặt nhân lực số là điều kiện bắt buộc của đầu tư AI.**
            Mọi dự án AI quy mô lớn nên có cấu phần đào tạo lại, nâng kỹ năng và chuyển đổi việc làm.
            **Hai là, AI phải đi cùng nhân lực số.**  
            Mọi dự án AI quy mô lớn cần có cấu phần đào tạo lại, nâng kỹ năng, chuyển đổi việc làm và an sinh lao động.

            **Ba là, xây dựng dashboard cảnh báo sớm rủi ro.**
            Cần theo dõi đồng thời cyber risk, environmental risk, dependency risk và NetJob.
            Kịch bản có GDP cao nhưng Risk score quá cao không nên được chọn nguyên trạng.
            **Ba là, dữ liệu và an ninh mạng là điều kiện nền.**  
            Nếu số hóa nhanh nhưng thiếu quản trị dữ liệu, rủi ro cyber và dependency có thể tăng mạnh.

            **Bốn là, ưu tiên vùng yếu và SME trong chính sách bao trùm.**
            Nếu chỉ ưu tiên vùng có năng lực hấp thụ cao, chuyển đổi số có thể làm tăng chênh lệch vùng miền.
            **Bốn là, ưu tiên vùng yếu và SME.**  
            Nếu chỉ ưu tiên vùng có năng lực hấp thụ cao, chuyển đổi số có thể làm tăng khoảng cách vùng miền.

            **Năm là, dùng mô hình như công cụ hỗ trợ, không thay thế quyết định chính trị - xã hội.**
            AIDEOM-VN nên được đặt trong quy trình có chuyên gia, địa phương, doanh nghiệp và đại diện người lao động tham gia.
            **Năm là, giữ vai trò con người trong quyết định cuối cùng.**  
            AIDEOM-VN là công cụ hỗ trợ ra quyết định, không thay thế quy trình chính trị - xã hội.
           """
)

        st.markdown("### Bộ KPI theo dõi hằng năm")

        kpi_tracking = pd.DataFrame(
        tracking = pd.DataFrame(
{
"Nhóm KPI": [
"Tăng trưởng",
@@ -1002,10 +1329,65 @@ def render():
}
)

        show_table(kpi_tracking, decimals=3)
        st.subheader("Bộ KPI theo dõi hằng năm")
        show_table(tracking, decimals=3)

    with tabs[7]:
        st.header("Kiểm định yêu cầu và tải dữ liệu")

        st.subheader("Nguồn dữ liệu")
        source_df = pd.DataFrame(
            {
                "Bộ dữ liệu": [
                    "vietnam_macro_2020_2025.csv",
                    "vietnam_regions_2024.csv",
                    "vietnam_sectors_2024.csv",
                ],
                "Nguồn đang dùng": [
                    macro_source,
                    region_source,
                    sector_source,
                ],
                "Số dòng": [
                    len(macro),
                    len(regions),
                    len(sectors),
                ],
            }
        )
        show_table(source_df, decimals=3)

        st.subheader("Checklist yêu cầu bài 12")
        show_table(validation_df, decimals=3)

        c1, c2, c3 = st.columns(3)

        c1.download_button(
            "Tải toàn bộ mô phỏng CSV",
            data=sim_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="bai12_simulation_2026_2030.csv",
            mime="text/csv",
            key="download_simulation",
        )

        c2.download_button(
            "Tải bảng kịch bản CSV",
            data=scenarios_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="bai12_scenarios.csv",
            mime="text/csv",
            key="download_scenarios",
        )

        c3.download_button(
            "Tải ứng viên S5 CSV",
            data=candidate_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="bai12_s5_candidates.csv",
            mime="text/csv",
            key="download_s5_candidates",
        )

st.info(
            "Kết luận: Bài 12 đóng vai trò dashboard tích hợp, chuyển kết quả định lượng từ các bài trước thành công cụ hỗ trợ ra quyết định chính sách."
            "Bài 12 bản nâng cấp đã có đủ: M1-M6, S1-S5, S5 tự tối ưu, KPI 2030, cảnh báo rủi ro, phân tích chính sách và tải dữ liệu."
)


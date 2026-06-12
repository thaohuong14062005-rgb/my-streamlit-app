"D": "#E76F51",
"AI": "#2A9D8F",
"H": "#F4A261",
    "GDP": "#053151",
    "Digital Index": "#E76F51",
    "AI Readiness": "#2A9D8F",
    "NetJob": "#F4A261",
    "Risk score": "#BC4749",
}


YEARS = np.arange(2026, 2031)


MODULES = [
{
"Module": "M1",
"Tên": "Dự báo kinh tế",
"Đầu vào": "Macro 2020-2025",
"Đầu ra": "GDP, TFP, lao động 2026-2030",
        "Kỹ thuật chính": "Cobb-Douglas + Bài 1, có thể bổ sung VAR/LSTM",
        "Kỹ thuật chính": "Cobb-Douglas + Bài 1",
},
{
"Module": "M2",
"Tên": "Đánh giá sẵn sàng số",
"Đầu vào": "Sectors, Regions",
        "Đầu ra": "Bản đồ Digital Index + AI Readiness",
        "Kỹ thuật chính": "Bài 6 TOPSIS + entropy weight",
        "Đầu ra": "Digital Index + AI Readiness",
        "Kỹ thuật chính": "Bài 6 TOPSIS",
},
{
"Module": "M3",
"Tên": "Tối ưu phân bổ",
"Đầu vào": "Budget, beta-matrix",
"Đầu ra": "Phân bổ ngành - vùng - thời gian",
        "Kỹ thuật chính": "Bài 4 LP + Bài 8 dynamic",
        "Kỹ thuật chính": "Bài 4 + Bài 8",
},
{
"Module": "M4",
"Tên": "Mô phỏng lao động",
"Đầu vào": "AI, H plans",
"Đầu ra": "NetJob từng ngành",
        "Kỹ thuật chính": "Bài 9 + bổ sung Markov chain",
        "Kỹ thuật chính": "Bài 9",
},
{
"Module": "M5",
"Tên": "Đánh giá rủi ro",
"Đầu vào": "Risk parameters",
        "Đầu ra": "Cyber, environmental, dependency",
        "Kỹ thuật chính": "Bài 7 đa mục tiêu + Bài 10 SP",
        "Đầu ra": "Cyber, môi trường, phụ thuộc",
        "Kỹ thuật chính": "Bài 7 + Bài 10",
},
{
"Module": "M6",
"Tên": "Dashboard ra quyết định",
"Đầu vào": "Outputs M1-M5",
        "Đầu ra": "Trực quan kịch bản, cảnh báo, khuyến nghị",
        "Kỹ thuật chính": "Streamlit / Dash / Plotly",
        "Đầu ra": "Kịch bản, cảnh báo, khuyến nghị",
        "Kỹ thuật chính": "Streamlit / Plotly",
},
]

@@ -84,17 +80,15 @@
"D": 0.10,
"AI": 0.10,
"H": 0.10,
        "Loại": "Cố định",
},
{
"Kịch bản": "S2. Số hóa nhanh",
        "Mô tả ngắn": "Tăng đầu tư chính phủ số, doanh nghiệp số, thanh toán số",
        "Mô tả ngắn": "Tăng chính phủ số, doanh nghiệp số, thanh toán số",
"Đặc điểm phân bổ": "25% K + 45% D + 15% AI + 15% H",
"K": 0.25,
"D": 0.45,
"AI": 0.15,
"H": 0.15,
        "Loại": "Cố định",
},
{
"Kịch bản": "S3. AI dẫn dắt",
@@ -104,7 +98,6 @@
"D": 0.20,
"AI": 0.45,
"H": 0.15,
        "Loại": "Cố định",
},
{
"Kịch bản": "S4. Bao trùm số",
@@ -114,7 +107,6 @@
"D": 0.20,
"AI": 0.10,
"H": 0.40,
        "Loại": "Cố định",
},
]

@@ -143,8 +135,6 @@ def fallback_macro_data():
"GDP_trillion_VND": [6345, 6480, 7200, 7550, 8075, 8550],
"GDP_growth_pct": [2.9, 2.6, 8.0, 5.1, 7.1, 6.5],
"digital_economy_share_GDP_pct": [12.0, 13.5, 14.8, 16.0, 17.2, 18.5],
            "labor_productivity_million_VND": [150, 158, 170, 178, 188, 198],
            "inflation_CPI_pct": [3.2, 1.8, 3.2, 3.3, 3.8, 3.5],
}
)

@@ -163,7 +153,6 @@ def fallback_region_data():
"digital_index_0_100": [48, 78, 55, 42, 82, 52],
"ai_readiness_0_100": [41, 77, 49, 36, 84, 45],
"trained_labor_pct": [24, 42, 29, 22, 45, 25],
            "gini_coef": [0.39, 0.36, 0.37, 0.40, 0.38, 0.35],
}
)

@@ -181,9 +170,8 @@ def fallback_sector_data():
"Thông tin - Truyền thông",
"Giáo dục - Y tế - Xã hội",
],
            "growth_rate_2024_pct": [3.3, 9.6, 7.2, 8.1, 7.8, 6.9, 10.5, 5.8],
            "ai_readiness_0_100": [38, 62, 44, 55, 82, 58, 88, 52],
"automation_risk_pct": [18, 42, 25, 38, 52, 35, 28, 22],
            "ai_readiness_0_100": [38, 62, 44, 55, 82, 58, 88, 52],
}
)

@@ -192,277 +180,202 @@ def load_macro_data():
path = find_file("vietnam_macro_2020_2025.csv")

if path is None:
        return fallback_macro_data(), "Fallback"
        return fallback_macro_data()

try:
df = pd.read_csv(path)

        required = ["year", "GDP_trillion_VND", "GDP_growth_pct"]

        if any(col not in df.columns for col in required):
            return fallback_macro_data(), "Fallback"
        if "year" not in df.columns or "GDP_trillion_VND" not in df.columns:
            return fallback_macro_data()

for col in df.columns:
if col != "year":
df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=required).reset_index(drop=True)

        if len(df) == 0:
            return fallback_macro_data(), "Fallback"

        return df, path
        return df.dropna(subset=["year", "GDP_trillion_VND"]).reset_index(drop=True)

except Exception:
        return fallback_macro_data(), "Fallback"
        return fallback_macro_data()


def load_region_data():
path = find_file("vietnam_regions_2024.csv")

if path is None:
        return fallback_region_data(), "Fallback"
        return fallback_region_data()

try:
df = pd.read_csv(path)

        required = ["region_name_vi", "digital_index_0_100", "ai_readiness_0_100"]

        if any(col not in df.columns for col in required):
            return fallback_region_data(), "Fallback"

        for col in df.columns:
            if col != "region_name_vi":
                df[col] = pd.to_numeric(df[col], errors="ignore")
        need = ["digital_index_0_100", "ai_readiness_0_100"]

        df = df.dropna(subset=required).reset_index(drop=True)
        if any(col not in df.columns for col in need):
            return fallback_region_data()

        if len(df) == 0:
            return fallback_region_data(), "Fallback"
        for col in need + ["trained_labor_pct"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df, path
        return df.dropna(subset=need).reset_index(drop=True)

except Exception:
        return fallback_region_data(), "Fallback"
        return fallback_region_data()


def load_sector_data():
path = find_file("vietnam_sectors_2024.csv")

if path is None:
        return fallback_sector_data(), "Fallback"
        return fallback_sector_data()

try:
df = pd.read_csv(path)

        required = ["sector_name_vi", "ai_readiness_0_100"]

        if any(col not in df.columns for col in required):
            return fallback_sector_data(), "Fallback"

        for col in df.columns:
            if col != "sector_name_vi":
                df[col] = pd.to_numeric(df[col], errors="ignore")

        df = df.dropna(subset=required).reset_index(drop=True)
        if "automation_risk_pct" not in df.columns:
            return fallback_sector_data()

        if len(df) == 0:
            return fallback_sector_data(), "Fallback"
        df["automation_risk_pct"] = pd.to_numeric(df["automation_risk_pct"], errors="coerce")

        return df, path
        return df.dropna(subset=["automation_risk_pct"]).reset_index(drop=True)

except Exception:
        return fallback_sector_data(), "Fallback"


def modules_df():
    return pd.DataFrame(MODULES)

        return fallback_sector_data()

def normalize_weights(weights):
    total = sum(weights.values())

    if total <= 0:
        n = len(weights)
        return {k: 1.0 / n for k in weights}

    return {k: v / total for k, v in weights.items()}


def base_values_from_data(macro, regions, sectors):
def get_base_values(macro, regions, sectors):
macro = macro.sort_values("year").reset_index(drop=True)
last = macro.iloc[-1]

gdp0 = float(last.get("GDP_trillion_VND", 8550))
    growth0 = float(last.get("GDP_growth_pct", 6.5))

if "digital_economy_share_GDP_pct" in macro.columns:
digital_share0 = float(last.get("digital_economy_share_GDP_pct", 18.5))
else:
digital_share0 = 18.5

    if "labor_productivity_million_VND" in macro.columns:
        productivity0 = float(last.get("labor_productivity_million_VND", 198))
    else:
        productivity0 = 198.0

    digital_index0 = float(pd.to_numeric(regions["digital_index_0_100"], errors="coerce").mean())
    ai_ready0 = float(pd.to_numeric(regions["ai_readiness_0_100"], errors="coerce").mean())
    digital_index0 = float(regions["digital_index_0_100"].mean())
    ai_ready0 = float(regions["ai_readiness_0_100"].mean())

if "trained_labor_pct" in regions.columns:
        trained0 = float(pd.to_numeric(regions["trained_labor_pct"], errors="coerce").mean())
        trained0 = float(regions["trained_labor_pct"].mean())
else:
trained0 = 32.0

    if "automation_risk_pct" in sectors.columns:
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
    risk0 = float(sectors["automation_risk_pct"].mean())

return {
        "gdp0": gdp0,
        "growth0": growth0,
        "digital_share0": digital_share0,
        "productivity0": productivity0,
        "digital_index0": digital_index0,
        "ai_ready0": ai_ready0,
        "trained0": trained0,
        "automation_risk0": automation_risk0,
        "GDP": gdp0,
        "Digital Index": digital_index0,
        "AI Readiness": ai_ready0,
        "Training": trained0,
        "Digital share": digital_share0,
        "Risk": risk0,
}


def simulate_one_scenario(scenario, base, annual_budget=1000):
def simulate_one_scenario(scenario, base):
rows = []

    gdp = base["gdp0"]
    digital_index = base["digital_index0"]
    ai_ready = base["ai_ready0"]
    trained = base["trained0"]
    digital_share = base["digital_share0"]
    automation_risk = base["automation_risk0"]
    gdp = base["GDP"]
    digital_index = base["Digital Index"]
    ai_ready = base["AI Readiness"]
    training = base["Training"]
    digital_share = base["Digital share"]
    risk_base = base["Risk"]

for year in YEARS:
        k = float(scenario["K"])
        d = float(scenario["D"])
        ai = float(scenario["AI"])
        h = float(scenario["H"])
        k = scenario["K"]
        d = scenario["D"]
        ai = scenario["AI"]
        h = scenario["H"]

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
            4.5
            + 1.0 * k
            + 1.4 * d
            + 1.7 * ai
            + 0.9 * h
            + 0.015 * (digital_index - 55)
            + 0.010 * (ai_ready - 55)
)

gdp = gdp * (1 + growth / 100)
        digital_index = min(100, digital_index + 3.7 * d + 1.4 * k + 1.1 * h)
        ai_ready = min(100, ai_ready + 4.5 * ai + 1.2 * d + 1.5 * h)
        trained = min(100, trained + 3.7 * h + 0.7 * d)
        digital_share = min(45, digital_share + 1.35 * d + 1.05 * ai + 0.45 * h)

        digital_index = min(100, digital_index + 3.6 * d + 1.3 * k + 0.9 * h)
        ai_ready = min(100, ai_ready + 4.3 * ai + 1.1 * d + 1.2 * h)
        training = min(100, training + 3.4 * h + 0.7 * d)
        digital_share = min(45, digital_share + 1.2 * d + 1.0 * ai + 0.4 * h)

        displaced = risk_base * (0.50 * ai + 0.22 * d + 0.10 * k) * 16
        new_job = 520 * ai + 270 * d + 160 * h
        net_job = new_job - displaced + training * h * 12

        cyber_risk = np.clip(35 + 18 * ai + 8 * d - 7 * h, 0, 100)
        env_risk = np.clip(35 + 14 * k + 17 * ai - 7 * d - 6 * h, 0, 100)
        labor_risk = np.clip(35 + displaced / 25 - 0.25 * training * h, 0, 100)

        risk_score = np.clip(0.35 * cyber_risk + 0.30 * env_risk + 0.35 * labor_risk, 0, 100)

rows.append(
{
                "Năm": int(year),
                "Năm": year,
"Kịch bản": scenario["Kịch bản"],
"GDP": gdp,
"GDP growth (%)": growth,
"Digital Index": digital_index,
"AI Readiness": ai_ready,
                "Lao động đào tạo (%)": trained,
                "Lao động đào tạo (%)": training,
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
                "Cyber risk": cyber_risk,
                "Environmental risk": env_risk,
                "Labor risk": labor_risk,
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
def normalize(x):
    arr = np.asarray(x, dtype=float)
    min_v = np.nanmin(arr)
    max_v = np.nanmax(arr)

    if np.isclose(max_v, min_v):
        return np.ones_like(arr) * 0.5

    return (arr - min_v) / (max_v - min_v)


def score_terminal(df2030):
score = (
        score_weights["GDP"] * gdp_growth_score
        + score_weights["Digital"] * digital_score
        + score_weights["AI"] * ai_score
        + score_weights["NetJob"] * netjob_score
        + score_weights["Risk"] * risk_score
        + score_weights["Training"] * training_score
        0.25 * normalize(df2030["GDP"])
        + 0.20 * normalize(df2030["Digital Index"])
        + 0.20 * normalize(df2030["AI Readiness"])
        + 0.20 * normalize(df2030["NetJob"])
        + 0.15 * normalize(-df2030["Risk score"])
)

    return float(score)
    return score


def optimize_s5_allocation(base, score_weights, annual_budget=1000, step=0.05):
def optimize_s5(base):
candidates = []

    grid = np.arange(0.10, 0.501, step)
    grid = np.arange(0.10, 0.501, 0.05)

for k in grid:
for d in grid:
for ai in grid:
                h = 1.0 - k - d - ai
                h = 1 - k - d - ai

if h < 0.10 or h > 0.50:
continue

if k < 0.15 or d < 0.15 or ai < 0.10:
continue

                if not np.isclose(k + d + ai + h, 1.0, atol=1e-8):
                    continue

scenario = {
"Kịch bản": "Candidate",
"K": k,
@@ -471,125 +384,60 @@ def optimize_s5_allocation(base, score_weights, annual_budget=1000, step=0.05):
"H": h,
}

                path = simulate_one_scenario(scenario, base, annual_budget=annual_budget)
                terminal = path[path["Năm"] == 2030].iloc[0]
                score = absolute_candidate_score(terminal, score_weights)
                sim = simulate_one_scenario(scenario, base)
                terminal = sim[sim["Năm"] == 2030].copy()

                balance_penalty = 0.035 * (
                    abs(k - 0.25) + abs(d - 0.25) + abs(ai - 0.25) + abs(h - 0.25)
                raw = (
                    0.25 * min(1, terminal["GDP"].iloc[0] / (base["GDP"] * 1.45))
                    + 0.20 * terminal["Digital Index"].iloc[0] / 100
                    + 0.20 * terminal["AI Readiness"].iloc[0] / 100
                    + 0.20 * np.clip((terminal["NetJob"].iloc[0] + 300) / 1400, 0, 1)
                    + 0.15 * (1 - terminal["Risk score"].iloc[0] / 100)
)

                final_score = score - balance_penalty
                balance_penalty = 0.04 * (
                    abs(k - 0.25) + abs(d - 0.25) + abs(ai - 0.25) + abs(h - 0.25)
                )

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
                        "Score": raw - balance_penalty,
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
    cand = pd.DataFrame(candidates).sort_values("Score", ascending=False).reset_index(drop=True)
    best = cand.iloc[0]

        candidate_df = pd.DataFrame(
            [
                {
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
            ]
        )

    candidate_df = candidate_df.sort_values("Optimized score", ascending=False).reset_index(drop=True)
    best = candidate_df.iloc[0].to_dict()

    s5 = {
    return {
"Kịch bản": "S5. Tối ưu cân bằng",
        "Mô tả ngắn": "Kết quả mô hình AIDEOM-VN tự tối ưu theo trọng số KPI",
        "Đặc điểm phân bổ": (
            f"{best['K']:.0%} K + {best['D']:.0%} D + "
            f"{best['AI']:.0%} AI + {best['H']:.0%} H"
        ),
        "Mô tả ngắn": "Kết quả mô hình AIDEOM-VN tự chạy ra",
        "Đặc điểm phân bổ": f"{best['K']:.0%} K + {best['D']:.0%} D + {best['AI']:.0%} AI + {best['H']:.0%} H",
"K": float(best["K"]),
"D": float(best["D"]),
"AI": float(best["AI"]),
"H": float(best["H"]),
        "Loại": "Tối ưu tự động",
}

    return s5, candidate_df


def build_scenarios(s5):
def build_scenarios(base):
    s5 = optimize_s5(base)
return BASE_SCENARIOS + [s5]


def simulate_scenarios(scenarios, macro, regions, sectors, annual_budget=1000):
    base = base_values_from_data(macro, regions, sectors)

    frames = []

    for scenario in scenarios:
        frames.append(simulate_one_scenario(scenario, base, annual_budget=annual_budget))

def simulate_all(scenarios, base):
    frames = [simulate_one_scenario(sc, base) for sc in scenarios]
return pd.concat(frames, ignore_index=True)


def normalize_series(x):
    arr = np.asarray(x, dtype=float)
    min_v = np.nanmin(arr)
    max_v = np.nanmax(arr)

    if np.isclose(max_v, min_v):
        return np.ones_like(arr) * 0.5

    return (arr - min_v) / (max_v - min_v)


def kpi_2030_table(sim_df, score_weights):
def kpi_2030(sim_df):
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

    df["Xếp hạng tổng hợp"] = df["AIDEOM score"].rank(ascending=False, method="min").astype(int)
    df["AIDEOM score"] = score_terminal(df)
    df["Xếp hạng"] = df["AIDEOM score"].rank(ascending=False, method="min").astype(int)

cols = [
"Kịch bản",
@@ -600,17 +448,16 @@ def kpi_2030_table(sim_df, score_weights):
"Lao động đào tạo (%)",
"Kinh tế số/GDP (%)",
"NetJob",
        "DisplacedJob",
"Risk score",
"AIDEOM score",
        "Xếp hạng tổng hợp",
        "Xếp hạng",
]

    return df[cols].sort_values("Xếp hạng tổng hợp").reset_index(drop=True)
    return df[cols].sort_values("Xếp hạng").reset_index(drop=True)


def risk_alert_table(sim_df):
    df = sim_df[sim_df["Năm"] == 2030].copy().reset_index(drop=True)
def risk_alerts(sim_df):
    df = sim_df[sim_df["Năm"] == 2030].copy()

rows = []

@@ -626,50 +473,39 @@ def risk_alert_table(sim_df):
if row["Environmental risk"] >= 70:
alerts.append("Áp lực môi trường cao")

        if row["Dependency risk"] >= 70:
            alerts.append("Phụ thuộc công nghệ cao")

        if row["Labor transition risk"] >= 70:
            alerts.append("Dịch chuyển lao động cao")
        if row["Labor risk"] >= 70:
            alerts.append("Rủi ro lao động cao")

if row["NetJob"] < 0:
alerts.append("NetJob âm")

        if row["AI Readiness"] >= 75 and row["Lao động đào tạo (%)"] < 40:
            alerts.append("AI nhanh hơn năng lực nhân lực")

if len(alerts) == 0:
level = "Xanh"
            action = "Theo dõi định kỳ"
            alert_text = "Không có cảnh báo lớn"
            text = "Không có cảnh báo lớn"
elif row["Risk score"] >= 70 or row["NetJob"] < 0:
level = "Đỏ"
            action = "Cần can thiệp chính sách"
            alert_text = "; ".join(alerts)
            text = "; ".join(alerts)
else:
level = "Vàng"
            action = "Cần theo dõi sát"
            alert_text = "; ".join(alerts)
            text = "; ".join(alerts)

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
                "Labor risk": row["Labor risk"],
"NetJob": row["NetJob"],
                "Cảnh báo": alert_text,
                "Cảnh báo": text,
}
)

return pd.DataFrame(rows)


def recommendation_table(kpi_df, risk_df):
def recommendations(kpi_df, risk_df):
merged = kpi_df.merge(
risk_df[["Kịch bản", "Mức cảnh báo", "Cảnh báo"]],
on="Kịch bản",
@@ -679,147 +515,29 @@ def recommendation_table(kpi_df, risk_df):
rows = []

for _, row in merged.iterrows():
        scenario = row["Kịch bản"]

        if scenario.startswith("S1"):
            priority = "Giảm tỷ trọng K, tăng D và H"
            rec = "Không nên chọn làm trục trung tâm. Phù hợp làm kịch bản đối chứng vì tăng trưởng truyền thống nhưng chuyển đổi số chậm."
        elif scenario.startswith("S2"):
            priority = "Tăng cyber governance và kỹ năng số"
            rec = "Phù hợp để tăng tốc chính phủ số và kinh tế số; cần kiểm soát rủi ro dữ liệu và đào tạo kỹ năng số."
        elif scenario.startswith("S3"):
            priority = "AI đi kèm H, an ninh mạng và chuẩn dữ liệu"
            rec = "Tạo AI readiness cao, nhưng không nên triển khai nguyên trạng nếu rủi ro lao động và phụ thuộc công nghệ tăng mạnh."
        elif scenario.startswith("S4"):
            priority = "Ưu tiên vùng yếu, SME, giáo dục số"
            rec = "Phù hợp mục tiêu bao trùm, giảm phân hóa vùng và hỗ trợ lao động; tốc độ tăng trưởng có thể thấp hơn."
        sc = row["Kịch bản"]

        if sc.startswith("S1"):
            rec = "Không nên chọn làm trục chính; chỉ phù hợp làm kịch bản đối chứng."
        elif sc.startswith("S2"):
            rec = "Phù hợp để tăng tốc số hóa, nhưng cần tăng an ninh dữ liệu và đào tạo kỹ năng số."
        elif sc.startswith("S3"):
            rec = "Tạo năng lực AI cao, nhưng cần kiểm soát rủi ro lao động và phụ thuộc công nghệ."
        elif sc.startswith("S4"):
            rec = "Tốt cho bao trùm và an sinh, nhưng cần bổ sung động lực tăng trưởng."
else:
            priority = "Triển khai làm baseline chính sách"
            rec = "Khuyến nghị chọn làm kịch bản nền vì cân bằng giữa tăng trưởng, số hóa, AI, lao động và rủi ro."
            rec = "Nên chọn làm kịch bản nền vì cân bằng giữa tăng trưởng, số hóa, AI, việc làm và rủi ro."

rows.append(
{
                "Kịch bản": scenario,
                "Xếp hạng tổng hợp": row["Xếp hạng tổng hợp"],
                "AIDEOM score": row["AIDEOM score"],
                "Kịch bản": sc,
                "Xếp hạng": row["Xếp hạng"],
"Mức cảnh báo": row["Mức cảnh báo"],
                "Ưu tiên điều chỉnh": priority,
                "Khuyến nghị": rec,
                "Khuyến nghị ngắn": rec,
}
)

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
    return pd.DataFrame(rows).sort_values("Xếp hạng").reset_index(drop=True)


def make_styled_table(df, decimals=3):
@@ -829,7 +547,7 @@ def make_styled_table(df, decimals=3):

for col in show_df.columns:
if pd.api.types.is_numeric_dtype(show_df[col]):
            if str(col).lower() in ["năm", "xếp hạng tổng hợp"]:
            if str(col).lower() in ["năm", "xếp hạng"]:
format_dict[col] = "{:.0f}"
else:
format_dict[col] = "{:." + str(decimals) + "f}"
@@ -922,472 +640,200 @@ def style_base_fig(fig, height=430):


def render():
    st.title("Bài 12. AIDEOM-VN tích hợp nâng cao")
    st.caption("Dashboard ra quyết định: M1-M6, kịch bản, S5 tự tối ưu, KPI 2030, cảnh báo rủi ro và khuyến nghị chính sách")
    st.title("Bài 12. AIDEOM-VN tích hợp")
    st.caption("Dashboard tinh gọn: kịch bản, KPI 2030, cảnh báo rủi ro và khuyến nghị chính sách")

    macro, macro_source = load_macro_data()
    regions, region_source = load_region_data()
    sectors, sector_source = load_sector_data()
    macro = load_macro_data()
    regions = load_region_data()
    sectors = load_sector_data()

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

        annual_budget = st.slider("Ngân sách mô phỏng hằng năm", 500, 3000, 1000, 100)
    base = get_base_values(macro, regions, sectors)
    scenarios = build_scenarios(base)
    scenarios_df = pd.DataFrame(scenarios)

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
    sim_df = simulate_all(scenarios, base)
    kpi_df = kpi_2030(sim_df)
    risk_df = risk_alerts(sim_df)
    rec_df = recommendations(kpi_df, risk_df)

    base = base_values_from_data(macro, regions, sectors)
    s5, candidate_df = optimize_s5_allocation(
        base=base,
        score_weights=score_weights,
        annual_budget=annual_budget,
        step=0.05,
    )
    best_scenario = kpi_df.iloc[0]["Kịch bản"]
    best_score = kpi_df.iloc[0]["AIDEOM score"]

    scenarios = build_scenarios(s5)
    scenarios_df = pd.DataFrame(scenarios)
    st.header("12.1. Yêu cầu chức năng")
    st.markdown("Mô hình **AIDEOM-VN** gồm 6 module liên kết theo cấu trúc Mục 14 của bài báo nguồn.")
    show_table(pd.DataFrame(MODULES), decimals=3)

    sim_df = simulate_scenarios(
        scenarios=scenarios,
        macro=macro,
        regions=regions,
        sectors=sectors,
        annual_budget=annual_budget,
    st.header("12.2. Năm kịch bản chính sách")
    show_table(
        scenarios_df[
            ["Kịch bản", "Mô tả ngắn", "Đặc điểm phân bổ", "K", "D", "AI", "H"]
        ],
        decimals=3,
)

    kpi_df = kpi_2030_table(sim_df, score_weights)
    risk_df = risk_alert_table(sim_df)
    rec_df = recommendation_table(kpi_df, risk_df)
    validation_df = validation_checks(scenarios, sim_df, kpi_df, risk_df, candidate_df)

    best_scenario = kpi_df.iloc[0]["Kịch bản"]
    best_score = kpi_df.iloc[0]["AIDEOM score"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Số module", "6")
    c2.metric("Kịch bản tốt nhất", best_scenario)
    c3.metric("AIDEOM score", f"{best_score:.3f}")

tabs = st.tabs(
[
            "Tổng quan AIDEOM-VN",
"Đường kịch bản",
"KPI năm 2030",
            "S5 tối ưu tự động",
"Cảnh báo rủi ro",
            "Tác nhân AI phân tích",
            "Thảo luận chính sách",
"Khuyến nghị chính sách",
            "Kiểm định & tải dữ liệu",
]
)

with tabs[0]:
        st.header("12.1. Yêu cầu chức năng")

        st.markdown(
            """
            Mô hình **AIDEOM-VN** gồm 6 module liên kết theo cấu trúc Mục 14 của bài báo nguồn.
            Bài 12 đóng vai trò tích hợp kết quả từ các bài trước thành dashboard hỗ trợ ra quyết định.
            """
        )

        show_table(modules_df(), decimals=3)

        st.header("12.2. Năm kịch bản chính sách")

        show_table(
            scenarios_df[
                ["Kịch bản", "Mô tả ngắn", "Đặc điểm phân bổ", "K", "D", "AI", "H", "Loại"]
            ],
            decimals=3,
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Số module", "6")
        c2.metric("Số kịch bản", "5")
        c3.metric("Kịch bản tốt nhất", best_scenario)
        c4.metric("AIDEOM score", f"{best_score:.3f}")

        alloc_long = scenarios_df.melt(
            id_vars=["Kịch bản"],
            value_vars=["K", "D", "AI", "H"],
            var_name="Hạng mục",
            value_name="Tỷ trọng",
        )

        fig_alloc = px.bar(
            alloc_long,
            x="Kịch bản",
            y="Tỷ trọng",
            color="Hạng mục",
            title="Cấu trúc phân bổ K/D/AI/H theo 5 kịch bản",
            color_discrete_map=MULTI_COLORS,
        )
        fig_alloc.update_traces(marker_line_color="white", marker_line_width=1)
        fig_alloc.update_layout(
            barmode="stack",
            xaxis_title="Kịch bản",
            yaxis_title="Tỷ trọng ngân sách",
        )
        style_base_fig(fig_alloc, height=470)
        st.plotly_chart(fig_alloc, use_container_width=True)

        st.success("Bản nâng cấp đã có S5 tự tối ưu theo trọng số KPI, không còn là tỷ trọng cố định.")

    with tabs[1]:
st.header("Đường kịch bản 2026-2030")

        metric_choice = st.selectbox(
        metric = st.selectbox(
"Chọn chỉ tiêu",
[
"GDP",
"GDP growth (%)",
"Digital Index",
"AI Readiness",
                "Lao động đào tạo (%)",
                "Kinh tế số/GDP (%)",
"NetJob",
                "DisplacedJob",
"Risk score",
],
index=0,
)

        fig_line = px.line(
        fig = px.line(
sim_df,
x="Năm",
            y=metric_choice,
            y=metric,
color="Kịch bản",
markers=True,
            title=f"Đường kịch bản theo chỉ tiêu: {metric_choice}",
            title=f"Đường kịch bản theo {metric}",
color_discrete_map=MULTI_COLORS,
)
        fig_line.update_traces(line=dict(width=4), marker=dict(size=8))
        fig_line.update_layout(xaxis_title="Năm", yaxis_title=metric_choice)
        style_base_fig(fig_line, height=500)
        st.plotly_chart(fig_line, use_container_width=True)
        fig.update_traces(line=dict(width=4), marker=dict(size=8))
        fig.update_layout(xaxis_title="Năm", yaxis_title=metric)
        style_base_fig(fig, height=480)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Dữ liệu mô phỏng 2026-2030")
        show_table(sim_df, decimals=3)

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
            [
                "Kịch bản",
                "GDP",
                "Digital Index",
                "AI Readiness",
                "Lao động đào tạo (%)",
                "Kinh tế số/GDP (%)",
                "NetJob",
                "Risk score",
            ]
        ].melt(
        alloc_long = scenarios_df.melt(
id_vars="Kịch bản",
            var_name="KPI",
            value_name="Giá trị",
            value_vars=["K", "D", "AI", "H"],
            var_name="Hạng mục",
            value_name="Tỷ trọng",
)

        fig_kpi = px.bar(
            kpi_long,
        fig_alloc = px.bar(
            alloc_long,
x="Kịch bản",
            y="Giá trị",
            color="KPI",
            barmode="group",
            title="So sánh KPI năm 2030",
            y="Tỷ trọng",
            color="Hạng mục",
            title="Cấu trúc phân bổ K/D/AI/H",
            color_discrete_map=MULTI_COLORS,
)
        fig_kpi.update_traces(marker_line_color="white", marker_line_width=1)
        fig_kpi.update_layout(xaxis_title="Kịch bản", yaxis_title="Giá trị KPI")
        style_base_fig(fig_kpi, height=540)
        st.plotly_chart(fig_kpi, use_container_width=True)
        fig_alloc.update_traces(marker_line_color="white", marker_line_width=1)
        fig_alloc.update_layout(barmode="stack", xaxis_title="Kịch bản", yaxis_title="Tỷ trọng")
        style_base_fig(fig_alloc, height=430)
        st.plotly_chart(fig_alloc, use_container_width=True)

    with tabs[1]:
        st.header("KPI năm 2030")

        show_table(kpi_df, decimals=3)

score_plot = kpi_df.sort_values("AIDEOM score", ascending=True)

        fig_score = px.bar(
        fig = px.bar(
score_plot,
x="AIDEOM score",
y="Kịch bản",
orientation="h",
text=score_plot["AIDEOM score"].round(3),
title="Xếp hạng AIDEOM score năm 2030",
)
        fig_score.update_traces(
            marker_color=BRAND,
            textposition="outside",
            textfont=dict(color=BRAND),
        )
        fig_score.update_layout(xaxis_title="AIDEOM score", yaxis_title="Kịch bản")
        style_base_fig(fig_score, height=430)
        st.plotly_chart(fig_score, use_container_width=True)

        st.download_button(
            "Tải KPI 2030 CSV",
            data=kpi_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="bai12_kpi_2030.csv",
            mime="text/csv",
            key="download_kpi_2030",
        )

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
        fig.update_traces(marker_color=BRAND, textposition="outside", textfont=dict(color=BRAND))
        fig.update_layout(xaxis_title="AIDEOM score", yaxis_title="Kịch bản")
        style_base_fig(fig, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
    with tabs[2]:
st.header("Cảnh báo rủi ro")

show_table(risk_df, decimals=3)

risk_long = risk_df[
            [
                "Kịch bản",
                "Risk score",
                "Cyber risk",
                "Environmental risk",
                "Dependency risk",
                "Labor transition risk",
            ]
            ["Kịch bản", "Risk score", "Cyber risk", "Environmental risk", "Labor risk"]
].melt(
id_vars="Kịch bản",
var_name="Loại rủi ro",
value_name="Điểm rủi ro",
)

        fig_risk = px.bar(
        fig = px.bar(
risk_long,
x="Kịch bản",
y="Điểm rủi ro",
color="Loại rủi ro",
barmode="group",
            title="Bản đồ rủi ro năm 2030",
            title="Rủi ro theo kịch bản năm 2030",
)
        fig_risk.update_traces(marker_line_color="white", marker_line_width=1)
        fig_risk.update_layout(xaxis_title="Kịch bản", yaxis_title="Điểm rủi ro 0-100")
        style_base_fig(fig_risk, height=520)
        st.plotly_chart(fig_risk, use_container_width=True)
        fig.update_traces(marker_line_color="white", marker_line_width=1)
        fig.update_layout(xaxis_title="Kịch bản", yaxis_title="Điểm rủi ro")
        style_base_fig(fig, height=470)
        st.plotly_chart(fig, use_container_width=True)

        red_alert = risk_df[risk_df["Mức cảnh báo"] == "Đỏ"]
        red = risk_df[risk_df["Mức cảnh báo"] == "Đỏ"]

        if len(red_alert) > 0:
            st.warning("Có kịch bản mức Đỏ: " + ", ".join(red_alert["Kịch bản"].tolist()))
        if len(red) > 0:
            st.warning("Có kịch bản cần can thiệp: " + ", ".join(red["Kịch bản"].tolist()))
else:
            st.success("Không có kịch bản nào ở mức Đỏ theo ngưỡng hiện tại.")
            st.success("Không có kịch bản nào ở mức cảnh báo Đỏ.")

        st.download_button(
            "Tải cảnh báo rủi ro CSV",
            data=risk_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="bai12_risk_alerts.csv",
            mime="text/csv",
            key="download_risk_alerts",
        )

    with tabs[5]:
        st.header("Tác nhân AI phân tích kết quả")
    with tabs[3]:
        st.header("Thảo luận chính sách")

        analysis_text, weights_df = ai_policy_analysis(kpi_df, risk_df, rec_df, s5, score_weights)
        st.markdown(
            f"""
            **1. Kịch bản nổi bật:**  
            Theo điểm tổng hợp, kịch bản tốt nhất là **{best_scenario}**.

        st.markdown(analysis_text)
            **2. Ý nghĩa của S5:**  
            S5 không cố định trước mà được mô hình tự chọn theo tiêu chí cân bằng giữa GDP, số hóa, AI, NetJob và rủi ro.

        st.subheader("Trọng số tác nhân đang sử dụng")
        show_table(weights_df, decimals=3)
            **3. Đánh đổi chính sách:**  
            S3 thường mạnh về AI nhưng có thể làm tăng rủi ro lao động và phụ thuộc công nghệ.  
            S4 tốt cho bao trùm nhưng tốc độ tăng trưởng có thể thấp hơn.  
            S5 là phương án trung dung, phù hợp làm baseline chính sách.

        st.info(
            "Tác nhân này là phân tích offline/rule-based để tránh lỗi API key khi deploy. "
            "Nội dung vẫn dựa trên kết quả định lượng của dashboard."
            **4. Nguyên tắc sử dụng mô hình:**  
            AIDEOM-VN chỉ là công cụ hỗ trợ ra quyết định, không thay thế đánh giá chính trị - xã hội.
            """
)

    with tabs[6]:
    with tabs[4]:
st.header("Khuyến nghị chính sách")

show_table(rec_df, decimals=3)

        st.markdown("### Khuyến nghị trung tâm")

        st.success(f"Chọn **{best_scenario}** làm kịch bản nền cho giai đoạn 2026-2030.")
        st.success(f"Khuyến nghị chính: chọn **{best_scenario}** làm kịch bản nền giai đoạn 2026-2030.")

st.markdown(
"""
            **Một là, dùng S5 tối ưu cân bằng làm baseline.**  
            Chính sách không nên cực đoan theo hướng chỉ đầu tư hạ tầng truyền thống hoặc chỉ chạy theo AI.

            **Hai là, AI phải đi cùng nhân lực số.**  
            Mọi dự án AI quy mô lớn cần có cấu phần đào tạo lại, nâng kỹ năng, chuyển đổi việc làm và an sinh lao động.

            **Ba là, dữ liệu và an ninh mạng là điều kiện nền.**  
            Nếu số hóa nhanh nhưng thiếu quản trị dữ liệu, rủi ro cyber và dependency có thể tăng mạnh.
            **Khuyến nghị cô đọng:**

            **Bốn là, ưu tiên vùng yếu và SME.**  
            Nếu chỉ ưu tiên vùng có năng lực hấp thụ cao, chuyển đổi số có thể làm tăng khoảng cách vùng miền.

            **Năm là, giữ vai trò con người trong quyết định cuối cùng.**  
            AIDEOM-VN là công cụ hỗ trợ ra quyết định, không thay thế quy trình chính trị - xã hội.
            - Không nên chỉ đầu tư theo hướng truyền thống S1.
            - Nếu chọn S2 hoặc S3, cần tăng mạnh an ninh dữ liệu và đào tạo lại lao động.
            - Nếu ưu tiên công bằng xã hội, S4 là kịch bản hỗ trợ tốt.
            - S5 nên được dùng làm phương án nền vì cân bằng giữa tăng trưởng, AI, việc làm và rủi ro.
            - Cần theo dõi hằng năm các KPI: GDP, Digital Index, AI Readiness, NetJob và Risk score.
           """
)

        tracking = pd.DataFrame(
            {
                "Nhóm KPI": [
                    "Tăng trưởng",
                    "Số hóa",
                    "AI",
                    "Lao động",
                    "Rủi ro",
                    "Bao trùm",
                ],
                "Chỉ tiêu đề xuất": [
                    "GDP, TFP, năng suất lao động",
                    "Digital Index, tỷ trọng kinh tế số/GDP",
                    "AI Readiness, số dự án AI công - tư",
                    "NetJob, tỷ lệ lao động được đào tạo lại",
                    "Cyber risk, dependency risk, environmental risk",
                    "Khoảng cách số vùng, SME digital adoption",
                ],
                "Tần suất": [
                    "Hằng năm",
                    "Hằng năm",
                    "6 tháng - 1 năm",
                    "6 tháng",
                    "Quý - 6 tháng",
                    "Hằng năm",
                ],
            }
        )

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
        st.download_button(
            "Tải bảng KPI 2030 CSV",
            data=kpi_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="bai12_kpi_2030.csv",
mime="text/csv",
            key="download_s5_candidates",
        )

        st.info(
            "Bài 12 bản nâng cấp đã có đủ: M1-M6, S1-S5, S5 tự tối ưu, KPI 2030, cảnh báo rủi ro, phân tích chính sách và tải dữ liệu."
)


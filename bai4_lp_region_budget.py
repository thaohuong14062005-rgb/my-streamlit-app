# bai04_regional_budget.py

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

BRAND = "#053151"

# ------------------------------
# Dữ liệu vùng và hạng mục
# ------------------------------
REGIONS = ["NMM", "RRD", "NCC", "CH", "SE", "MD"]
REGION_LABELS = {
    "NMM": "Trung du & miền núi Bắc Bộ",
    "RRD": "Đồng bằng sông Hồng",
    "NCC": "Bắc Trung Bộ & DHMT",
    "CH": "Tây Nguyên",
    "SE": "Đông Nam Bộ",
    "MD": "Đồng bằng sông Cửu Long",
}

ITEMS = ["I", "D", "AI", "H"]
ITEM_LABELS = {
    "I": "Hạ tầng số",
    "D": "Dữ liệu/nền tảng",
    "AI": "AI",
    "H": "Nhân lực số",
}

MULTI_COLORS = {
    "Hạ tầng số": "#053151",
    "Dữ liệu/nền tảng": "#E76F51",
    "AI": "#2A9D8F",
    "Nhân lực số": "#F4A261",
    "Có công bằng": "#053151",
    "Không công bằng": "#E76F51",
}

BETA = {
    ("NMM", "I"): 1.15, ("NMM", "D"): 0.85, ("NMM", "AI"): 0.55, ("NMM", "H"): 1.30,
    ("RRD", "I"): 0.95, ("RRD", "D"): 1.25, ("RRD", "AI"): 1.40, ("RRD", "H"): 1.05,
    ("NCC", "I"): 1.05, ("NCC", "D"): 0.95, ("NCC", "AI"): 0.85, ("NCC", "H"): 1.15,
    ("CH", "I"): 1.20, ("CH", "D"): 0.75, ("CH", "AI"): 0.45, ("CH", "H"): 1.35,
    ("SE", "I"): 0.90, ("SE", "D"): 1.30, ("SE", "AI"): 1.55, ("SE", "H"): 1.00,
    ("MD", "I"): 1.10, ("MD", "D"): 0.85, ("MD", "AI"): 0.65, ("MD", "H"): 1.25,
}

D0 = {
    "NMM": 38,
    "RRD": 78,
    "NCC": 55,
    "CH": 32,
    "SE": 82,
    "MD": 48,
}

# ------------------------------
# Bảng Beta
# ------------------------------
def beta_table():
    rows = []
    for r in REGIONS:
        row = {"Vùng": r, "Tên vùng": REGION_LABELS[r]}
        for j in ITEMS:
            row[ITEM_LABELS[j]] = BETA[(r, j)]
        rows.append(row)
    return pd.DataFrame(rows)

# ------------------------------
# Solve bằng PuLP
# ------------------------------
try:
    import pulp
    PULP_AVAILABLE = True
except Exception:
    PULP_AVAILABLE = False

def solve_pulp(budget=50000.0, floor_region=5000.0, cap_region=12000.0, h_min=12000.0,
               gamma=0.002, lam=0.68, fairness=True, use_cap=True):
    if not PULP_AVAILABLE:
        return {"success": False, "status": "PuLP chưa cài", "x": None, "objective": np.nan, "M": np.nan}

    model = pulp.LpProblem("VN_Digital_Budget", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", (REGIONS, ITEMS), lowBound=0)
    model += pulp.lpSum(BETA[(r, j)] * x[r][j] for r in REGIONS for j in ITEMS), "Z"
    model += pulp.lpSum(x[r][j] for r in REGIONS for j in ITEMS) <= budget

    for r in REGIONS:
        model += pulp.lpSum(x[r][j] for j in ITEMS) >= floor_region
        if use_cap:
            model += pulp.lpSum(x[r][j] for j in ITEMS) <= cap_region

    model += pulp.lpSum(x[r]["H"] for r in REGIONS) >= h_min

    M = None
    if fairness:
        M = pulp.LpVariable("Dmax", lowBound=0)
        for r in REGIONS:
            model += D0[r] + gamma * x[r]["D"] <= M
            model += D0[r] + gamma * x[r]["D"] >= lam * M

    solver = pulp.PULP_CBC_CMD(msg=False)
    status_code = model.solve(solver)
    status = pulp.LpStatus[status_code]

    if status != "Optimal":
        return {"success": False, "status": status, "x": None, "objective": np.nan, "M": np.nan}

    x_matrix = pd.DataFrame({r: {j: float(pulp.value(x[r][j])) for j in ITEMS} for r in REGIONS}).T
    obj = float(pulp.value(model.objective))
    m_value = float(pulp.value(M)) if fairness and M is not None else np.nan

    return {"success": True, "status": status, "x": x_matrix, "objective": obj, "M": m_value}

# ------------------------------
# Hiển thị bảng & long format
# ------------------------------
def matrix_display(x_matrix):
    out = x_matrix.copy()
    out.index = [REGION_LABELS[r] for r in out.index]
    out = out.rename(columns=ITEM_LABELS)
    out.insert(0, "Vùng", out.index)
    out = out.reset_index(drop=True)
    return out

def long_allocation(x_matrix):
    rows = []
    for r in REGIONS:
        for j in ITEMS:
            rows.append({
                "Vùng": REGION_LABELS[r],
                "Mã vùng": r,
                "Hạng mục": ITEM_LABELS[j],
                "Mã hạng mục": j,
                "Ngân sách": float(x_matrix.loc[r, j]),
                "Beta": BETA[(r, j)],
                "Đóng góp Z": float(x_matrix.loc[r, j]) * BETA[(r, j)]
            })
    return pd.DataFrame(rows)

def region_summary(x_matrix):
    long_df = long_allocation(x_matrix)
    region_df = long_df.groupby(["Mã vùng", "Vùng"], as_index=False).agg(
        **{"Tổng ngân sách": ("Ngân sách", "sum"),
           "Tổng đóng góp Z": ("Đóng góp Z", "sum")}
    ).sort_values("Tổng ngân sách", ascending=False).reset_index(drop=True)

    top_item = long_df.sort_values(["Mã vùng", "Ngân sách"], ascending=[True, False])\
                      .groupby("Mã vùng").head(1)[["Mã vùng", "Hạng mục", "Ngân sách"]]\
                      .rename(columns={"Hạng mục": "Hạng mục ưu tiên", "Ngân sách": "Ngân sách hạng mục ưu tiên"})
    region_df = region_df.merge(top_item, on="Mã vùng", how="left")
    return region_df

def digital_fairness_table(x_matrix, gamma=0.002):
    rows = []
    for r in REGIONS:
        d_invest = float(x_matrix.loc[r, "D"])
        d_after = D0[r] + gamma * d_invest
        rows.append({"Vùng": REGION_LABELS[r], "D0": D0[r], "Đầu tư D": d_invest, "D sau đầu tư": d_after})
    return pd.DataFrame(rows).sort_values("D sau đầu tư", ascending=False).reset_index(drop=True)

# ------------------------------
# Styling bảng
# ------------------------------
def make_styled_table(df, decimals=3):
    show_df = df.copy()
    format_dict = {col: f"{{:.{decimals}f}}" for col in show_df.columns if pd.api.types.is_numeric_dtype(show_df[col])}
    styler = show_df.style.format(format_dict)
    try: styler = styler.hide(axis="index")
    except Exception: pass
    styler = styler.set_properties(**{"background-color": "#ffffff", "color": BRAND,
                                      "border": f"1px solid {BRAND}", "padding": "10px 12px", "font-size": "16px"})
    styler = styler.set_table_styles([
        {"selector": "thead th", "props":[("background-color","#ffffff"),("color",BRAND),("font-weight","800"),
                                          ("border",f"1px solid {BRAND}"),("padding","12px"),("font-size","16px"),
                                          ("text-align","left")]},
        {"selector": "tbody td", "props":[("background-color","#ffffff"),("color",BRAND),("border",f"1px solid {BRAND}")]},
        {"selector": "table", "props":[("border-collapse","collapse"),("width","100%"),("background-color","#ffffff")]}
    ])
    return styler

def show_table(df, decimals=3):
    st.table(make_styled_table(df, decimals=decimals))

# ------------------------------
# Styling Plotly
# ------------------------------
def style_base_fig(fig, height=430):
    fig.update_layout(height=height, plot_bgcolor="white", paper_bgcolor="white",
                      font=dict(color=BRAND, size=15),
                      title_font=dict(color=BRAND, size=20),
                      xaxis=dict(showline=True, linecolor=BRAND, tickfont=dict(color=BRAND)),
                      yaxis=dict(showline=True, linecolor=BRAND, gridcolor="rgba(5,49,81,0.18)", tickfont=dict(color=BRAND)),
                      legend=dict(font=dict(color=BRAND)))
    return fig

# ------------------------------
# Render Streamlit
# ------------------------------
def render():
    st.title("Bài 4. Phân bổ ngân sách số theo vùng")
    st.caption("PuLP, Heatmap phân bổ, ràng buộc công bằng vùng miền và chi phí công bằng")

    if not PULP_AVAILABLE:
        st.error("Chưa cài PuLP. Hãy thêm `pulp` vào requirements.txt.")
        return

    budget = 50000.0
    floor_region = 5000.0
    cap_region = 12000.0
    h_min = 12000.0
    gamma = 0.002
    lam = 0.68

    pulp_res = solve_pulp(budget, floor_region, cap_region, h_min, gamma, lam, fairness=True)
    pulp_no_fair = solve_pulp(budget, floor_region, cap_region, h_min, gamma, lam, fairness=False)
    pulp_no_cap = solve_pulp(budget, floor_region, cap_region, h_min, gamma, lam, fairness=True, use_cap=False)

    tabs = st.tabs(["4.4.1 PuLP","4.4.3 Heatmap","4.4.4 Chi phí công bằng","4.5 Chính sách"])

    # Tab PuLP
    with tabs[0]:
        st.header("4.4.1 Mô hình PuLP và nghiệm tối ưu")
        show_table(beta_table(), decimals=3)
        if not pulp_res["success"]:
            st.error(f"Mô hình PuLP không khả thi. Trạng thái: {pulp_res['status']}")
        else:
            x_matrix = pulp_res["x"]
            region_df = region_summary(x_matrix)
            top_region = region_df.iloc[0]["Vùng"]

            st.subheader("Ma trận phân bổ tối ưu 6×4")
            show_table(matrix_display(x_matrix), decimals=2)

            st.subheader("Tổng hợp theo vùng")
            show_table(region_df, decimals=2)

            long_df = long_allocation(x_matrix)
            fig = px.bar(long_df, x="Vùng", y="Ngân sách", color="Hạng mục",
                         color_discrete_map=MULTI_COLORS, title="Phân bổ ngân sách theo vùng và hạng mục")
            fig.update_traces(marker_line_color="white", marker_line_width=1)
            fig.update_layout(barmode="stack", xaxis_title="Vùng", yaxis_title="Tỷ VND")
            style_base_fig(fig, height=470)
            st.plotly_chart(fig, use_container_width=True)

# Hàm run cho Streamlit
def run():
    render()

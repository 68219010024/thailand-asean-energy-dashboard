"""
แดชบอร์ดไฟฟ้าไทยและอาเซียน (World Energy Consumption)
ข้อมูล: data/energy_clean.csv (ผ่านการทำความสะอาด 5 ขั้น และสำรวจข้อมูล 4 ขั้น)

รันบนเครื่อง:  streamlit run app.py
"""
import hmac
import io
import os
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="ไฟฟ้าไทยและอาเซียน", page_icon=":material/bolt:", layout="wide")

# ---------------------------------------------------------------------------
# 1) คำศัพท์ภาษาไทย (ตรวจทานแล้ว ใช้ชุดเดียวทั้งแอป) และสีเชื้อเพลิง
# ---------------------------------------------------------------------------
FUELS = {  # คีย์: (ชื่อไทย, สี)
    "coal": ("ถ่านหิน", "#4A4039"),
    "oil": ("น้ำมัน", "#8C5A3C"),
    "gas": ("ก๊าซธรรมชาติ", "#E0913A"),
    "nuclear": ("นิวเคลียร์", "#7B61A8"),
    "hydro": ("พลังน้ำ", "#2E78AD"),
    "wind": ("พลังลม", "#5FB3A1"),
    "solar": ("พลังแสงอาทิตย์", "#E8B923"),
    "biofuel": ("ชีวมวล", "#7A9A3A"),
    "other_renewable_exc_biofuel": ("พลังงานหมุนเวียนอื่น", "#9AA5A8"),
}
FUEL_TWH = {f: f"{f}_electricity" for f in FUELS}
FUEL_SHARE = {f: f"{f}_share_elec" for f in FUELS}
FUEL_SHARE["other_renewable_exc_biofuel"] = "other_renewables_share_elec_exc_biofuel"
FUEL_COLOR = {th: c for th, c in FUELS.values()}
FUEL_ORDER = [v[0] for v in FUELS.values()]

# ชื่อคอลัมน์ภาษาไทย (พร้อมหน่วย) ใช้ในตาราง หัวกราฟ และไฟล์ที่ดาวน์โหลด
LABELS = {
    "country": "ประเทศ", "region_th": "ทวีป", "year": "ปี",
    "population": "ประชากร (คน)", "gdp": "GDP (ดอลลาร์สากล ปรับค่าครองชีพ)",
    "electricity_generation": "ไฟฟ้าที่ผลิตได้ (TWh)",
    "electricity_demand": "ความต้องการใช้ไฟฟ้า (TWh)",
    "net_elec_imports": "ไฟฟ้านำเข้าสุทธิ (TWh)",
    "per_capita_electricity": "ไฟฟ้าต่อประชากร (kWh/คน)",
    "carbon_intensity_elec": "การปล่อย CO₂ ต่อหน่วยไฟฟ้า (gCO₂/kWh)",
    "greenhouse_gas_emissions": "ก๊าซเรือนกระจกจากภาคไฟฟ้า (MtCO₂e)",
    "primary_energy_consumption": "การใช้พลังงานปฐมภูมิ (TWh)",
    "energy_per_capita": "พลังงานปฐมภูมิต่อประชากร (kWh/คน)",
    "fossil_electricity": "ไฟฟ้าจากเชื้อเพลิงฟอสซิล (TWh)",
    "renewables_electricity": "ไฟฟ้าจากพลังงานหมุนเวียน (TWh)",
    "low_carbon_electricity": "ไฟฟ้าจากพลังงานคาร์บอนต่ำ (TWh)",
    "fossil_share_elec": "สัดส่วนเชื้อเพลิงฟอสซิล (%)",
    "renewables_share_elec": "สัดส่วนพลังงานหมุนเวียน (%)",
    "low_carbon_share_elec": "สัดส่วนพลังงานคาร์บอนต่ำ (%)",
    "year_complete": "ข้อมูลปีนี้ครบถ้วน", "gdp_available": "มีข้อมูล GDP",
    "is_outlier_pc_elec": "ค่าผิดปกติ (ไฟฟ้าต่อประชากร)", "low_precision": "ข้อมูลความละเอียดต่ำ",
    "gdp_per_capita": "GDP ต่อประชากร (ดอลลาร์สากล)",
}
for _f, (_th, _) in FUELS.items():
    LABELS[FUEL_TWH[_f]] = f"ไฟฟ้าจาก{_th} (TWh)"
    LABELS[FUEL_SHARE[_f]] = f"สัดส่วน{_th} (%)"

METRICS = {  # ตัวชี้วัดในหน้าเปรียบเทียบ: (ชื่อไทย, หน่วย, ทศนิยม)
    "per_capita_electricity": ("ไฟฟ้าต่อประชากร", "kWh/คน", 0),
    "renewables_share_elec": ("สัดส่วนพลังงานหมุนเวียน", "%", 1),
    "fossil_share_elec": ("สัดส่วนเชื้อเพลิงฟอสซิล", "%", 1),
    "gas_share_elec": ("สัดส่วนก๊าซธรรมชาติ", "%", 1),
    "coal_share_elec": ("สัดส่วนถ่านหิน", "%", 1),
    "solar_share_elec": ("สัดส่วนพลังแสงอาทิตย์", "%", 1),
    "carbon_intensity_elec": ("การปล่อย CO₂ ต่อหน่วยไฟฟ้า", "gCO₂/kWh", 0),
    "electricity_generation": ("ไฟฟ้าที่ผลิตได้", "TWh", 1),
    "electricity_demand": ("ความต้องการใช้ไฟฟ้า", "TWh", 1),
}
REGION_TH = {"Asia": "เอเชีย", "Europe": "ยุโรป", "Africa": "แอฟริกา", "North America": "อเมริกาเหนือ",
             "South America": "อเมริกาใต้", "Oceania": "โอเชียเนีย"}
QUALITY_TH = {"แถว": "จำนวนแถว", "คอลัมน์": "จำนวนคอลัมน์", "ประเทศ/ชื่อ": "จำนวนประเทศและรายชื่อ",
              "กลุ่มภูมิภาคปน (นับซ้ำได้)": "กลุ่มภูมิภาคที่ปะปน", "คีย์ซ้ำ (%)": "ข้อมูลซ้ำ (%)",
              "ค่าว่างเฉลี่ยทั้งตาราง (%)": "ค่าว่างเฉลี่ยทั้งตาราง (%)", "ค่าว่างฟีเจอร์สำคัญ (%)": "ค่าว่างของตัวแปรสำคัญ (%)",
              "0 ปลอม (ผลิตไฟฟ้า=0 แต่มีประชากร)": "ค่า 0 ที่ไม่ถูกต้อง (แถว)"}
SOURCE_NOTE = "ที่มา: Our World in Data (Energy) · ปี 2000–2022 · ปี 2022 เป็นข้อมูลเบื้องต้น"

FONT = "IBM Plex Sans Thai, sans-serif"
NAVY, INK, MUTED, LINE, PAPER = "#13212E", "#1C2733", "#5B6773", "#E1E7EC", "#F3F6F8"
BLUE, GREY_BAR, ORANGE = "#1F5F8B", "#C3CDD6", "#E0913A"
GOOD, BAD = "#1E7D4F", "#B3412E"
GRID = "#E6EBF0"
LINE_COLORS = ["#1F5F8B", "#E0913A", "#5FB3A1", "#7B61A8", "#B3412E", "#7A9A3A", "#8C5A3C", "#E8B923", "#4A4039", "#9AA5A8"]
PLOT_CFG = {"displaylogo": False, "displayModeBar": "hover", "toImageButtonOptions": {"format": "png", "scale": 2}}

# ---------------------------------------------------------------------------
# 2) ธีมหน้าเว็บ (CSS)
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, textarea, select, label {{ font-family: {FONT} !important; }}
.stApp {{ background: {PAPER}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1320px; }}
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
h1 {{ font-weight: 700; }}

/* แถบด้านข้าง */
[data-testid="stSidebar"] {{ background: {NAVY}; }}
[data-testid="stSidebar"] * {{ color: #D5DEE6; }}
[data-testid="stSidebarNav"] a {{ border-radius: 8px; padding: 0.4rem 0.6rem; }}
[data-testid="stSidebarNav"] a:hover {{ background: rgba(255,255,255,0.06); }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{ background: rgba(255,255,255,0.12); }}
[data-testid="stSidebarNav"] a[aria-current="page"] span {{ color: #FFFFFF; font-weight: 600; }}
[data-testid="stSidebar"] button {{ background: transparent; border: 1px solid rgba(255,255,255,0.25); }}
.brand {{ font-size: 1.15rem; font-weight: 700; color: #FFFFFF !important; line-height: 1.35; }}
.brand-sub {{ font-size: 0.82rem; color: #8FA1B1 !important; }}

/* หัวหน้า (hero) */
.hero {{ background: {NAVY}; border-radius: 14px; padding: 1.7rem 2rem 0 2rem; color: #fff; overflow: hidden; margin-bottom: 1.2rem; }}
.hero .kick {{ color: #8FA1B1; font-size: 0.85rem; letter-spacing: 0.04em; margin-bottom: 0.3rem; }}
.hero h1 {{ color: #fff; font-size: 2.1rem; font-weight: 700; margin: 0 0 0.35rem 0; line-height: 1.25; }}
.hero p {{ color: #B9C6D2; margin: 0 0 1.2rem 0; font-size: 1.02rem; max-width: 64ch; }}
.hero .stripe-label {{ display: flex; justify-content: space-between; color: #8FA1B1; font-size: 0.8rem; margin-bottom: 0.35rem; }}
.stripe {{ display: flex; height: 14px; margin: 0 -2rem; }}
.stripe span {{ display: block; height: 100%; }}

/* ตัวเลขสรุป */
.kpi {{ background: #fff; border: 1px solid {LINE}; border-radius: 12px; padding: 1rem 1.15rem 0.95rem; height: 100%; }}
.kpi .label {{ color: {MUTED}; font-size: 0.9rem; margin-bottom: 0.2rem; }}
.kpi .value {{ color: {INK}; font-size: 1.9rem; font-weight: 700; line-height: 1.2; }}
.kpi .value small {{ font-size: 0.95rem; font-weight: 500; color: {MUTED}; margin-left: 0.25rem; }}
.kpi .delta {{ font-size: 0.86rem; margin-top: 0.35rem; color: {MUTED}; }}
.kpi .delta.good {{ color: {GOOD}; }} .kpi .delta.bad {{ color: {BAD}; }}
.kpi.thai {{ border-left: 5px solid {BLUE}; }}

/* แผงกราฟ */
[class*="st-key-panel"] {{ background: #fff; border-radius: 12px; border-color: {LINE} !important; }}
.panel-title {{ font-size: 1.1rem; font-weight: 600; color: {INK}; margin: 0.1rem 0; line-height: 1.4; }}
.panel-sub {{ color: {MUTED}; font-size: 0.86rem; margin-bottom: 0.2rem; }}
.section {{ font-size: 1.3rem; font-weight: 700; color: {INK}; margin: 1.6rem 0 0.6rem 0; padding-left: 0.7rem;
            border-left: 4px solid {BLUE}; line-height: 1.3; }}
.note {{ border-top: 1px solid {LINE}; padding: 0.7rem 0.1rem 0.1rem; margin-top: 0.2rem; line-height: 1.65;
         font-size: 0.92rem; color: {INK}; display: grid; grid-template-columns: 6.2rem 1fr; gap: 0.15rem 0.6rem; }}
.note b {{ color: {BLUE}; font-weight: 600; }}
.crumb {{ color: {MUTED}; font-size: 0.88rem; }}
.source {{ color: {MUTED}; font-size: 0.8rem; margin-top: 1.2rem; }}
.legend-row {{ display: flex; flex-wrap: wrap; gap: 0.3rem 1rem; font-size: 0.84rem; color: {MUTED}; margin: 0.2rem 0 0.4rem; }}
.legend-row i {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 0.35rem; }}

/* การ์ดคำตอบ และตาราง */
.answer {{ background: #fff; border: 1px solid {LINE}; border-radius: 12px; padding: 1.1rem 1.2rem; height: 100%; }}
.answer .q {{ color: {MUTED}; font-size: 0.86rem; line-height: 1.5; min-height: 2.6rem; }}
.answer .a {{ color: {BLUE}; font-size: 1.75rem; font-weight: 700; margin: 0.35rem 0 0.25rem; line-height: 1.2; }}
.answer .t {{ color: {INK}; font-size: 0.9rem; line-height: 1.55; }}
.tbl {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; }}
.tbl th {{ text-align: left; color: {MUTED}; font-weight: 600; border-bottom: 2px solid {INK}; padding: 0.55rem 0.6rem; }}
.tbl td {{ border-bottom: 1px solid {LINE}; padding: 0.6rem; vertical-align: top; line-height: 1.5; color: {INK}; }}
.tbl td:first-child {{ font-weight: 600; }}

/* แท็บ */
.stTabs [data-baseweb="tab-list"] {{ gap: 0.4rem; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ padding: 0.5rem 0.9rem; font-weight: 500; }}
.stTabs [aria-selected="true"] {{ color: {BLUE} !important; }}
@media (max-width: 640px) {{ .hero h1 {{ font-size: 1.6rem; }} .kpi .value {{ font-size: 1.5rem; }} .note {{ grid-template-columns: 1fr; }} }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 3) โหลดข้อมูล (แคช = โหลดครั้งเดียว)
# ---------------------------------------------------------------------------
def find_file(name):
    for folder in ["data", ".", os.path.join("..", "final"), "final"]:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder, name)
        if os.path.exists(path):
            return path
    return None


@st.cache_data(show_spinner=False)
def load_data():
    t0 = time.perf_counter()
    path = find_file("energy_clean.csv")
    if path is None:
        raise FileNotFoundError("energy_clean.csv")
    df = pd.read_csv(path)
    for c in ["country", "iso_code", "region"]:
        df[c] = df[c].astype("category")
    df["region_th"] = df["region"].map(REGION_TH).astype("category")
    df["gdp_per_capita"] = df["gdp"] / df["population"]
    load_ms = (time.perf_counter() - t0) * 1000

    # คำนวณภาพรวมโลกไว้ล่วงหน้าครั้งเดียว (เฉพาะปีที่ข้อมูลครบ)
    t1 = time.perf_counter()
    twh = ["electricity_generation"] + list(FUEL_TWH.values()) + ["renewables_electricity", "fossil_electricity"]
    world = df[df["year_complete"]].groupby("year")[twh].sum()
    world["renewables_share"] = world["renewables_electricity"] / world["electricity_generation"] * 100
    world["fossil_share"] = world["fossil_electricity"] / world["electricity_generation"] * 100
    for f in FUELS:
        world[FUEL_SHARE[f]] = world[FUEL_TWH[f]] / world["electricity_generation"] * 100
    agg_ms = (time.perf_counter() - t1) * 1000
    return df, world, {"rows": len(df), "load_ms": load_ms, "agg_ms": agg_ms}


@st.cache_data(show_spinner=False)
def load_side_table(name):
    path = find_file(name)
    return pd.read_csv(path) if path else None


# ---------------------------------------------------------------------------
# 4) ตัวช่วยแสดงผล
# ---------------------------------------------------------------------------
def fmt(v, dec=1, unit=""):
    if v is None or pd.isna(v):
        return "ไม่มีข้อมูล"
    return f"{v:,.{dec}f} {unit}".strip()


def kpi(col, label, value, unit="", delta="", tone="", thai=False, help_text=""):
    """การ์ดตัวเลขสรุป tone = good / bad / '' (กลาง)"""
    title = f' title="{help_text}"' if help_text else ""
    col.markdown(f'<div class="kpi{" thai" if thai else ""}"{title}><div class="label">{label}</div>'
                 f'<div class="value">{value}<small>{unit}</small></div>'
                 f'<div class="delta {tone}">{delta}</div></div>', unsafe_allow_html=True)


def tone_of(change, higher_is_better=True):
    if change is None or pd.isna(change) or abs(change) < 1e-9:
        return ""
    return "good" if (change > 0) == higher_is_better else "bad"


def arrow(v):
    return "เพิ่มขึ้น" if v >= 0 else "ลดลง"


_panel_no = [0]


def panel():
    """แผงสีขาวมีขอบ ใช้ห่อกราฟแต่ละชิ้น"""
    _panel_no[0] += 1
    return st.container(border=True, key=f"panel_{_panel_no[0]}")


def panel_title(title, sub=""):
    st.markdown(f'<div class="panel-title">{title}</div>' + (f'<div class="panel-sub">{sub}</div>' if sub else ""),
                unsafe_allow_html=True)


def section(text):
    st.markdown(f'<div class="section">{text}</div>', unsafe_allow_html=True)


def breadcrumb(*parts):
    st.markdown(f'<div class="crumb">ภาพรวม › {" › ".join(parts)}</div>', unsafe_allow_html=True)


def note(finding, meaning, action=None):
    """คำอธิบายใต้กราฟ: ข้อค้นพบ → ความหมาย → ข้อเสนอแนะ (What → So What → Now What, เกณฑ์ 3.5.3.2.1)"""
    html = f'<div class="note"><b>ข้อค้นพบ</b><span>{finding}</span><b>ความหมาย</b><span>{meaning}</span>'
    if action:
        html += f'<b>ข้อเสนอแนะ</b><span>{action}</span>'
    st.markdown(html + "</div>", unsafe_allow_html=True)


def fuel_legend():
    items = "".join(f'<span><i style="background:{c}"></i>{th}</span>' for th, c in FUELS.values())
    st.markdown(f'<div class="legend-row">{items}</div>', unsafe_allow_html=True)


def source(extra=""):
    st.markdown(f'<div class="source">{SOURCE_NOTE}{" · " + extra if extra else ""}</div>', unsafe_allow_html=True)


def empty_state(message, reset_keys=None):
    """เมื่อไม่มีข้อมูล: แจ้งสาเหตุและแนวทางแก้ไข (เกณฑ์ 3.5.4.2.3)"""
    st.info(message)
    if reset_keys:
        def _reset():
            for k in reset_keys:
                st.session_state.pop(k, None)
        st.button("ล้างตัวกรอง", key=f"reset_{reset_keys[0]}", on_click=_reset)


def show(fig):
    st.plotly_chart(fig, width="stretch", config=PLOT_CFG)


def style_fig(fig, height=340, ytitle=None, xtitle=None, legend=True):
    """ธีมกราฟ: พื้นโปร่งใสให้กลืนกับแผงสีขาว เส้นตารางจาง ตัวอักษรสีเดียวกับหน้าเว็บ"""
    fig.update_layout(height=height, font=dict(family=FONT, size=13, color=INK),
                      margin=dict(l=4, r=8, t=8 if not legend else 36, b=4),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=legend,
                      colorway=LINE_COLORS,
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title=None, font=dict(size=12),
                                  bgcolor="rgba(0,0,0,0)"),
                      hoverlabel=dict(font_family=FONT, bgcolor="white", bordercolor=LINE, font_color=INK),
                      modebar=dict(bgcolor="rgba(0,0,0,0)", color="#9AA5A8", activecolor=BLUE))
    fig.update_xaxes(showgrid=False, title=xtitle, linecolor=LINE, tickfont=dict(color=MUTED), title_font=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, title=ytitle, zeroline=False, tickfont=dict(color=MUTED), title_font=dict(color=MUTED))
    return fig


def to_excel_bytes(frame):
    buf = io.BytesIO()
    try:
        with pd.ExcelWriter(buf) as xw:
            frame.to_excel(xw, index=False, sheet_name="data")
        return buf.getvalue()
    except ModuleNotFoundError:
        return None


def html_table(rows, headers):
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    st.markdown(f'<table class="tbl"><tr>{head}</tr>{body}</table>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 5) กราฟ (ใช้ซ้ำได้ทุกหน้า)
# ---------------------------------------------------------------------------
def fig_mix_strip(rows, height=210):
    """แถบ 100% สัดส่วนเชื้อเพลิง rows = [(ชื่อแถว, dict ของสัดส่วน)]"""
    recs = [{"กลุ่ม": name, "เชื้อเพลิง": FUELS[f][0], "สัดส่วน": vals.get(FUEL_SHARE[f], np.nan)}
            for name, vals in rows for f in FUELS]
    long = pd.DataFrame(recs).dropna()
    fig = px.bar(long, y="กลุ่ม", x="สัดส่วน", color="เชื้อเพลิง", orientation="h",
                 color_discrete_map=FUEL_COLOR, category_orders={"เชื้อเพลิง": FUEL_ORDER})
    fig.update_traces(hovertemplate="%{y} · %{fullData.name}: %{x:.1f}%<extra></extra>",
                      textposition="inside", insidetextanchor="middle", textfont=dict(color="white", size=12))
    fig.for_each_trace(lambda t: t.update(texttemplate=[f"{v:.0f}%" if v >= 6 else "" for v in t.x]))
    fig = style_fig(fig, height, legend=False)
    fig.update_xaxes(range=[0, 100], showticklabels=False)
    fig.update_yaxes(title=None, autorange="reversed", tickfont=dict(size=14, color=INK), gridcolor="rgba(0,0,0,0)")
    fig.update_layout(bargap=0.3)
    return fig


def solid_area(fig):
    """ให้พื้นที่กราฟเป็นสีทึบตรงกับสีเชื้อเพลิง"""
    fig.for_each_trace(lambda t: t.update(fillcolor=FUEL_COLOR.get(t.name, None), line=dict(width=0.6, color="white")))
    return fig


def fig_world_trend(world, year):
    trend = world.loc[:year, list(FUEL_TWH.values())].rename(columns={v: FUELS[k][0] for k, v in FUEL_TWH.items()})
    long = trend.reset_index().melt("year", var_name="เชื้อเพลิง", value_name="TWh")
    fig = px.area(long, x="year", y="TWh", color="เชื้อเพลิง", color_discrete_map=FUEL_COLOR,
                  category_orders={"เชื้อเพลิง": FUEL_ORDER})
    fig.update_traces(hovertemplate="ปี %{x}: %{y:,.0f} TWh<extra>%{fullData.name}</extra>")
    fig = style_fig(solid_area(fig), 330, "TWh", None, legend=False)
    fig.update_yaxes(rangemode="tozero")
    return fig


def fig_rank(frame, metric, dec, unit, highlight="Thailand", height=330, horizontal=True):
    f = frame.sort_values(metric, ascending=horizontal)
    colors = [BLUE if c == highlight else GREY_BAR for c in f["country"]]
    text = f[metric].map(lambda v: f"{v:,.{dec}f}")
    if horizontal:
        fig = go.Figure(go.Bar(x=f[metric], y=f["country"].astype(str), orientation="h", marker_color=colors,
                               text=text, textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}: %{x:,." + str(dec) + "f} " + unit + "<extra></extra>"))
        fig = style_fig(fig, height, None, None, legend=False)
        fig.update_xaxes(showticklabels=False, range=[0, f[metric].max() * 1.15])
        fig.update_yaxes(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=INK))
    else:
        fig = go.Figure(go.Bar(x=f["country"].astype(str), y=f[metric], marker_color=colors, text=text,
                               textposition="outside", cliponaxis=False,
                               hovertemplate="%{x}: %{y:,." + str(dec) + "f} " + unit + "<extra></extra>"))
        fig = style_fig(fig, height, unit, None, legend=False)
        fig.update_xaxes(categoryorder="array", categoryarray=f["country"].astype(str).tolist())
        if len(f) > 15:
            fig.update_traces(text=None)
            fig.update_xaxes(tickangle=-60, tickfont=dict(size=10))
    fig.update_layout(bargap=0.35)
    return fig


def fig_gen_demand(d):
    fig = go.Figure()
    fig.add_scatter(x=d["year"], y=d["electricity_demand"], name="ความต้องการใช้ไฟฟ้า", mode="lines",
                    line=dict(color=INK, width=2.5))
    fig.add_scatter(x=d["year"], y=d["electricity_generation"], name="ผลิตในประเทศ", mode="lines",
                    line=dict(color=BLUE, width=2.5), fill="tonexty", fillcolor="rgba(224,145,58,0.16)")
    fig.update_traces(hovertemplate="ปี %{x}: %{y:,.1f} TWh")
    fig = style_fig(fig, 300, "TWh")
    fig.update_yaxes(rangemode="tozero")
    return fig


def fig_mix_time(d):
    long = d.melt("year", value_vars=list(FUEL_SHARE.values()), var_name="col", value_name="สัดส่วน")
    long["เชื้อเพลิง"] = long["col"].map({v: FUELS[k][0] for k, v in FUEL_SHARE.items()})
    fig = px.area(long.dropna(), x="year", y="สัดส่วน", color="เชื้อเพลิง", color_discrete_map=FUEL_COLOR,
                  category_orders={"เชื้อเพลิง": FUEL_ORDER})
    fig.update_traces(hovertemplate="ปี %{x}: %{y:.1f}%<extra>%{fullData.name}</extra>")
    fig = style_fig(solid_area(fig), 300, "%", legend=False)
    fig.update_yaxes(range=[0, 100], ticksuffix="%")
    return fig


def fig_scatter(df, year, highlight):
    allc = df[(df["year"] == year) & ~df["low_precision"]].dropna(
        subset=["per_capita_electricity", "renewables_share_elec"])
    allc = allc[allc["per_capita_electricity"] > 0]
    fig = px.scatter(allc, x="per_capita_electricity", y="renewables_share_elec", color="region_th",
                     hover_name="country", log_x=True,
                     color_discrete_sequence=["#9CB9D1", "#D9A0A0", "#D8C27A", "#8FC0A9", "#B3A3CF", "#C9A88B"],
                     labels={"per_capita_electricity": LABELS["per_capita_electricity"],
                             "renewables_share_elec": LABELS["renewables_share_elec"], "region_th": "ทวีป"})
    fig.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=0)))
    hl = allc[allc["country"].isin(highlight)]
    fig.add_scatter(x=hl["per_capita_electricity"], y=hl["renewables_share_elec"], mode="markers+text",
                    text=hl["country"].astype(str), textposition="top center", name="ประเทศที่เลือก",
                    textfont=dict(size=11, color=INK),
                    marker=dict(size=11, color=BLUE, line=dict(color="white", width=1.5)))
    r = allc[["per_capita_electricity", "renewables_share_elec"]].corr(method="spearman").iloc[0, 1]
    return style_fig(fig, 340, "สัดส่วนพลังงานหมุนเวียน (%)", "ไฟฟ้าต่อประชากร (kWh/คน, สเกลลอการิทึม)"), r, len(allc)


def fig_region_median(df, year):
    y = df[(df["year"] == year) & ~df["low_precision"]]
    med = y.groupby("region_th", observed=True)["per_capita_electricity"].median().sort_values()
    fig = go.Figure(go.Bar(x=med.values, y=med.index.astype(str), orientation="h", marker_color=GREY_BAR,
                           text=[f"{v:,.0f}" for v in med.values], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:,.0f} kWh/คน<extra></extra>"))
    fig = style_fig(fig, 300, None, None, legend=False)
    fig.update_xaxes(showticklabels=False, range=[0, med.max() * 1.2])
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=INK))
    return fig, med


def fig_carbon(d, med, name):
    fig = go.Figure()
    fig.add_scatter(x=med.index, y=med.values, name="มัธยฐานโลก", mode="lines", line=dict(color="#9AA5A8", width=2, dash="dash"))
    fig.add_scatter(x=d["year"], y=d["carbon_intensity_elec"], name=name, mode="lines+markers",
                    line=dict(color=BLUE, width=2.5), marker=dict(size=5))
    fig.update_traces(hovertemplate="ปี %{x}: %{y:,.0f} gCO₂/kWh")
    fig = style_fig(fig, 300, "gCO₂/kWh")
    fig.update_yaxes(rangemode="tozero")
    return fig


# ---------------------------------------------------------------------------
# 6) การวิเคราะห์ (คำนวณจากข้อมูลจริง แคชไว้)
# ---------------------------------------------------------------------------
ANALYSIS_VARS = ["per_capita_electricity", "renewables_share_elec", "fossil_share_elec", "gas_share_elec",
                 "carbon_intensity_elec", "electricity_generation"]


def strength_th(r):
    a = abs(r)
    s = "แทบไม่มีความสัมพันธ์" if a < 0.2 else "ความสัมพันธ์ระดับต่ำ" if a < 0.4 else "ความสัมพันธ์ระดับปานกลาง" if a < 0.7 else "ความสัมพันธ์ระดับสูง"
    return s if a < 0.2 else s + (" ทิศทางเดียวกัน" if r > 0 else " ทิศทางตรงข้าม")


def skew_th(s):
    return "สมมาตรโดยประมาณ" if abs(s) < 0.5 else "เบ้เล็กน้อย" if abs(s) < 1 else "เบ้มาก"


@st.cache_data(show_spinner=False)
def research_answers():
    df, world, _ = load_data()
    latest = int(df.loc[df["year_complete"], "year"].max())
    th = df[df["country"] == "Thailand"].set_index("year").sort_index()
    out = {"latest": latest}
    d0, d1 = th.loc[2000, "electricity_demand"], th.loc[2022, "electricity_demand"]
    out["q1"] = dict(d0=d0, d1=d1, pct=(d1 / d0 - 1) * 100, cagr=((d1 / d0) ** (1 / 22) - 1) * 100)
    out["q2"] = dict(r0=th.loc[2000, "renewables_share_elec"], r1=th.loc[2022, "renewables_share_elec"],
                     w=world.loc[latest, "renewables_share"],
                     bio=th.loc[2022, "biofuel_share_elec"] - th.loc[2000, "biofuel_share_elec"],
                     hydro=th.loc[2022, "hydro_share_elec"] - th.loc[2000, "hydro_share_elec"])
    q3c = ["Thailand", "Vietnam", "Malaysia", "Indonesia", "Philippines"]
    g = df[df["country"].isin(q3c) & df["year"].between(2010, 2022)].pivot(index="year", columns="country", values="gas_share_elec")
    rank = g.rank(axis=1, ascending=False)["Thailand"]
    out["q3"] = dict(mean=g["Thailand"].mean(), std=g["Thailand"].std(), others=g.drop(columns="Thailand").stack().mean(),
                     years_first=int((rank == 1).sum()), years=len(rank))
    t = th.loc[2000:2018]
    gp = t["gdp"] / t["population"]
    gr = pd.DataFrame({"g": gp, "e": t["per_capita_electricity"]}).pct_change().dropna()
    out["q4"] = dict(r=gr.corr().iloc[0, 1], el=np.polyfit(np.log(gp), np.log(t["per_capita_electricity"]), 1)[0], n=len(gr))
    out["imp"] = dict(now=th.loc[latest, "net_elec_imports"] / th.loc[latest, "electricity_demand"] * 100,
                      then=th.loc[2000, "net_elec_imports"] / th.loc[2000, "electricity_demand"] * 100,
                      twh=th.loc[latest, "net_elec_imports"])
    out["gas_now"] = th.loc[latest, "gas_share_elec"]
    return out


@st.cache_data(show_spinner=False)
def describe_year(year):
    df, _, _ = load_data()
    y = df[df["year"] == year]
    th = y[y["country"] == "Thailand"]
    rows = []
    for c in ANALYSIS_VARS:
        s = y[c].dropna()
        q1, q3 = s.quantile([.25, .75])
        thv = th[c].iloc[0] if len(th) and pd.notna(th[c].iloc[0]) else np.nan
        rows.append({"ตัวแปร": LABELS[c], "n": len(s), "ค่าเฉลี่ย": s.mean(), "มัธยฐาน": s.median(),
                     "S.D.": s.std(), "IQR": q3 - q1, "ต่ำสุด": s.min(), "สูงสุด": s.max(),
                     "ความเบ้": s.skew(), "ค่าของไทย": thv,
                     "เปอร์เซ็นไทล์ไทย": (s < thv).mean() * 100 if pd.notna(thv) else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7) เข้าสู่ระบบ (บัญชีทดสอบสำหรับกรรมการ ตั้งค่าได้ใน .streamlit/secrets.toml)
# ---------------------------------------------------------------------------
def get_users():
    """อ่านบัญชีจาก Secrets ถ้าอ่านไม่ได้หรือว่าง ใช้บัญชีทดสอบเริ่มต้น"""
    default = {"judge": "energy2025"}
    try:
        users = {str(k).strip().lower(): str(v).strip().strip('"\'“”') for k, v in dict(st.secrets["users"]).items()}
        return {**default, **users} if users else default
    except Exception:
        return default


def login_page():
    stripe = "".join(f'<span style="width:{100 / len(FUELS):.2f}%;background:{c}"></span>' for _, c in FUELS.values())
    st.markdown('<div class="hero"><div class="kick">ระบบนำเสนอข้อมูลพลังงานไฟฟ้า</div><h1>ไฟฟ้าไทยและอาเซียน</h1>'
                '<p>แหล่งเชื้อเพลิงที่ใช้ผลิตไฟฟ้า ปริมาณการใช้ไฟฟ้า และความก้าวหน้าของพลังงานหมุนเวียน '
                'จากข้อมูล 211 ประเทศ ปี 2000–2022</p>'
                f'<div class="stripe" style="margin-top:0.4rem">{stripe}</div></div>', unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        with panel():
            panel_title("เข้าสู่ระบบ", "กรุณากรอกชื่อผู้ใช้และรหัสผ่านที่ได้รับ")
            with st.form("login", border=False):
                user = st.text_input("ชื่อผู้ใช้")
                pw = st.text_input("รหัสผ่าน", type="password")
                ok = st.form_submit_button("เข้าสู่ระบบ", type="primary", width="stretch")
            if ok:
                users = get_users()
                user = user.strip().lower()          # รองรับตัวพิมพ์ใหญ่/เว้นวรรคเกินจากมือถือ
                pw = pw.strip()
                if user in users and hmac.compare_digest(pw, users[user]):
                    st.session_state["user"] = user
                    st.rerun()
                else:
                    st.error("ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง กรุณาตรวจสอบภาษาของแป้นพิมพ์แล้วลองอีกครั้ง")
        source()


# ---------------------------------------------------------------------------
# 8) หน้าภาพรวม: รวมกราฟสำคัญไว้หน้าเดียว
# ---------------------------------------------------------------------------
def page_home():
    df, world, _ = DATA
    years = sorted(world.index)

    c1, c2 = st.columns([2.2, 1])
    year = c1.select_slider("ปีที่แสดง (เฉพาะปีที่ข้อมูลครบถ้วน)", options=years, value=years[-1], key="hm_year")
    countries = sorted(df["country"].astype(str).unique())
    focus = c2.selectbox("ประเทศที่ต้องการวิเคราะห์", countries, index=countries.index("Thailand"), key="hm_focus")
    base = years[0]
    w, w0 = world.loc[year], world.loc[base]

    stripe = "".join(f'<span style="width:{w[FUEL_SHARE[f]]:.2f}%;background:{c}" title="{th} {w[FUEL_SHARE[f]]:.1f}%"></span>'
                     for f, (th, c) in FUELS.items() if w[FUEL_SHARE[f]] > 0)
    fname = "ไทย" if focus == "Thailand" else focus
    st.markdown(f'<div class="hero"><div class="kick">ภาพรวม ปี {year}</div>'
                f'<h1>การผลิตไฟฟ้าของโลกและ{"ประเทศไทย" if focus == "Thailand" else " " + focus}</h1>'
                f'<p>ปี {year} โลกผลิตไฟฟ้า {w["electricity_generation"]:,.0f} TWh มาจากเชื้อเพลิงฟอสซิล {w["fossil_share"]:.0f}% '
                f'และพลังงานหมุนเวียน {w["renewables_share"]:.0f}% แถบสีด้านล่างแสดงสัดส่วนเชื้อเพลิงของโลกในปีดังกล่าว</p>'
                f'<div class="stripe-label"><span>เชื้อเพลิงฟอสซิล</span><span>พลังงานคาร์บอนต่ำ</span></div>'
                f'<div class="stripe">{stripe}</div></div>', unsafe_allow_html=True)

    y = df[df["year"] == year]
    fc = y[y["country"] == focus]
    if fc.empty or pd.isna(fc.iloc[0]["electricity_generation"]):
        empty_state(f"ไม่พบข้อมูลไฟฟ้าของ {focus} ในปี {year} กรุณาเลือกปีอื่น หรือกดล้างตัวกรองเพื่อกลับสู่ค่าเริ่มต้น",
                    ["hm_year", "hm_focus"])
        return
    fc = fc.iloc[0]
    hist = df[df["country"] == focus].sort_values("year")
    fc0 = hist[hist["year"] == base].iloc[0] if (hist["year"] == base).any() else fc
    asean = y[y["is_asean"]].dropna(subset=["renewables_share_elec"])

    k = st.columns(4)
    g_chg = (w["electricity_generation"] / w0["electricity_generation"] - 1) * 100
    kpi(k[0], "ไฟฟ้าที่โลกผลิตได้", f"{w['electricity_generation']:,.0f}", "TWh", f"{arrow(g_chg)} {abs(g_chg):.1f}% จากปี {base}")
    r_chg = w["renewables_share"] - w0["renewables_share"]
    kpi(k[1], "สัดส่วนพลังงานหมุนเวียนของโลก", f"{w['renewables_share']:.1f}", "%",
        f"{arrow(r_chg)} {abs(r_chg):.1f} จุดร้อยละ จากปี {base}", tone_of(r_chg))
    kpi(k[2], f"{fname}: ไฟฟ้าต่อประชากร", fmt(fc["per_capita_electricity"], 0), "kWh/คน",
        f"มัธยฐานโลก {y['per_capita_electricity'].median():,.0f} kWh/คน", "", thai=True,
        help_text="ใช้มัธยฐานแทนค่าเฉลี่ย เนื่องจากข้อมูลมีความเบ้สูง")
    fr_chg = fc["renewables_share_elec"] - fc0["renewables_share_elec"]
    rank_txt = ""
    if focus in asean["country"].values:
        rank = int(asean["renewables_share_elec"].rank(ascending=False)[asean["country"] == focus].iloc[0])
        rank_txt = f" · อันดับ {rank} จาก {len(asean)} ในอาเซียน"
    kpi(k[3], f"{fname}: สัดส่วนพลังงานหมุนเวียน", fmt(fc["renewables_share_elec"], 1), "%",
        f"{arrow(fr_chg)} {abs(fr_chg):.1f} จุดร้อยละ{rank_txt}", tone_of(fr_chg), thai=True)

    section("แหล่งเชื้อเพลิงที่ใช้ผลิตไฟฟ้า")
    asean_tot = y[y["is_asean"]][list(FUEL_TWH.values()) + ["electricity_generation"]].sum()
    asean_row = {FUEL_SHARE[f]: asean_tot[FUEL_TWH[f]] / asean_tot["electricity_generation"] * 100 for f in FUELS}
    rows = [(fname, fc.to_dict()), ("อาเซียน", asean_row), ("โลก", w.to_dict())]
    with panel():
        main_f = max(FUELS, key=lambda f: fc[FUEL_SHARE[f]] if pd.notna(fc[FUEL_SHARE[f]]) else -1)
        panel_title(f"{fname}ผลิตไฟฟ้าจาก{FUELS[main_f][0]}เป็นหลัก ร้อยละ {fc[FUEL_SHARE[main_f]]:.0f}",
                    f"สัดส่วนเชื้อเพลิงในการผลิตไฟฟ้า ปี {year} เปรียบเทียบกับอาเซียนและโลก")
        fuel_legend()
        show(fig_mix_strip(rows))
        note(f"{fname}ใช้{FUELS[main_f][0]}ร้อยละ {fc[FUEL_SHARE[main_f]]:.1f} ขณะที่ค่าของโลกอยู่ที่ร้อยละ {w[FUEL_SHARE[main_f]]:.1f}",
             "การพึ่งพาเชื้อเพลิงชนิดเดียวในสัดส่วนสูง ทำให้ต้นทุนค่าไฟฟ้าผันผวนตามราคาเชื้อเพลิงนั้น",
             "ควรกระจายแหล่งเชื้อเพลิงเพื่อลดความเสี่ยงด้านราคา")

    section(f"แนวโน้มของ{fname} ตั้งแต่ปี 2000")
    h = hist[hist["year"] <= year]
    left, right = st.columns(2)
    with left, panel():
        imp = fc["net_elec_imports"]
        imp_share = imp / fc["electricity_demand"] * 100 if pd.notna(imp) and fc["electricity_demand"] else np.nan
        panel_title("การผลิตในประเทศไม่เพียงพอต่อความต้องการ" if pd.notna(imp) and imp > 0 else "การผลิตในประเทศเทียบกับความต้องการใช้ไฟฟ้า",
                    "หน่วย TWh ต่อปี · พื้นที่แรเงาคือส่วนต่างระหว่างความต้องการกับการผลิตในประเทศ")
        show(fig_gen_demand(h))
        if pd.notna(imp) and imp > 0:
            meaning = ("ไฟฟ้านำเข้าของไทยส่วนใหญ่มาจากเขื่อนพลังน้ำใน สปป.ลาว ความมั่นคงด้านไฟฟ้าจึงขึ้นอยู่กับประเทศเพื่อนบ้าน"
                       if focus == "Thailand" else "ประเทศนี้ต้องพึ่งพาไฟฟ้าจากต่างประเทศส่วนหนึ่ง")
            action = "ควรเพิ่มกำลังผลิตในประเทศ หรือจัดทำสัญญาซื้อไฟฟ้าระยะยาวเพื่อลดความเสี่ยง"
        elif pd.notna(imp) and imp < 0:
            meaning, action = "ประเทศนี้ผลิตไฟฟ้าได้มากกว่าความต้องการ และส่งออกส่วนเกิน", None
        else:
            meaning, action = "การผลิตในประเทศใกล้เคียงกับความต้องการ", None
        note(f"ปี {year} ไฟฟ้านำเข้าสุทธิ {fmt(imp, 1, 'TWh')} คิดเป็นร้อยละ {fmt(imp_share, 1)} ของความต้องการ", meaning, action)
    with right, panel():
        grow = max(FUELS, key=lambda f: (fc[FUEL_SHARE[f]] - fc0[FUEL_SHARE[f]])
                   if pd.notna(fc[FUEL_SHARE[f]]) and pd.notna(fc0[FUEL_SHARE[f]]) else -999)
        panel_title(f"{FUELS[grow][0]}มีสัดส่วนเพิ่มขึ้นมากที่สุด", "สัดส่วนเชื้อเพลิงตามช่วงเวลา (รวมร้อยละ 100)")
        fuel_legend()
        show(fig_mix_time(h))
        note(f"{FUELS[grow][0]}เพิ่มจากร้อยละ {fc0[FUEL_SHARE[grow]]:.1f} (ปี {base}) เป็นร้อยละ {fc[FUEL_SHARE[grow]]:.1f} (ปี {year})",
             "แต่ละสีแสดงสัดส่วนของเชื้อเพลิง จึงเห็นได้ว่าเชื้อเพลิงใดเข้ามาทดแทนเชื้อเพลิงใด")

    section("การเปรียบเทียบกับอาเซียนและโลก")
    left, right = st.columns([1, 1.25])
    with left, panel():
        top = asean.sort_values("renewables_share_elec").iloc[-1]
        panel_title(f"{top['country']} มีสัดส่วนพลังงานหมุนเวียนสูงสุดในอาเซียน", f"สัดส่วนพลังงานหมุนเวียน (%) ปี {year}")
        show(fig_rank(asean, "renewables_share_elec", 1, "%", highlight=focus, height=360))
        note(f"{top['country']} ร้อยละ {top['renewables_share_elec']:.1f} · มัธยฐานอาเซียน ร้อยละ {asean['renewables_share_elec'].median():.1f}",
             "ประเทศที่มีสัดส่วนสูงส่วนใหญ่มีแหล่งพลังน้ำขนาดใหญ่",
             "ไทยมีข้อจำกัดด้านพื้นที่สร้างเขื่อน จึงควรมุ่งพัฒนาพลังแสงอาทิตย์และพลังลม")
    with right, panel():
        sc, r, n = fig_scatter(df, year, [focus] + (["Laos", "Vietnam"] if focus == "Thailand" else []))
        panel_title("ปริมาณการใช้ไฟฟ้าไม่ได้กำหนดสัดส่วนพลังงานสะอาด", f"{n} ประเทศ ปี {year} · หนึ่งจุดแทนหนึ่งประเทศ")
        show(sc)
        note(f"ค่าสหสัมพันธ์ของสเปียร์แมน (Spearman) = {r:.2f} ({strength_th(r)})",
             "สัดส่วนพลังงานสะอาดขึ้นอยู่กับทรัพยากรธรรมชาติและนโยบาย มากกว่าระดับการใช้ไฟฟ้า",
             "ค่าสหสัมพันธ์ไม่ได้แสดงความเป็นเหตุเป็นผล ควรพิจารณาร่วมกับข้อมูลนโยบาย")

    left, right = st.columns([1.25, 1])
    with left, panel():
        panel_title(f"การผลิตไฟฟ้าของโลกเพิ่มขึ้นร้อยละ {g_chg:.0f} โดยยังพึ่งพาเชื้อเพลิงฟอสซิลเป็นหลัก",
                    f"หน่วย TWh ต่อปี แยกตามเชื้อเพลิง ปี {base}–{year}")
        fuel_legend()
        show(fig_world_trend(world, year))
        note(f"เชื้อเพลิงฟอสซิล ร้อยละ {w0['fossil_share']:.1f} → {w['fossil_share']:.1f} · พลังงานหมุนเวียน ร้อยละ {w0['renewables_share']:.1f} → {w['renewables_share']:.1f}",
             "พลังงานหมุนเวียนเติบโตต่อเนื่อง แต่ยังไม่ทันความต้องการไฟฟ้าที่เพิ่มขึ้น")
    with right, panel():
        rm, med = fig_region_median(df, year)
        panel_title(f"{med.index[-1]}มีการใช้ไฟฟ้าต่อประชากรสูงสุด", f"มัธยฐานไฟฟ้าต่อประชากร (kWh/คน) แยกตามทวีป ปี {year}")
        show(rm)
        note(f"{med.index[-1]} {med.iloc[-1]:,.0f} kWh/คน สูงกว่า{med.index[0]} ({med.iloc[0]:,.0f} kWh/คน) ประมาณ {med.iloc[-1] / med.iloc[0]:.0f} เท่า",
             "ความแตกต่างสะท้อนระดับการพัฒนาและการเข้าถึงไฟฟ้า")

    section("ข้อมูลเพิ่มเติม")
    n1, n2, n3, n4 = st.columns(4)
    for col, key, label, desc in [
        (n1, "country", "รายประเทศ", "แนวโน้มและการปล่อย CO₂ ของแต่ละประเทศ"),
        (n2, "compare", "เปรียบเทียบ", "จัดอันดับตามอาเซียน ทวีป หรือกลุ่มที่กำหนดเอง"),
        (n3, "analysis", "ผลการวิเคราะห์", "สถิติ ความสัมพันธ์ และข้อสรุปของการศึกษา"),
        (n4, "explorer", "ตารางข้อมูล", "ค้นหา กรอง และส่งออกเป็น CSV หรือ Excel"),
    ]:
        with col, panel():
            st.page_link(PAGES[key], label=label, icon=":material/arrow_forward:")
            st.caption(desc)
    source(f"ภาพรวมโลกใช้เฉพาะปีที่ข้อมูลครบถ้วน ({years[0]}–{years[-1]})")


# ---------------------------------------------------------------------------
# 9) หน้ารายประเทศ
# ---------------------------------------------------------------------------
def page_country():
    df, _, _ = DATA
    breadcrumb("รายประเทศ")
    st.title("รายประเทศ")
    countries = sorted(df["country"].astype(str).unique())
    c1, c2 = st.columns([1, 1.6])
    country = c1.selectbox("เลือกประเทศ (พิมพ์ชื่อภาษาอังกฤษเพื่อค้นหา)", countries, index=countries.index("Thailand"), key="ct_country")
    yr = c2.slider("ช่วงปี", 2000, 2022, (2000, 2022), key="ct_years")
    d = df[(df["country"] == country) & df["year"].between(*yr)].sort_values("year")
    d_ok = d.dropna(subset=["electricity_generation"])
    if d_ok.empty:
        empty_state(f"ไม่พบข้อมูลไฟฟ้าของ {country} ในช่วงปี {yr[0]}–{yr[1]} กรุณาขยายช่วงปีหรือเลือกประเทศอื่น", ["ct_years"])
        return
    first, last = d_ok.iloc[0], d_ok.iloc[-1]
    if not last["year_complete"]:
        st.caption(f"หมายเหตุ: ข้อมูลปี {int(last['year'])} เป็นข้อมูลเบื้องต้น อาจมีการปรับปรุงภายหลัง")
    if d["low_precision"].any():
        st.warning("ประเทศนี้ผลิตไฟฟ้าน้อยกว่า 0.1 TWh ตัวเลขในแหล่งข้อมูลถูกปัดเศษ ค่าสัดส่วนและการปล่อย CO₂ อาจคลาดเคลื่อน")

    k = st.columns(4)
    g = (last["electricity_generation"] / first["electricity_generation"] - 1) * 100 if first["electricity_generation"] else np.nan
    kpi(k[0], f"ไฟฟ้าที่ผลิตได้ (ปี {int(last['year'])})", fmt(last["electricity_generation"], 1), "TWh",
        f"{arrow(g)} {abs(g):.0f}% จากปี {int(first['year'])}" if pd.notna(g) else "")
    pc = last["per_capita_electricity"] - first["per_capita_electricity"]
    kpi(k[1], "ไฟฟ้าต่อประชากร", fmt(last["per_capita_electricity"], 0), "kWh/คน",
        f"{arrow(pc)} {abs(pc):,.0f} kWh/คน" if pd.notna(pc) else "")
    rs = last["renewables_share_elec"] - first["renewables_share_elec"]
    kpi(k[2], "สัดส่วนพลังงานหมุนเวียน", fmt(last["renewables_share_elec"], 1), "%",
        f"{arrow(rs)} {abs(rs):.1f} จุดร้อยละ" if pd.notna(rs) else "", tone_of(rs))
    ci = last["carbon_intensity_elec"] - first["carbon_intensity_elec"]
    kpi(k[3], "การปล่อย CO₂ ต่อหน่วยไฟฟ้า", fmt(last["carbon_intensity_elec"], 0), "g/kWh",
        f"{arrow(ci)} {abs(ci):.0f} g/kWh (ค่ายิ่งต่ำยิ่งดี)" if pd.notna(ci) else "ค่ายิ่งต่ำยิ่งดี",
        tone_of(ci, higher_is_better=False))

    section("สัดส่วนเชื้อเพลิง")
    with panel():
        panel_title(f"เปรียบเทียบปี {int(first['year'])} กับปี {int(last['year'])}")
        fuel_legend()
        show(fig_mix_strip([(f"ปี {int(first['year'])}", first.to_dict()), (f"ปี {int(last['year'])}", last.to_dict())], 170))
        main_f = max(FUELS, key=lambda f: last[FUEL_SHARE[f]] if pd.notna(last[FUEL_SHARE[f]]) else -1)
        note(f"เชื้อเพลิงหลักปี {int(last['year'])} คือ{FUELS[main_f][0]} ร้อยละ {last[FUEL_SHARE[main_f]]:.1f}",
             "การพึ่งพาเชื้อเพลิงชนิดเดียวในสัดส่วนสูง ทำให้ต้นทุนไฟฟ้าผันผวนตามราคาเชื้อเพลิงนั้น")

    left, right = st.columns(2)
    with left, panel():
        panel_title("การผลิตในประเทศเทียบกับความต้องการใช้ไฟฟ้า", "หน่วย TWh ต่อปี")
        show(fig_gen_demand(d))
    with right, panel():
        panel_title("สัดส่วนเชื้อเพลิงตามช่วงเวลา", "รวมร้อยละ 100")
        fuel_legend()
        show(fig_mix_time(d))

    with panel():
        med = df[df["year_complete"]].groupby("year")["carbon_intensity_elec"].median().loc[yr[0]:yr[1]]
        panel_title("การปล่อย CO₂ ต่อหน่วยไฟฟ้า เทียบกับมัธยฐานโลก", "กรัม CO₂ ต่อไฟฟ้า 1 kWh · ค่ายิ่งต่ำแสดงว่าไฟฟ้าสะอาดกว่า")
        show(fig_carbon(d, med, country))
        note(f"ปี {int(last['year'])} {country} ปล่อย CO₂ {fmt(last['carbon_intensity_elec'], 0, 'กรัม')} ต่อไฟฟ้า 1 kWh",
             "หากเส้นอยู่ต่ำกว่ามัธยฐานโลก แสดงว่าการผลิตไฟฟ้าสะอาดกว่าประเทศส่วนใหญ่",
             "การเพิ่มสัดส่วนพลังงานหมุนเวียนแทนถ่านหินและก๊าซธรรมชาติจะช่วยลดค่านี้")

    with st.expander("ตารางข้อมูลของประเทศนี้ และการดาวน์โหลด"):
        cols = ["year", "electricity_generation", "electricity_demand", "per_capita_electricity",
                "renewables_share_elec", "fossil_share_elec", "gas_share_elec", "carbon_intensity_elec"]
        out = d[cols].rename(columns=LABELS)
        st.dataframe(out, hide_index=True, width="stretch")
        st.download_button("ดาวน์โหลด CSV", out.to_csv(index=False).encode("utf-8-sig"), f"{country}_{yr[0]}-{yr[1]}.csv", "text/csv")
    source()


# ---------------------------------------------------------------------------
# 10) หน้าเปรียบเทียบ: เลือกชุดประเทศได้ (อาเซียน / ทวีป / 10 อันดับแรก / กำหนดเอง)
# ---------------------------------------------------------------------------
def country_groups(df):
    groups = {"อาเซียน": sorted(df.loc[df["is_asean"], "country"].astype(str).unique())}
    for en, th in REGION_TH.items():
        groups[f"ทวีป{th}"] = sorted(df.loc[df["region"] == en, "country"].astype(str).unique())
    y = df[df["year"] == int(df.loc[df["year_complete"], "year"].max())]
    groups["ผู้ผลิตไฟฟ้าสูงสุด 10 อันดับ"] = y.nlargest(10, "electricity_generation")["country"].astype(str).tolist()
    groups["ผู้ใช้ไฟฟ้าต่อประชากรสูงสุด 10 อันดับ"] = (y[~y["low_precision"]].nlargest(10, "per_capita_electricity")["country"]
                                                   .astype(str).tolist())
    return groups


def page_compare():
    df, _, _ = DATA
    breadcrumb("เปรียบเทียบ")
    st.title("เปรียบเทียบประเทศ")
    all_countries = sorted(df["country"].astype(str).unique())
    groups = country_groups(df)

    def apply_group():
        g = st.session_state.get("cp_group")
        if g in groups:
            st.session_state["cp_countries"] = groups[g]

    st.session_state.setdefault("cp_group", "อาเซียน")
    st.session_state.setdefault("cp_countries", groups["อาเซียน"])

    with panel():
        f1, f2, f3 = st.columns([1.2, 1, 0.8])
        metric = f1.selectbox("ตัวชี้วัด", list(METRICS), format_func=lambda c: f"{METRICS[c][0]} ({METRICS[c][1]})",
                              index=1, key="cp_metric")
        year = f2.select_slider("ปี", options=list(range(2000, 2023)), value=2021, key="cp_year")
        order = f3.radio("การเรียงลำดับ", ["มากไปน้อย", "น้อยไปมาก"], horizontal=True, key="cp_order")
        g1, g2 = st.columns([1, 2.2])
        g1.selectbox("ชุดประเทศ", list(groups) + ["กำหนดเอง"], key="cp_group", on_change=apply_group,
                     help="เลือกชุดสำเร็จรูป แล้วเพิ่มหรือลบประเทศได้ในช่องด้านขวา")
        countries = g2.multiselect("ประเทศ (พิมพ์ชื่อภาษาอังกฤษเพื่อค้นหา)", all_countries, key="cp_countries",
                                   on_change=lambda: st.session_state.update(cp_group="กำหนดเอง"))
        hide_small = st.checkbox("ไม่แสดงประเทศที่ผลิตไฟฟ้าน้อยกว่า 0.1 TWh (ข้อมูลความละเอียดต่ำ)", value=True, key="cp_small")
    name, unit, dec = METRICS[metric]

    if not countries:
        empty_state("ยังไม่ได้เลือกประเทศ กรุณาเลือกชุดประเทศ หรือเพิ่มประเทศอย่างน้อย 1 ประเทศ")
        return
    if year == 2022:
        st.caption("หมายเหตุ: ปี 2022 เป็นข้อมูลเบื้องต้น หลายประเทศยังไม่มีข้อมูล")
    sel = df[df["country"].isin(countries)]
    if hide_small:
        sel = sel[~sel["low_precision"]]
    y = sel[sel["year"] == year].dropna(subset=[metric])
    missing = sorted(set(countries) - set(y["country"].astype(str)))
    if y.empty:
        empty_state(f"ไม่พบข้อมูล{name}ของประเทศที่เลือกในปี {year} กรุณาเลือกปีอื่น (ปี 2021 มีข้อมูลครบถ้วนที่สุด)", ["cp_year"])
        return

    with panel():
        panel_title(f"{name} ปี {year}", f"หน่วย: {unit} · {len(y)} ประเทศ · ประเทศไทยแสดงเป็นสีน้ำเงิน")
        yy = y.sort_values(metric, ascending=(order == "น้อยไปมาก"))
        fig = fig_rank(yy, metric, dec, unit, horizontal=False, height=420 if len(yy) <= 15 else 480)
        fig.update_xaxes(categoryarray=yy["country"].astype(str).tolist())
        show(fig)
        med = y[metric].median()
        hi, lo = y.loc[y[metric].idxmax()], y.loc[y[metric].idxmin()]
        th_txt = ""
        if "Thailand" in y["country"].values:
            thv = y.loc[y["country"] == "Thailand", metric].iloc[0]
            th_txt = f" · ไทย {thv:,.{dec}f} {unit} ({'สูงกว่า' if thv > med else 'ต่ำกว่า'}มัธยฐานของกลุ่ม)"
        note(f"สูงสุดคือ {hi['country']} ({hi[metric]:,.{dec}f}) ต่ำสุดคือ {lo['country']} ({lo[metric]:,.{dec}f}) "
             f"มัธยฐานของกลุ่ม {med:,.{dec}f} {unit}{th_txt}",
             "ความแตกต่างระหว่างประเทศสะท้อนทรัพยากรและนโยบายพลังงานที่ต่างกัน",
             "ศึกษาประเทศที่มีผลการดำเนินงานดีกว่าเพิ่มเติมได้ในหน้ารายประเทศ")
        if missing:
            st.caption(f"ไม่มีข้อมูลในปีนี้: {', '.join(missing)}")

    with panel():
        t = sel.dropna(subset=[metric])
        shown = t["country"].astype(str).unique().tolist()
        if len(shown) > 12:   # ถ้าประเทศมาก แสดงเฉพาะ 10 อันดับแรก + ไทย เพื่อให้อ่านได้
            keep = y.nlargest(10, metric)["country"].astype(str).tolist()
            if "Thailand" in shown and "Thailand" not in keep:
                keep.append("Thailand")
            t = t[t["country"].isin(keep)]
            sub = f"แสดง 10 ประเทศที่มีค่าสูงสุดในปี {year}{' และประเทศไทย' if 'Thailand' in keep else ''} · คลิกชื่อประเทศเพื่อซ่อนหรือแสดง"
        else:
            sub = "เส้นสีน้ำเงินหนาคือประเทศไทย · คลิกชื่อประเทศเพื่อซ่อนหรือแสดง"
        panel_title(f"แนวโน้ม{name}", sub)
        fig = px.line(t, x="year", y=metric, color="country", labels={"year": "ปี", metric: f"{name} ({unit})", "country": "ประเทศ"})
        fig.update_traces(line=dict(width=1.8), opacity=0.7)
        fig.for_each_trace(lambda tr: tr.update(line=dict(width=4, color=BLUE), opacity=1) if tr.name == "Thailand" else None)
        show(style_fig(fig, 380, unit))
    source()


# ---------------------------------------------------------------------------
# 11) หน้าผลการวิเคราะห์ (สรุปผล + สถิติ + การกระจาย + ความสัมพันธ์ + คุณภาพข้อมูล)
# ---------------------------------------------------------------------------
def page_analysis():
    df, _, _ = DATA
    breadcrumb("ผลการวิเคราะห์")
    st.title("ผลการวิเคราะห์ข้อมูล")
    st.caption("สรุปผลจากการสำรวจข้อมูลเชิงสถิติ ตัวเลขทั้งหมดคำนวณจากชุดข้อมูลเดียวกับทุกหน้าของระบบ")
    a = research_answers()
    latest = a["latest"]
    complete_years = sorted(df.loc[df["year_complete"], "year"].unique())
    tabs = st.tabs(["สรุปผลการศึกษา", "สถิติเชิงพรรณนา", "การกระจายของข้อมูล", "ความสัมพันธ์ระหว่างตัวแปร", "คุณภาพข้อมูล"])

    # --- สรุปผล ---
    with tabs[0]:
        section("คำตอบของคำถามการศึกษา")
        q = [
            ("1. ความต้องการใช้ไฟฟ้าของไทยเพิ่มขึ้นเท่าใด (2000–2022)",
             f"+{a['q1']['pct']:.0f}%", f"จาก {a['q1']['d0']:.0f} เป็น {a['q1']['d1']:.0f} TWh อัตราการเติบโตเฉลี่ยร้อยละ {a['q1']['cagr']:.1f} ต่อปี"),
            ("2. สัดส่วนพลังงานหมุนเวียนของไทยเปลี่ยนแปลงอย่างไร",
             f"{a['q2']['r0']:.1f}% → {a['q2']['r1']:.1f}%", f"เพิ่มขึ้นจากชีวมวลเป็นหลัก แต่ยังต่ำกว่าค่าของโลก (ร้อยละ {a['q2']['w']:.1f} ในปี {latest})"),
            ("3. ไทยพึ่งพาก๊าซธรรมชาติมากกว่าประเทศเพื่อนบ้านหรือไม่",
             "ใช่", f"สูงสุดใน 5 ประเทศ {a['q3']['years_first']} จาก {a['q3']['years']} ปี เฉลี่ยร้อยละ {a['q3']['mean']:.1f} ± {a['q3']['std']:.1f} "
                    f"(ประเทศอื่นเฉลี่ยร้อยละ {a['q3']['others']:.1f})"),
            ("4. การใช้ไฟฟ้าต่อประชากรสัมพันธ์กับ GDP หรือไม่",
             f"r = {a['q4']['r']:.2f}", f"อัตราการเติบโตรายปีสัมพันธ์ในทิศทางเดียวกัน (ปี 2000–2018) GDP ต่อประชากรเพิ่มร้อยละ 1 "
                                        f"ไฟฟ้าต่อประชากรเพิ่มประมาณร้อยละ {a['q4']['el']:.1f}"),
        ]
        cols = st.columns(4)
        for col, (qq, aa, tt) in zip(cols, q):
            col.markdown(f'<div class="answer"><div class="q">{qq}</div><div class="a">{aa}</div><div class="t">{tt}</div></div>',
                         unsafe_allow_html=True)

        section("ข้อเสนอแนะ")
        with panel():
            html_table([
                (f"ใช้ก๊าซธรรมชาติร้อยละ {a['gas_now']:.1f} ของการผลิตไฟฟ้า", "กำหนดเป้าหมายลดสัดส่วนก๊าซธรรมชาติในแผนพัฒนากำลังผลิตไฟฟ้า (PDP)", "กระทรวงพลังงาน / กกพ.", "ภายใน 5 ปี"),
                (f"ความต้องการใช้ไฟฟ้าเติบโตร้อยละ {a['q1']['cagr']:.1f} ต่อปี", "เพิ่มกำลังผลิตจากพลังงานสะอาดแทนโรงไฟฟ้าก๊าซธรรมชาติแห่งใหม่", "กฟผ.", "แผน 10 ปี"),
                (f"พลังงานหมุนเวียนร้อยละ {a['q2']['r1']:.1f} ต่ำกว่าค่าของโลก", "ศึกษามาตรการจูงใจของเวียดนาม และขยายการติดตั้งพลังแสงอาทิตย์บนหลังคา", "หน่วยงานภาครัฐ", "1–3 ปี"),
                (f"ไฟฟ้านำเข้าร้อยละ {a['imp']['now']:.1f} ของความต้องการ", "กระจายแหล่งนำเข้า และเพิ่มกำลังผลิตสำรองในประเทศ", "กฟผ.", "ทบทวนทุกปี"),
            ], ["ข้อค้นพบ", "ข้อเสนอแนะ", "หน่วยงานที่เกี่ยวข้อง", "กรอบเวลา"])
        st.caption("ข้อจำกัด: ข้อมูล GDP สิ้นสุดปี 2018 · ข้อมูลปี 2022 เป็นข้อมูลเบื้องต้น · เป็นข้อมูลรายปีระดับประเทศ · "
                   "ค่าสหสัมพันธ์ไม่ได้แสดงความเป็นเหตุเป็นผล")

    # --- สถิติเชิงพรรณนา ---
    with tabs[1]:
        year = st.select_slider("ปีที่วิเคราะห์", options=complete_years, value=latest, key="an_desc_year")
        stats = describe_year(year)
        with panel():
            panel_title(f"สถิติสรุปของตัวแปรสำคัญ ปี {year}", "หน่วยการวิเคราะห์คือประเทศ · ค่าของไทยแสดงในสองคอลัมน์สุดท้าย")
            st.dataframe(stats, hide_index=True, width="stretch", column_config={
                c: st.column_config.NumberColumn(format="%.1f") for c in stats.columns if c not in ("ตัวแปร", "n")})
            st.caption("n = จำนวนประเทศ · S.D. = ส่วนเบี่ยงเบนมาตรฐาน · IQR = พิสัยระหว่างควอไทล์ที่ 1 และ 3 · "
                       "เปอร์เซ็นไทล์ไทย = ร้อยละของประเทศที่มีค่าต่ำกว่าไทย")
            top = stats.loc[stats["ความเบ้"].idxmax()]
            pc = stats.iloc[0]
            note(f"{top['ตัวแปร']} มีความเบ้สูงสุด ({top['ความเบ้']:.1f}) และ {pc['ตัวแปร']} มีค่าเฉลี่ยสูงกว่ามัธยฐาน "
                 f"{pc['ค่าเฉลี่ย'] / pc['มัธยฐาน']:.1f} เท่า",
                 "ข้อมูลมีความเบ้ขวา ค่าเฉลี่ยจึงถูกดึงสูงโดยประเทศจำนวนน้อย ระบบนี้จึงใช้มัธยฐานเป็นค่ากลางหลัก")
        st.download_button("ดาวน์โหลดตารางสถิติ (CSV)", stats.round(2).to_csv(index=False).encode("utf-8-sig"),
                           f"descriptive_stats_{year}.csv", "text/csv")

    # --- การกระจาย ---
    with tabs[2]:
        c1, c2, c3 = st.columns([1.4, 1, 0.8])
        var = c1.selectbox("ตัวแปร", ANALYSIS_VARS, format_func=lambda c: LABELS[c], key="an_dist_var")
        year = c2.select_slider("ปี", options=complete_years, value=latest, key="an_dist_year")
        log = c3.toggle("สเกลลอการิทึม", value=var in ("per_capita_electricity", "electricity_generation"), key="an_dist_log",
                        help="เหมาะกับตัวแปรที่ค่าต่างกันหลายร้อยเท่า")
        y = df[(df["year"] == year)].dropna(subset=[var])
        if log:
            y = y[y[var] > 0]
        s = y[var]
        thv = y.loc[y["country"] == "Thailand", var]
        left, right = st.columns(2)
        with left, panel():
            panel_title("ฮิสโทแกรม (Histogram)", f"จำนวนประเทศในแต่ละช่วงค่า ปี {year}")
            xv = np.log10(s) if log else s                 # สเกลลอการิทึม: แปลงค่าก่อนแบ่งช่วง แล้วแสดงป้ายเป็นค่าจริง
            fig = go.Figure(go.Histogram(x=xv, nbinsx=30, marker=dict(color=GREY_BAR, line=dict(color="white", width=1)),
                                         hovertemplate="%{y} ประเทศ<extra></extra>"))
            pos = lambda v: np.log10(v) if log else v
            fig.add_vline(x=pos(s.median()), line_color=ORANGE, line_dash="dash", annotation_text=f"มัธยฐาน {s.median():,.1f}",
                          annotation_position="top right", annotation_font_color=ORANGE)
            if len(thv):
                fig.add_vline(x=pos(thv.iloc[0]), line_color=BLUE, line_width=3, annotation_text=f"ไทย {thv.iloc[0]:,.1f}",
                              annotation_position="top left", annotation_font_color=BLUE)
            if log:
                ticks = [10 ** k for k in range(int(np.floor(xv.min())), int(np.ceil(xv.max())) + 1)]
                fig.update_xaxes(tickvals=[np.log10(t) for t in ticks], ticktext=[f"{t:,.0f}" if t >= 1 else f"{t:g}" for t in ticks])
            show(style_fig(fig, 340, "จำนวนประเทศ", LABELS[var], legend=False))
        with right, panel():
            panel_title("แผนภาพกล่อง (Boxplot) แยกตามทวีป", "เส้นกลางกล่อง = มัธยฐาน · จุด = ค่าที่อยู่นอกช่วงปกติ")
            fig = px.box(y, x=var, y="region_th", points="outliers", log_x=log, color_discrete_sequence=[BLUE],
                         labels={var: LABELS[var], "region_th": ""}, hover_name="country")
            fig.update_traces(fillcolor="rgba(31,95,139,0.12)", line_color=BLUE, marker=dict(color=ORANGE, size=6))
            fig = style_fig(fig, 340, None, LABELS[var], legend=False)
            if log:
                fig.update_xaxes(dtick=1, tickformat=",")
            show(fig)
        pct = (s < thv.iloc[0]).mean() * 100 if len(thv) else np.nan
        note(f"ความเบ้ {s.skew():.2f} ({skew_th(s.skew())}) · มัธยฐาน {s.median():,.1f} · ค่าของไทยอยู่ที่เปอร์เซ็นไทล์ {fmt(pct, 0)}",
             f"ไทยมีค่าสูงกว่าประมาณร้อยละ {fmt(pct, 0)} ของประเทศทั้งหมด" if pd.notna(pct) else "ไม่มีข้อมูลของไทยในปีนี้")

    # --- ความสัมพันธ์ ---
    with tabs[3]:
        year = st.select_slider("ปีที่วิเคราะห์ (ข้อมูล GDP มีถึงปี 2018)", options=complete_years, value=2018, key="an_corr_year")
        cvars = ["per_capita_electricity", "energy_per_capita", "renewables_share_elec", "gas_share_elec", "coal_share_elec",
                 "hydro_share_elec", "carbon_intensity_elec"]
        if year <= 2018:
            cvars.insert(2, "gdp_per_capita")
        base = df[(df["year"] == year) & ~df["low_precision"]][cvars]
        corr = base.corr(method="spearman")
        short = {c: LABELS[c].split(" (")[0] for c in cvars}
        with panel():
            panel_title(f"ค่าสหสัมพันธ์ของสเปียร์แมน ปี {year}", f"{len(base)} ประเทศ · สีน้ำเงิน = ทิศทางเดียวกัน · สีส้ม = ทิศทางตรงข้าม")
            lab = [short[c] for c in cvars]
            fig = go.Figure(go.Heatmap(z=corr.values, x=lab, y=lab, zmin=-1, zmax=1,
                                       colorscale=[[0, "#C0602A"], [0.5, "#F7F9FA"], [1, BLUE]],
                                       text=np.round(corr.values, 2), texttemplate="%{text}", textfont=dict(size=12),
                                       hovertemplate="%{y} กับ %{x}: %{z:.2f}<extra></extra>", colorbar=dict(thickness=12)))
            fig = style_fig(fig, 520, None, None, legend=False)
            fig.update_xaxes(tickangle=-35)
            fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
            show(fig)
            pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack().sort_values(key=abs, ascending=False)
            items = " · ".join(f"{short[i]} กับ{short[j]} ({v:.2f})" for (i, j), v in pairs.head(3).items())
            note(f"คู่ตัวแปรที่สัมพันธ์กันมากที่สุด: {items}",
                 "ใช้วิธีสเปียร์แมนเนื่องจากข้อมูลมีความเบ้ วิธีนี้พิจารณาจากอันดับ จึงไม่ถูกค่าสุดโต่งดึง",
                 "ค่าสหสัมพันธ์ไม่ได้แสดงความเป็นเหตุเป็นผล ควรใช้ประกอบกับข้อมูลด้านอื่น")

    # --- คุณภาพข้อมูล ---
    with tabs[4]:
        q = load_side_table("data_quality_summary.csv")
        left, right = st.columns([1, 1.45])
        with left, panel():
            panel_title("คุณภาพข้อมูลก่อนและหลังการทำความสะอาด")
            if q is not None:
                q = q.rename(columns={q.columns[0]: "ตัวชี้วัด"})
                q["ตัวชี้วัด"] = q["ตัวชี้วัด"].replace(QUALITY_TH)
                st.dataframe(q, hide_index=True, width="stretch")
            note("ค่าว่างของตัวแปรสำคัญลดลงจากประมาณร้อยละ 69 เหลือต่ำกว่าร้อยละ 2",
                 "ข้อมูลผ่านการตัดกลุ่มภูมิภาค แก้ค่า 0 ที่ไม่ถูกต้อง และตรวจความสอดคล้องแล้ว")
        with right, panel():
            panel_title(f"การตรวจหาค่าผิดปกติ ปี {latest}", "จำนวนประเทศที่พบค่าผิดปกติ เปรียบเทียบวิธี IQR และ Z-score (|z| > 3)")
            y = df[df["year"] == latest]
            rows = []
            for c in ["per_capita_electricity", "electricity_generation", "renewables_share_elec", "gas_share_elec",
                      "coal_share_elec", "carbon_intensity_elec", "net_elec_imports"]:
                s = y[c].dropna()
                q1, q3 = s.quantile([.25, .75]); i = q3 - q1
                m = (s < q1 - 1.5 * i) | (s > q3 + 1.5 * i)
                rows.append({"ตัวแปร": LABELS[c].split(" (")[0], "IQR": int(m.sum()),
                             "Z-score": int((((s - s.mean()) / s.std()).abs() > 3).sum()),
                             "ตัวอย่างประเทศ": ", ".join(y.loc[s[m].sort_values(ascending=False).index, "country"].astype(str).head(3))})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            note("วิธี Z-score พบค่าผิดปกติน้อยกว่า IQR ในตัวแปรที่เบ้มาก และตัวแปรที่ส่วนใหญ่มีค่าใกล้ศูนย์ (เช่น สัดส่วนถ่านหิน) "
                 "ถูกตีความว่าผิดปกติจำนวนมาก",
                 "ค่าผิดปกติส่วนใหญ่เป็นค่าจริง (เช่น ไอซ์แลนด์มีอุตสาหกรรมที่ใช้ไฟฟ้าสูง) จึงเก็บไว้และใช้มัธยฐานแทนการลบทิ้ง")
        with panel():
            cols = ["gdp", "electricity_generation", "electricity_demand", "carbon_intensity_elec", "renewables_share_elec", "population"]
            by_year = df.groupby("year")[cols].apply(lambda g: g.isna().mean() * 100)
            panel_title("สัดส่วนค่าว่างแยกตามปี (%)", "สีเข้มแสดงค่าว่างมาก")
            fig = go.Figure(go.Heatmap(z=by_year.T.values, x=by_year.index, y=[LABELS[c].split(" (")[0] for c in cols],
                                       zmin=0, zmax=100, colorscale=[[0, "#F7F9FA"], [1, NAVY]],
                                       hovertemplate="ปี %{x} · %{y}: ว่าง %{z:.1f}%<extra></extra>", colorbar=dict(thickness=12)))
            fig = style_fig(fig, 280, None, None, legend=False)
            fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
            show(fig)
            note("ค่าว่างกระจุกตัวใน 2 ส่วน ได้แก่ GDP หลังปี 2018 และข้อมูลปี 2022",
                 "ค่าว่างเกิดจากรอบการเก็บข้อมูลของแหล่งข้อมูล ไม่ใช่การสุ่ม จึงไม่เติมค่าแทน และใช้ภาพรวมโลกถึงปี 2021")
    source()


# ---------------------------------------------------------------------------
# 12) หน้าตารางข้อมูล
# ---------------------------------------------------------------------------
def page_explorer():
    df, _, _ = DATA
    breadcrumb("ตารางข้อมูล")
    st.title("ตารางข้อมูล")
    groups = {
        "ตัวชี้วัดหลัก": ["electricity_generation", "electricity_demand", "per_capita_electricity", "carbon_intensity_elec"],
        "สัดส่วนเชื้อเพลิง (%)": list(FUEL_SHARE.values()) + ["fossil_share_elec", "renewables_share_elec"],
        "ปริมาณไฟฟ้าตามเชื้อเพลิง (TWh)": list(FUEL_TWH.values()),
        "ข้อมูลประกอบ": ["population", "gdp", "primary_energy_consumption"],
        "สถานะคุณภาพข้อมูล": ["year_complete", "is_outlier_pc_elec", "low_precision", "gdp_available"],
    }
    with panel():
        f1, f2, f3 = st.columns(3)
        regions = f1.multiselect("ทวีป", sorted(df["region_th"].dropna().unique()), key="ex_region", placeholder="ทุกทวีป")
        search = f2.text_input("ค้นหาชื่อประเทศ (ภาษาอังกฤษ)", key="ex_search", placeholder="เช่น Thai, Viet")
        yr = f3.slider("ช่วงปี", 2000, 2022, (2015, 2022), key="ex_years")
        g1, g2, g3 = st.columns([2, 1, 1])
        picked = g1.multiselect("กลุ่มข้อมูล", list(groups), default=["ตัวชี้วัดหลัก"], key="ex_groups")
        asean_only = g2.checkbox("เฉพาะอาเซียน", key="ex_asean")
        complete_only = g3.checkbox("เฉพาะปีที่ข้อมูลครบถ้วน", key="ex_complete")

    d = df[df["year"].between(*yr)]
    if regions:
        d = d[d["region_th"].isin(regions)]
    if search.strip():
        d = d[d["country"].astype(str).str.contains(search.strip(), case=False, regex=False)]
    if asean_only:
        d = d[d["is_asean"]]
    if complete_only:
        d = d[d["year_complete"]]
    cols = ["country", "region_th", "year"] + [c for g in picked for c in groups[g]]
    if d.empty:
        empty_state("ไม่พบข้อมูลตามเงื่อนไขที่เลือก กรุณาตรวจสอบคำค้นหา (ใช้ชื่อภาษาอังกฤษ) หรือกดล้างตัวกรอง",
                    ["ex_region", "ex_search", "ex_years", "ex_asean", "ex_complete"])
        return
    s1, s2 = st.columns([1, 2])
    sort_col = s1.selectbox("เรียงตาม", cols, index=2, key="ex_sort", format_func=lambda c: LABELS.get(c, c))
    d = d[cols].sort_values([sort_col, "country"], ascending=[sort_col in ("country", "region_th", "year"), True])
    out = d.rename(columns=LABELS)
    s2.markdown(f"<div style='padding-top:2rem;color:{MUTED}'>พบ {len(d):,} แถว จาก {d['country'].nunique()} ประเทศ · "
                f"คลิกหัวคอลัมน์เพื่อเรียงลำดับได้</div>", unsafe_allow_html=True)
    st.dataframe(out, hide_index=True, width="stretch", height=440)

    b1, b2, _ = st.columns([1, 1, 3])
    b1.download_button("ดาวน์โหลด CSV", out.to_csv(index=False).encode("utf-8-sig"), "energy_filtered.csv", "text/csv", width="stretch")
    xls = to_excel_bytes(out)
    if xls:
        b2.download_button("ดาวน์โหลด Excel", xls, "energy_filtered.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
    dic = load_side_table("data_dictionary.csv")
    if dic is not None:
        with st.expander("คำอธิบายข้อมูลแต่ละคอลัมน์ (Data Dictionary)"):
            show_d = dic[dic["column"].isin(cols)][["column", "description_th", "unit_or_domain", "missing_pct"]].copy()
            show_d.insert(0, "ชื่อที่แสดง", show_d["column"].map(lambda c: LABELS.get(c, c)))
            st.dataframe(show_d.rename(columns={"column": "ชื่อคอลัมน์ต้นฉบับ", "description_th": "ความหมาย",
                                                "unit_or_domain": "หน่วย", "missing_pct": "ค่าว่าง (%)"}),
                         hide_index=True, width="stretch")
    source()


# ---------------------------------------------------------------------------
# 13) หน้าคู่มือและแหล่งข้อมูล
# ---------------------------------------------------------------------------
def page_about():
    df, _, info = DATA
    breadcrumb("คู่มือและแหล่งข้อมูล")
    st.title("คู่มือการใช้งานและแหล่งข้อมูล")
    left, right = st.columns(2)
    with left, panel():
        panel_title("ขั้นตอนการใช้งาน")
        st.markdown("""
1. **ภาพรวม** แสดงกราฟสำคัญทั้งหมด เลือกปีและประเทศที่ต้องการวิเคราะห์ได้ที่ด้านบน
2. **รายประเทศ** เลือกประเทศและช่วงปี เพื่อดูแนวโน้มและการปล่อย CO₂
3. **เปรียบเทียบ** เลือกชุดประเทศ (อาเซียน ทวีป หรือ 10 อันดับแรก) และตัวชี้วัด ระบบจะจัดอันดับให้
4. **ผลการวิเคราะห์** สถิติเชิงพรรณนา การกระจาย ความสัมพันธ์ และข้อสรุปของการศึกษา
5. **ตารางข้อมูล** กรอง ค้นหา และส่งออกเป็น CSV หรือ Excel
""")
    with right, panel():
        panel_title("ข้อควรทราบ")
        st.markdown("""
- ข้อมูล GDP สิ้นสุดที่ปี 2018
- ภาพรวมโลกใช้เฉพาะปีที่ข้อมูลครบถ้วน (ถึงปี 2021)
- ไฟฟ้าต่อประชากรมีความเบ้สูง จึงใช้ **มัธยฐาน** เป็นค่ากลาง
- ประเทศที่ผลิตไฟฟ้าน้อยกว่า 0.1 TWh มีข้อมูลความละเอียดต่ำ
- ชื่อประเทศใช้ภาษาอังกฤษตามแหล่งข้อมูล เพื่อความถูกต้องในการค้นหาและอ้างอิง
""")
    with panel():
        panel_title("แหล่งข้อมูล",
                    f"Our World in Data (Energy) · {info['rows']:,} แถว · {df['country'].nunique()} ประเทศ · ปี {df['year'].min()}–{df['year'].max()} "
                    f"· ปีล่าสุดที่ข้อมูลครบถ้วน {int(df.loc[df['year_complete'], 'year'].max())}")
        st.markdown("ข้อมูลผ่านการทำความสะอาด 5 ขั้น และการสำรวจข้อมูล 4 ขั้น ด้วย Python (pandas) บน Jupyter Notebook "
                    "รายละเอียดคุณภาพข้อมูลดูได้ที่หน้า **ผลการวิเคราะห์ › คุณภาพข้อมูล**")
    with panel():
        panel_title("ประสิทธิภาพของระบบ", "วัดจากเซิร์ฟเวอร์จริงทุกครั้งที่เปิดหน้านี้")
        t0 = time.perf_counter()
        _ = df[df["country"] == "Thailand"]
        filter_ms = (time.perf_counter() - t0) * 1000
        m = st.columns(3)
        kpi(m[0], "เวลาโหลดข้อมูลครั้งแรก", f"{info['load_ms']:.0f}", "ms")
        kpi(m[1], "เวลาคำนวณภาพรวมล่วงหน้า", f"{info['agg_ms']:.0f}", "ms")
        kpi(m[2], "เวลากรองข้อมูล 1 ประเทศ", f"{filter_ms:.1f}", "ms")
        st.caption("แนวทางเพิ่มความเร็ว: แคชข้อมูลด้วย st.cache_data · คำนวณภาพรวมล่วงหน้า · "
                   "ใช้ชนิดข้อมูล category · ไฟล์ข้อมูลขนาดประมาณ 1 MB")
    source()


# ---------------------------------------------------------------------------
# 14) เริ่มแอป
# ---------------------------------------------------------------------------
if "user" not in st.session_state:
    login_page()
    st.stop()

try:
    with st.spinner("กำลังโหลดข้อมูล..."):
        DATA = load_data()
except FileNotFoundError:
    st.error("ไม่พบไฟล์ข้อมูล energy_clean.csv กรุณาวางไฟล์ไว้ในโฟลเดอร์ data หรือโฟลเดอร์เดียวกับ app.py แล้วรีเฟรชหน้า")
    st.stop()
except Exception as e:
    st.error(f"โหลดข้อมูลไม่สำเร็จ ({e}) กรุณารีเฟรชหน้า หากยังไม่สำเร็จ ให้รันขั้นทำความสะอาดที่ 5 ใหม่")
    st.stop()

PAGES = {
    "home": st.Page(page_home, title="ภาพรวม", icon=":material/dashboard:", url_path="home", default=True),
    "country": st.Page(page_country, title="รายประเทศ", icon=":material/public:", url_path="country"),
    "compare": st.Page(page_compare, title="เปรียบเทียบ", icon=":material/bar_chart:", url_path="compare"),
    "analysis": st.Page(page_analysis, title="ผลการวิเคราะห์", icon=":material/analytics:", url_path="analysis"),
    "explorer": st.Page(page_explorer, title="ตารางข้อมูล", icon=":material/table_view:", url_path="data"),
    "about": st.Page(page_about, title="คู่มือและแหล่งข้อมูล", icon=":material/menu_book:", url_path="about"),
}
with st.sidebar:
    st.markdown('<div class="brand">ไฟฟ้าไทยและอาเซียน</div><div class="brand-sub">ระบบนำเสนอข้อมูลพลังงานไฟฟ้า</div>',
                unsafe_allow_html=True)
    st.caption(f"ผู้ใช้: {st.session_state['user']}")
    st.button("ออกจากระบบ", on_click=lambda: st.session_state.pop("user", None), width="stretch")
nav = st.navigation(list(PAGES.values()))
nav.run()

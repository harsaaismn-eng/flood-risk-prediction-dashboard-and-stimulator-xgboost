
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from xgboost_rainfall import forecast_next_12_months, evaluate_model

# --------------------------- PAGE CONFIG ---------------------------
st.set_page_config(
    page_title="Flood Intelligence | Risk Monitor",
    page_icon="🌊",
    layout="wide"
)

APP_DIR = Path(__file__).resolve().parent
CSV_FILE = APP_DIR / "flood_data.csv"

RISK_COLORS = {
    "Low": "#34d399",
    "Moderate": "#facc15",
    "High": "#fb923c",
    "Severe": "#fb7185"
}

PLOT_BG = "rgba(0,0,0,0)"


# --------------------------- PREMIUM UI ---------------------------
st.markdown("""
<style>
:root {
    --cyan: #48e1f5;
    --line: rgba(111,220,245,.18);
    --muted: #9dbbd0;
}

.stApp {
    background:
        radial-gradient(
            ellipse at 8% 0%,
            rgba(8,145,178,.20),
            transparent 32%
        ),
        radial-gradient(
            ellipse at 100% 24%,
            rgba(14,116,144,.12),
            transparent 28%
        ),
        linear-gradient(
            145deg,
            #06111f,
            #0a1d31 54%,
            #071526
        );
    color: #eaf7ff;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.6rem;
    padding-bottom: 2rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #06111f,
        #0a243a
    );
    border-right: 1px solid var(--line);
}

section[data-testid="stSidebar"] * {
    color: #eaf7ff;
}

section[data-testid="stSidebar"]
div[role="radiogroup"] label {
    background: rgba(18,48,70,.45);
    border: 1px solid transparent;
    border-radius: 11px;
    padding: 9px 12px;
    margin: 4px 0;
    transition: all .2s;
}

section[data-testid="stSidebar"]
div[role="radiogroup"] label:hover {
    background: rgba(34,211,238,.1);
    border-color: var(--line);
}

/* Typography */
h1, h2, h3, h4 {
    color: #eefaff !important;
    letter-spacing: -.45px;
}

p, label {
    color: #c7dce9;
}

/* Hero banner */
.hero {
    position: relative;
    overflow: hidden;
    background:
        radial-gradient(
            circle at 88% 8%,
            rgba(34,211,238,.22),
            transparent 31%
        ),
        linear-gradient(
            125deg,
            #0d3a53,
            #08192b 72%
        );
    padding: 34px 36px;
    border-radius: 24px;
    margin-bottom: 26px;
    border: 1px solid rgba(103,232,249,.23);
    box-shadow: 0 18px 55px rgba(0,0,0,.22);
}

.hero-title {
    font-size: clamp(29px,3.8vw,43px);
    font-weight: 800;
    line-height: 1.14;
    letter-spacing: -1.1px;
    color: #f3fbff;
    margin-top: 4px;
}

.hero-subtitle {
    font-size: 16px;
    color: #b5d8e9;
    margin-top: 11px;
    line-height: 1.6;
}

.hero-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 9px;
    margin-top: 21px;
}

.hero-pill {
    display: inline-block;
    padding: 7px 12px;
    border: 1px solid rgba(103,232,249,.23);
    border-radius: 999px;
    background: rgba(34,211,238,.08);
    color: #c7f5ff;
    font-size: 12px;
}

/* Metric cards */
div[data-testid="stMetric"],
div[data-testid="metric-container"] {
    background: linear-gradient(
        145deg,
        rgba(15,47,70,.96),
        rgba(8,28,46,.96)
    );
    border: 1px solid var(--line);
    border-radius: 17px;
    padding: 17px 19px;
    box-shadow: 0 10px 28px rgba(0,0,0,.16);
    transition: transform .18s, border-color .18s;
}

div[data-testid="stMetric"]:hover,
div[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    border-color: rgba(103,232,249,.42);
}

div[data-testid="stMetricLabel"],
div[data-testid="metric-container"] label {
    color: #a9c8da !important;
}

div[data-testid="stMetricValue"] {
    color: var(--cyan) !important;
    font-weight: 750;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(
        135deg,
        #0891b2,
        #075985
    );
    color: #fff !important;
    border: 1px solid rgba(103,232,249,.3);
    border-radius: 11px;
    min-height: 43px;
    font-weight: 650;
    transition: all .18s;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: linear-gradient(
        135deg,
        #06b6d4,
        #0284c7
    );
    border-color: #67e8f9;
    transform: translateY(-1px);
}

/* Inputs */
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stNumberInput input,
.stTextInput input,
.stTextArea textarea {
    background: rgba(7,25,42,.88);
    border-radius: 10px;
}

input, textarea {
    color: #eaf7ff !important;
}

/* Panels and tables */
div[data-testid="stAlert"] {
    border-radius: 13px;
}

div[data-testid="stExpander"] {
    background: rgba(10,34,53,.65);
    border: 1px solid var(--line);
    border-radius: 13px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 13px;
    overflow: hidden;
}

hr {
    border-color: var(--line);
}

/* Charts */
div[data-testid="stPlotlyChart"] {
    background: rgba(9,29,47,.5);
    border: 1px solid rgba(117,221,245,.1);
    border-radius: 16px;
    padding: 7px;
}

/* Footer */
.footer {
    text-align: center;
    padding: 25px 12px 10px;
    margin-top: 28px;
    color: #8eb2c7;
    border-top: 1px solid var(--line);
    font-size: 13px;
    line-height: 1.8;
}

.footer strong {
    color: #bceefa;
}

@media(max-width:768px) {
    .block-container {
        padding-top: 1rem;
    }

    .hero {
        padding: 24px 22px;
        border-radius: 18px;
    }

    .hero-subtitle {
        font-size: 14px;
    }
}
</style>
""", unsafe_allow_html=True)


# --------------------------- DATA ---------------------------
@st.cache_data
def load_data():
    if not CSV_FILE.exists():
        return None

    try:
        return pd.read_csv(CSV_FILE)
    except Exception:
        return None


df = load_data()


# --------------------------- RISK CALCULATION ---------------------------
def calculate_risk(
    rainfall,
    river_level,
    soil_saturation,
    drainage_capacity
):
    rainfall_score = min(
        max(float(rainfall), 0) / 300 * 100,
        100
    )

    river_score = min(
        max(float(river_level), 0) / 10 * 100,
        100
    )

    soil_score = min(
        max(float(soil_saturation), 0),
        100
    )

    drainage_risk = 100 - min(
        max(float(drainage_capacity), 0),
        100
    )

    score = round(
        0.35 * rainfall_score
        + 0.30 * river_score
        + 0.20 * soil_score
        + 0.15 * drainage_risk
    )

    if score <= 30:
        level = "Low"
    elif score <= 60:
        level = "Moderate"
    elif score <= 80:
        level = "High"
    else:
        level = "Severe"

    return {
        "score": score,
        "level": level,
        "breakdown": {
            "Rainfall pressure": round(rainfall_score, 1),
            "River level": round(river_score, 1),
            "Soil saturation": round(soil_score, 1),
            "Drainage risk": round(drainage_risk, 1)
        }
    }


# --------------------------- GAUGE ---------------------------
def make_gauge(score, level):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={
                "suffix": "/100",
                "font": {
                    "size": 42,
                    "color": "#eaf7ff"
                }
            },
            title={
                "text": f"{level.upper()} RISK",
                "font": {
                    "size": 16,
                    "color": "#c4e8f5"
                }
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "#9bbbd0"
                },
                "bar": {
                    "color": RISK_COLORS[level],
                    "thickness": 0.28
                },
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {
                        "range": [0, 30],
                        "color": "rgba(52,211,153,.18)"
                    },
                    {
                        "range": [30, 60],
                        "color": "rgba(250,204,21,.18)"
                    },
                    {
                        "range": [60, 80],
                        "color": "rgba(251,146,60,.20)"
                    },
                    {
                        "range": [80, 100],
                        "color": "rgba(251,113,133,.22)"
                    }
                ]
            }
        )
    )

    fig.update_layout(
        paper_bgcolor=PLOT_BG,
        font_color="#eaf7ff",
        height=285,
        margin=dict(l=25, r=25, t=55, b=15)
    )

    return fig


# --------------------------- CHART STYLING ---------------------------
def style_chart(fig, height=None):
    fig.update_layout(
        paper_bgcolor=PLOT_BG,
        plot_bgcolor="rgba(7,25,42,.28)",
        font_color="#d8edf7",
        legend_title_text="",
        margin=dict(l=15, r=15, t=55, b=15),
        hoverlabel=dict(
            bgcolor="#0b263b",
            font_color="#effaff"
        )
    )

    fig.update_xaxes(
        gridcolor="rgba(150,200,220,.10)",
        zerolinecolor="rgba(150,200,220,.10)"
    )

    fig.update_yaxes(
        gridcolor="rgba(150,200,220,.10)",
        zerolinecolor="rgba(150,200,220,.10)"
    )

    if height:
        fig.update_layout(height=height)

    return fig


# --------------------------- SIMULATION LOGIC ---------------------------
def simulate_hours(
    rainfall,
    river_level,
    soil_saturation,
    drainage_capacity
):
    rows = []

    for hour in range(13):
        rain = min(
            300,
            rainfall * (1 + hour * 0.025)
        )

        river = min(
            10,
            river_level * (1 + hour * 0.015)
        )

        soil = min(
            100,
            soil_saturation + hour * 1.5
        )

        result = calculate_risk(
            rain,
            river,
            soil,
            drainage_capacity
        )

        rows.append({
            "Hour": hour,
            "Rainfall (mm)": round(rain, 1),
            "River Level (m)": round(river, 2),
            "Soil Saturation (%)": round(soil, 1),
            "Risk Score": result["score"],
            "Risk Level": result["level"]
        })

    return pd.DataFrame(rows)


# --------------------------- SIDEBAR ---------------------------
st.sidebar.markdown("""
<div style="text-align:center;padding:12px 4px 18px">
  <div style="font-size:48px">🌊</div>
  <div style="font-size:21px;font-weight:800;letter-spacing:1px;color:#effaff">
      FLOOD INTELLIGENCE
  </div>
  <div style="font-size:11px;letter-spacing:2px;color:#7fb8d1;margin-top:5px">
      RISK MONITORING SYSTEM
  </div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "WORKSPACE",
    [
        "🏠 Overview",
        "🌊 Flood Simulator",
        "🤖 Prediction Lab",
        "📊 Historical Data"
    ]
)

st.sidebar.divider()

if df is not None:
    st.sidebar.success(
        f"Dataset connected · {len(df):,} rows"
    )
else:
    st.sidebar.error("Dataset not found")

st.sidebar.caption(f"Source: {CSV_FILE.name}")
st.sidebar.markdown("---")
st.sidebar.caption(
    "Prototype for educational analysis. "
    "Not an official warning service."
)


# --------------------------- HERO BANNER ---------------------------
st.markdown("""
<div class="hero">
  <div style="font-size:42px;margin-bottom:12px">
      🌊 🌧️ 💧
  </div>

  <div class="hero-title">
      Flood Risk Predictor &amp; Simulator
  </div>

  <div class="hero-subtitle">
      An interactive environmental analytics workspace for rainfall
      forecasting, flood-risk simulation, and historical data exploration.
  </div>

  <div class="hero-pills">
    <span class="hero-pill">🌧️ Rainfall Analytics</span>
    <span class="hero-pill">🌊 Risk Simulation</span>
    <span class="hero-pill">🤖 XGBoost Forecasting</span>
    <span class="hero-pill">📊 Historical Insights</span>
  </div>
</div>
""", unsafe_allow_html=True)


# --------------------------- OVERVIEW ---------------------------
if page == "🏠 Overview":
    st.markdown("## Command centre")

    st.caption(
        "A quick snapshot of the selected baseline conditions "
        "and the rule-based risk estimate."
    )

    if "rainfall" not in st.session_state:
        st.session_state.rainfall = 100
        st.session_state.river_level = 3.0
        st.session_state.soil_saturation = 50
        st.session_state.drainage_capacity = 70

    result = calculate_risk(
        st.session_state.rainfall,
        st.session_state.river_level,
        st.session_state.soil_saturation,
        st.session_state.drainage_capacity
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("FLOOD RISK SCORE", f"{result['score']}/100")
    m2.metric(
        "RISK CATEGORY",
        result["level"],
        delta="Rule-based estimate"
    )
    m3.metric(
        "RAINFALL INPUT",
        f"{st.session_state.rainfall} mm"
    )
    m4.metric(
        "RIVER LEVEL",
        f"{st.session_state.river_level:.1f} m"
    )

    left, right = st.columns([1.05, 1])

    with left:
        st.markdown("### Risk indicator")
        st.plotly_chart(
            make_gauge(result["score"], result["level"]),
            use_container_width=True
        )

    with right:
        st.markdown("### Contributing factors")

        factors = pd.DataFrame({
            "Factor": list(result["breakdown"].keys()),
            "Score": list(result["breakdown"].values())
        })

        fig = px.bar(
            factors,
            x="Score",
            y="Factor",
            orientation="h",
            color="Factor",
            range_x=[0, 100],
            text="Score",
            title="Normalized factor values"
        )

        fig.update_traces(
            textposition="outside",
            cliponaxis=False
        )

        st.plotly_chart(
            style_chart(fig, 285),
            use_container_width=True
        )

    st.info(
        "The score combines normalized rainfall (35%), river level "
        "(30%), soil saturation (20%), and drainage risk (15%). "
        "It is a demonstration formula, not a calibrated hydrological model."
    )

    st.markdown("### Explore the workspace")

    a, b, c = st.columns(3)

    a.markdown(
        "**🌊 Flood Simulator**\n\n"
        "Adjust environmental inputs, compare scenarios, "
        "and view a 12-hour simulated trend."
    )

    b.markdown(
        "**🤖 Prediction Lab**\n\n"
        "Explore monthly rainfall forecasts generated by the XGBoost model."
    )

    c.markdown(
        "**📊 Historical Data**\n\n"
        "Inspect source records, missing values, trends, and correlations."
    )


# --------------------------- FLOOD SIMULATOR ---------------------------
elif page == "🌊 Flood Simulator":
    st.markdown("## Interactive flood simulator")

    st.caption(
        "Change the inputs to explore how the demonstration risk score responds."
    )

    presets = {
        "🌤️ Normal": {
            "rainfall": 50,
            "river_level": 2.0,
            "soil_saturation": 30,
            "drainage_capacity": 80
        },
        "🌧️ Heavy rain": {
            "rainfall": 150,
            "river_level": 4.0,
            "soil_saturation": 70,
            "drainage_capacity": 50
        },
        "🌊 Extreme scenario": {
            "rainfall": 250,
            "river_level": 7.0,
            "soil_saturation": 90,
            "drainage_capacity": 30
        }
    }

    preset_cols = st.columns(3)

    for col, (name, values) in zip(preset_cols, presets.items()):
        if col.button(name, use_container_width=True):
            for key, value in values.items():
                st.session_state[key] = value
            st.rerun()

    defaults = [
        ("rainfall", 100),
        ("river_level", 3.0),
        ("soil_saturation", 50),
        ("drainage_capacity", 70)
    ]

    for key, default in defaults:
        if key not in st.session_state:
            st.session_state[key] = default

    controls, visuals = st.columns([0.9, 1.1], gap="large")

    with controls:
        st.markdown("### Environmental inputs")

        rainfall = st.slider(
            "🌧️ Rainfall (mm)",
            0, 300,
            key="rainfall"
        )

        river = st.slider(
            "💧 River level (m)",
            0.0, 10.0,
            step=0.1,
            key="river_level"
        )

        soil = st.slider(
            "🌱 Soil saturation (%)",
            0, 100,
            key="soil_saturation"
        )

        drainage = st.slider(
            "🚰 Drainage capacity (%)",
            0, 100,
            key="drainage_capacity"
        )

        if st.button("↺ Reset inputs", use_container_width=True):
            for key, value in [
                ("rainfall", 100),
                ("river_level", 3.0),
                ("soil_saturation", 50),
                ("drainage_capacity", 70)
            ]:
                st.session_state[key] = value

            st.rerun()

    result = calculate_risk(
        rainfall,
        river,
        soil,
        drainage
    )

    with visuals:
        st.markdown("### Estimated risk")

        st.plotly_chart(
            make_gauge(result["score"], result["level"]),
            use_container_width=True
        )

        if result["level"] == "Low":
            st.success(
                f"{result['level']} risk · {result['score']}/100"
            )
        elif result["level"] in ["Moderate", "High"]:
            st.warning(
                f"{result['level']} risk · {result['score']}/100"
            )
        else:
            st.error(
                f"{result['level']} risk · {result['score']}/100"
            )

    st.markdown("### Scenario comparison")

    if "baseline_scenario" not in st.session_state:
        st.session_state.baseline_scenario = None

    save_col, clear_col = st.columns(2)

    if save_col.button(
        "📌 Save current as baseline",
        use_container_width=True
    ):
        st.session_state.baseline_scenario = {
            "score": result["score"],
            "level": result["level"],
            "rainfall": rainfall,
            "river": river,
            "soil": soil,
            "drainage": drainage
        }
        st.success("Baseline scenario saved.")

    if clear_col.button(
        "✕ Clear baseline",
        use_container_width=True
    ):
        st.session_state.baseline_scenario = None

    baseline = st.session_state.baseline_scenario

    if baseline:
        x, y, z = st.columns(3)

        x.metric(
            "Baseline score",
            f"{baseline['score']}/100",
            baseline["level"]
        )

        y.metric(
            "Current score",
            f"{result['score']}/100",
            delta=result["score"] - baseline["score"]
        )

        z.markdown(
            f"**Baseline inputs**\n\n"
            f"Rainfall: {baseline['rainfall']} mm\n\n"
            f"River: {baseline['river']} m\n\n"
            f"Soil: {baseline['soil']}%\n\n"
            f"Drainage: {baseline['drainage']}%"
        )

    st.markdown("### 12-hour scenario projection")

    if st.button(
        "▶ Run simulation",
        type="primary",
        use_container_width=True
    ):
        st.session_state.hourly_simulation = simulate_hours(
            rainfall,
            river,
            soil,
            drainage
        )

    if "hourly_simulation" in st.session_state:
        sim = st.session_state.hourly_simulation

        line = px.line(
            sim,
            x="Hour",
            y="Risk Score",
            markers=True,
            range_y=[0, 100],
            title="Simulated risk score over time"
        )

        line.update_traces(
            line_color="#48e1f5",
            line_width=3,
            marker_size=7
        )

        st.plotly_chart(
            style_chart(line, 350),
            use_container_width=True
        )

        st.dataframe(
            sim,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇ Download simulation CSV",
            sim.to_csv(index=False).encode(),
            "flood_simulation.csv",
            "text/csv"
        )


# --------------------------- XGBOOST PREDICTION LAB ---------------------------
elif page == "🤖 Prediction Lab":
    st.markdown("## XGBoost rainfall prediction lab")

    st.caption(
        "Forecast monthly rainfall from historical patterns "
        "and inspect holdout evaluation metrics."
    )

    st.info(
        "Important: the available CSV contains historical monthly rainfall, "
        "not observed flood/no-flood labels. XGBoost forecasts rainfall; "
        "the flood-risk score is a separate rule-based simulator. "
        "This is not a live sensor feed or an official flood warning."
    )

    if df is None:
        st.error("Place flood_data.csv in the same folder as app.py.")
    else:
        try:
            forecast = forecast_next_12_months(str(CSV_FILE))
            metrics, comparison = evaluate_model(str(CSV_FILE))

            q1, q2, q3 = st.columns(3)

            q1.metric(
                "HOLDOUT MAE",
                f"{metrics['mae']:.1f} mm"
            )

            q2.metric(
                "HOLDOUT RMSE",
                f"{metrics['rmse']:.1f} mm"
            )

            q3.metric(
                "TEST MONTHS",
                str(metrics["test_months"])
            )

            st.caption(
                "MAE and RMSE measure rainfall prediction error in millimetres; "
                "they are not flood-classification accuracy."
            )

            chart = px.line(
                forecast,
                x="Month",
                y="Forecast Rainfall (mm)",
                markers=True,
                title="12-month rainfall forecast"
            )

            chart.update_traces(
                line_color="#48e1f5",
                line_width=3,
                marker_size=7
            )

            st.plotly_chart(
                style_chart(chart, 390),
                use_container_width=True
            )

            st.dataframe(
                forecast,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇ Download forecast CSV",
                forecast.to_csv(index=False).encode(),
                "xgboost_rainfall_forecast.csv",
                "text/csv"
            )

            with st.expander(
                "Compare predicted and actual holdout rainfall"
            ):
                comp_chart = go.Figure()

                comp_chart.add_trace(
                    go.Scatter(
                        x=comparison["Month"],
                        y=comparison["Actual Rainfall (mm)"],
                        mode="lines+markers",
                        name="Actual",
                        line=dict(color="#48e1f5", width=3)
                    )
                )

                comp_chart.add_trace(
                    go.Scatter(
                        x=comparison["Month"],
                        y=comparison["Predicted Rainfall (mm)"],
                        mode="lines+markers",
                        name="Predicted",
                        line=dict(
                            color="#fb923c",
                            width=3,
                            dash="dash"
                        )
                    )
                )

                st.plotly_chart(
                    style_chart(comp_chart, 330),
                    use_container_width=True
                )

                st.dataframe(
                    comparison,
                    use_container_width=True,
                    hide_index=True
                )

        except Exception as exc:
            st.error(f"Could not run the XGBoost model: {exc}")
            st.caption(
                "Check that requirements are installed and flood_data.csv "
                "contains Year and Jan–Dec columns."
            )


# --------------------------- HISTORICAL DATA ---------------------------
elif page == "📊 Historical Data":
    st.markdown("## Historical data explorer")

    if df is None:
        st.error(
            "flood_data.csv could not be found or read. "
            "Put it beside app.py."
        )
    else:
        numeric = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        missing = int(df.isna().sum().sum())

        d1, d2, d3, d4 = st.columns(4)

        d1.metric("RECORDS", f"{len(df):,}")
        d2.metric("COLUMNS", str(len(df.columns)))
        d3.metric("MISSING VALUES", str(missing))
        d4.metric("NUMERIC FIELDS", str(len(numeric)))

        st.markdown("### Dataset preview")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇ Download source CSV",
            df.to_csv(index=False).encode(),
            "flood_data_copy.csv",
            "text/csv"
        )

        st.markdown("### Explore a numeric field")

        if numeric:
            selected = st.selectbox(
                "Choose a column",
                numeric
            )

            vals = pd.to_numeric(
                df[selected],
                errors="coerce"
            ).dropna().reset_index(drop=True)

            trend = pd.DataFrame({
                "Record": np.arange(1, len(vals) + 1),
                selected: vals
            })

            fig = px.line(
                trend,
                x="Record",
                y=selected,
                title=f"{selected} by record"
            )

            fig.update_traces(
                line_color="#48e1f5",
                line_width=2
            )

            st.plotly_chart(
                style_chart(fig, 350),
                use_container_width=True
            )

        st.markdown("### Data quality by column")

        quality = pd.DataFrame({
            "Column": df.columns,
            "Data type": [str(x) for x in df.dtypes],
            "Missing values": [
                int(df[c].isna().sum())
                for c in df.columns
            ],
            "Missing %": [
                round(float(df[c].isna().mean() * 100), 2)
                for c in df.columns
            ]
        })

        st.dataframe(
            quality,
            use_container_width=True,
            hide_index=True
        )

        if len(numeric) >= 2:
            st.markdown("### Correlation map")

            corr = df[numeric].corr(numeric_only=True)

            fig = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale=[
                    [0, "#10283d"],
                    [0.5, "#0891b2"],
                    [1, "#a5f3fc"]
                ],
                zmin=-1,
                zmax=1,
                title="Numeric feature correlations"
            )

            st.plotly_chart(
                style_chart(fig, 430),
                use_container_width=True
            )


# --------------------------- FOOTER ---------------------------
st.markdown("""
<div class="footer">
    <strong>🌊 Flood Risk Predictor &amp; Simulator</strong><br>
    Rainfall Analysis &nbsp; • &nbsp;
    Flood Simulation &nbsp; • &nbsp;
    XGBoost Forecasting<br>
    <span style="font-size:11px;color:#7597ad">
         Developed as an academic project to explore rainfall patterns
        and understand flood risk under different conditions.
    </span>
</div>
""", unsafe_allow_html=True)

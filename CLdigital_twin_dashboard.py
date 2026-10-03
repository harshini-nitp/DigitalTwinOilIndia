"""
Well-to-Surface Digital Twin — CSS / SRP Control Room
=======================================================

This dashboard implements, end to end:
  1. CSS (Cyclic Steam Stimulation) cycle parameter optimization
  2. Reservoir heating / cooling / production performance prediction
  3. Continuous SRP optimization (stroke length + SPM) vs. well conditions
  4. Rod-floating detection and impact-loading minimization
  5. Pump efficiency and equipment reliability tracking
  6. Steam / energy consumption optimization and cost reduction
  7. A simulation-based performance panel comparing
   ML-optimized operation against a fixed-setting baseline

ARCHITECTURE NOTE:
  The simulation advances one simulated "day" per Streamlit script run and
  is stored in st.session_state, not in an infinite while-loop. After each
  day it sleeps briefly and calls st.rerun(). This is what lets the
  sidebar controls (oil price, risk tolerance, etc.) actually take effect
  — an infinite `while True:` loop would block Streamlit from ever
  re-reading widget values for the rest of the session.

NOTE ON DATA:
  The models below are trained on `srp_daily_operations.csv`, which is
  expected to contain (at minimum):
      reservoir_zone_temp_C, oil_viscosity_cP, stroke_length_in, spm,
      oil_rate_bbl_per_day, rod_floating_flag
  If that file isn't found, a synthetic dataset with the same schema is
  generated so the dashboard still runs end-to-end for demo/testing.
  The CSS-cycle economics (steam volume -> peak temperature -> SOR) use a
  simplified, clearly-labeled physics/heuristic proxy. In production this
  should be replaced by a model trained on your actual CSS cycle records
  and steam injection parameters (volume, rate, pressure) once available —
  the function signatures are written so that swap is a drop-in change.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# =====================================================================
# 1. PAGE SETUP
# =====================================================================
st.set_page_config(page_title="Digital Twin Control Room", layout="wide", initial_sidebar_state="expanded")

# =====================================================================
# 2. DATA LOADING (with synthetic fallback so the app always runs)
# =====================================================================
FEATURES = ['reservoir_zone_temp_C', 'oil_viscosity_cP', 'stroke_length_in', 'spm']


def generate_synthetic_operations_data(n=3000, seed=42):
    """Fallback dataset matching the expected schema, used only if the
    real srp_daily_operations.csv isn't found. Replace with real
    production history / VFD-SRP data / rod failure history for
    production use."""
    rng = np.random.default_rng(seed)
    temp = rng.uniform(50, 230, n)
    visc = 800 * np.exp(-0.012 * (temp - 140)) + rng.normal(0, 40, n)
    visc = np.clip(visc, 50, 4000)
    stroke = rng.choice([80, 100, 120], n)
    spm = rng.uniform(2.0, 8.0, n)

    base_rate = 15 + 0.35 * (230 - visc / 15) + 0.9 * spm + 0.05 * stroke
    overspeed_penalty = np.where(spm > (9 - visc / 700), (spm - (9 - visc / 700)) * 8, 0)
    oil_rate = np.clip(base_rate - overspeed_penalty + rng.normal(0, 4, n), 0, None)

    floating_logit = -6 + 0.9 * spm + 0.004 * visc - 0.01 * stroke
    floating_prob = 1 / (1 + np.exp(-floating_logit))
    rod_floating_flag = (rng.uniform(0, 1, n) < floating_prob).astype(int)

    return pd.DataFrame({
        'reservoir_zone_temp_C': temp,
        'oil_viscosity_cP': visc,
        'stroke_length_in': stroke,
        'spm': spm,
        'oil_rate_bbl_per_day': oil_rate,
        'rod_floating_flag': rod_floating_flag,
    })


@st.cache_resource
def load_and_train_models():
    try:
        daily_ops = pd.read_csv("srp_daily_operations.csv").dropna()
        data_source = "srp_daily_operations.csv"
    except FileNotFoundError:
        daily_ops = generate_synthetic_operations_data()
        data_source = "Physics-informed synthetic data (srp_daily_operations.csv not found)"

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.metrics import precision_score, recall_score, f1_score

    X = daily_ops[FEATURES]
    y_oil = daily_ops["oil_rate_bbl_per_day"]
    y_rod = daily_ops["rod_floating_flag"]

    # Same 80/20 split for both models
    X_train, X_test, y_oil_train, y_oil_test, y_rod_train, y_rod_test = train_test_split(
        X,
        y_oil,
        y_rod,
        test_size=0.20,
        random_state=42
    )

    # Oil production model
    pump_ml = RandomForestRegressor(
        n_estimators=150,
        random_state=42
    )
    pump_ml.fit(X_train, y_oil_train)

    oil_test_pred = pump_ml.predict(X_test)

    oil_mae = mean_absolute_error(y_oil_test, oil_test_pred)
    oil_r2 = r2_score(y_oil_test, oil_test_pred)

    # Rod floating risk model
    health_ml = RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced",
        random_state=42
    )
    health_ml.fit(X_train, y_rod_train)

    rod_test_pred = health_ml.predict(X_test)

    rod_precision = precision_score(
        y_rod_test,
        rod_test_pred,
        zero_division=0
    )

    rod_recall = recall_score(
        y_rod_test,
        rod_test_pred,
        zero_division=0
    )

    rod_f1 = f1_score(
        y_rod_test,
        rod_test_pred,
        zero_division=0
    )

    validation_metrics = {
        "oil_mae": oil_mae,
        "oil_r2": oil_r2,
        "rod_precision": rod_precision,
        "rod_recall": rod_recall,
        "rod_f1": rod_f1,
    }

    return pump_ml, health_ml, data_source, validation_metrics


pump_ml, health_ml, DATA_SOURCE, VALIDATION_METRICS = load_and_train_models()

def predict_oil_rate(temp, visc, stroke, spm):
    df = pd.DataFrame([[temp, visc, stroke, spm]], columns=FEATURES)
    return pump_ml.predict(df)[0]


def predict_rod_floating_risk(temp, visc, stroke, spm):
    df = pd.DataFrame([[temp, visc, stroke, spm]], columns=FEATURES)
    return health_ml.predict_proba(df)[0][1]


# =====================================================================
# 3. PHYSICS-INFORMED RESERVOIR HEATING / COOLING / VISCOSITY MODEL
# =====================================================================
AMBIENT_RESERVOIR_TEMP = 40.0
COOLING_TIME_CONSTANT = 35.0
STEAM_TO_TEMP_FACTOR = 0.012
MAX_PEAK_TEMP = 235.0
STEAM_THRESHOLD_TEMP = 60.0
STEAM_INJECTION_RATE_BBL_PER_DAY = 3000  # assumed field injection rate, used to size injection duration


def viscosity_from_temp(temp_c):
    return float(np.clip(800 * np.exp(-0.012 * (temp_c - 140)), 50, 6000))


def cool_one_day(temp_c):
    return AMBIENT_RESERVOIR_TEMP + (temp_c - AMBIENT_RESERVOIR_TEMP) * np.exp(-1 / COOLING_TIME_CONSTANT)


def days_until_threshold(temp_c, threshold=STEAM_THRESHOLD_TEMP):
    """t = TAU * ln((temp0 - ambient) / (threshold - ambient))"""
    if temp_c <= threshold:
        return 0
    num = temp_c - AMBIENT_RESERVOIR_TEMP
    den = threshold - AMBIENT_RESERVOIR_TEMP
    return int(max(0, round(COOLING_TIME_CONSTANT * np.log(num / den))))


def estimate_injection_duration_days(steam_volume_bbl):
    """Shared by both the ML-optimized and baseline wells so the two never
    drift apart due to a duplicated formula."""
    return max(1, int(round(steam_volume_bbl / STEAM_INJECTION_RATE_BBL_PER_DAY)))


def simulate_cycle_production(peak_temp, stroke, spm, max_days=150):
    """Roll a production cycle forward from peak_temp down to the
    re-injection threshold, using the trained pump_ml at each day's
    predicted temperature/viscosity and a FIXED representative SRP
    setting for the whole cycle (chosen by the SRP optimizer at peak
    conditions — see optimize_css_cycle). Returns (total_oil_bbl,
    cycle_length_days)."""
    temp = peak_temp
    total_oil = 0.0
    days = 0
    while temp > STEAM_THRESHOLD_TEMP and days < max_days:
        visc = viscosity_from_temp(temp)
        total_oil += predict_oil_rate(temp, visc, stroke, spm)
        temp = cool_one_day(temp)
        days += 1
    return total_oil, days


# =====================================================================
# 4. CONTINUOUS SRP OPTIMIZER
# =====================================================================
STROKE_CANDIDATES = [80, 100, 120]
SPM_CANDIDATES = np.arange(2.0, 8.0, 0.5)


def estimate_motor_load(visc, stroke, spm):
    return 5.0 + (visc / 300) + (spm * 0.5) + (stroke - 80) * 0.02


def estimate_pump_fillage(visc):
    return float(np.clip(95 - (visc / 50), 40, 98))


def get_optimal_srp_settings(temp, visc, risk_tolerance, oil_price, energy_cost_per_kwh):
    """Risk-constrained, cost-aware SRP setpoint optimizer: maximizes
    (oil revenue - energy cost) subject to rod-floating risk staying
    under the operator's tolerance. Falls back to the safest setting on
    the grid if nothing clears the risk bar."""
    best = None
    safest = None
    for stroke in STROKE_CANDIDATES:
        for spm in SPM_CANDIDATES:
            risk = predict_rod_floating_risk(temp, visc, stroke, spm)
            oil_pred = predict_oil_rate(temp, visc, stroke, spm)
            motor_load_kw = estimate_motor_load(visc, stroke, spm)
            energy_cost = motor_load_kw * 24 * energy_cost_per_kwh
            net_value = oil_pred * oil_price - energy_cost

            if safest is None or risk < safest["risk"]:
                safest = {"stroke": stroke, "spm": round(float(spm), 1), "oil_pred": oil_pred,
                          "risk": risk, "motor_load_kw": motor_load_kw, "net_value": net_value}

            if risk < risk_tolerance:
                if best is None or net_value > best["net_value"]:
                    best = {"stroke": stroke, "spm": round(float(spm), 1), "oil_pred": oil_pred,
                            "risk": risk, "motor_load_kw": motor_load_kw, "net_value": net_value}

    return best if best is not None else safest


# =====================================================================
# 5. CSS CYCLE OPTIMIZER
# =====================================================================
STEAM_VOLUME_CANDIDATES = np.arange(4000, 18000, 2000)   # bbl (CWE)
SOAK_DAYS_CANDIDATES = [3, 5, 7, 10]

FALLBACK_CSS_PLAN = {
    "steam_volume_bbl": 10000, "soak_days": 5, "peak_temp_C": 200.0,
    "projected_cycle_oil_bbl": 0, "projected_cycle_days": 60,
    "projected_SOR": float("nan"), "projected_net_value_usd": 0, "met_sor_target": False,
}


def optimize_css_cycle(current_temp, oil_price, steam_cost, risk_tolerance, energy_cost_per_kwh, max_sor=None):
    """Grid-search candidate (steam volume, soak time) pairs and pick the
    one that maximizes projected net cycle value (oil revenue - steam
    cost - soak opportunity cost), constrained to a maximum acceptable
    SOR when possible.

    For each candidate steam volume, the representative SRP setting used
    to project cycle oil is chosen by the SAME SRP optimizer that runs
    the well day-to-day (at peak-temperature conditions), so the CSS
    economics stay consistent with how the well will actually be run —
    rather than an arbitrary fixed stroke/SPM.

    Never returns None: if every candidate's SOR exceeds max_sor, falls
    back to the best candidate ignoring that constraint (flagged via
    "met_sor_target": False) instead of leaving the caller with nothing.
    """
    best = None
    best_unconstrained = None

    for steam_volume in STEAM_VOLUME_CANDIDATES:
        peak_temp = min(MAX_PEAK_TEMP, current_temp + steam_volume * STEAM_TO_TEMP_FACTOR)
        peak_visc = viscosity_from_temp(peak_temp)
        rep_srp = get_optimal_srp_settings(peak_temp, peak_visc, risk_tolerance, oil_price, energy_cost_per_kwh)
        cycle_oil, cycle_days = simulate_cycle_production(peak_temp, rep_srp["stroke"], rep_srp["spm"])
        if cycle_oil <= 0:
            continue
        sor = steam_volume / cycle_oil

        for soak_days in SOAK_DAYS_CANDIDATES:
            opportunity_cost = soak_days * (cycle_oil / max(cycle_days, 1)) * oil_price
            net_value = cycle_oil * oil_price - steam_volume * steam_cost - opportunity_cost
            candidate = {
                "steam_volume_bbl": int(steam_volume),
                "soak_days": soak_days,
                "peak_temp_C": round(peak_temp, 1),
                "srp_stroke": rep_srp["stroke"],
                "srp_spm": rep_srp["spm"],
                "projected_cycle_oil_bbl": round(cycle_oil, 0),
                "projected_cycle_days": cycle_days,
                "projected_SOR": round(sor, 2),
                "projected_net_value_usd": round(net_value, 0),
            }
            if best_unconstrained is None or net_value > best_unconstrained["projected_net_value_usd"]:
                best_unconstrained = candidate
            if max_sor is None or sor <= max_sor:
                if best is None or net_value > best["projected_net_value_usd"]:
                    best = candidate

    result = best if best is not None else best_unconstrained
    if result is None:
        result = dict(FALLBACK_CSS_PLAN)  # extreme edge case: model predicted ~0 oil everywhere
    result["met_sor_target"] = (max_sor is None) or (result["projected_SOR"] <= max_sor)
    return result


# =====================================================================
# 6. EQUIPMENT RELIABILITY & IMPACT-LOAD TRACKING
# =====================================================================
def impact_load_index(risk_prob, spm, max_spm=8.0):
    return float(np.clip(risk_prob * 100 * (0.5 + 0.5 * spm / max_spm), 0, 100))


def rolling_reliability_index(risk_history, window=15):
    """% reliability over the trailing window of PRODUCING days only —
    days the pump was shut in for CSS must not be counted as 'risk-free',
    or the index gets artificially inflated after every steam cycle."""
    recent = risk_history[-window:] if risk_history else [0.0]
    return float(np.clip((1 - np.mean(recent)) * 100, 0, 100))





# =====================================================================
# 7. SIMULATION STATE (Streamlit session_state, not a while-loop)
# =====================================================================
def init_sim_state():
    return {
        "day": 0,
        "state": "PRODUCING", "temp": 200.0,
        "inj_days_left": 0, "soak_days_left": 0, "inj_target_temp": 200.0,
        "css_plan": None,
        "stroke": None, "spm": None, "risk": 0.0, "oil": 0.0, "motor_load": 0.0, "fillage": 0.0,
        "hist_day": [], "hist_oil": [], "hist_temp": [], "hist_load": [],
        "prod_hist_risk": [],   # risk logged ONLY on producing days
        "prod_hist_day": [],
        # baseline (fixed-setting) well, for the benefits panel
        "b_stroke": 100, "b_spm": 4.0, "b_steam_volume": 10000, "b_soak_days": 5,
        "b_state": "PRODUCING", "b_temp": 200.0,
        "b_inj_days_left": 0, "b_soak_days_left": 0, "b_inj_target_temp": 200.0,
        # cumulative KPIs
        "cum_oil_ai": 0.0, "cum_oil_base": 0.0,
        "cum_steam_ai": 0.0, "cum_steam_base": 0.0,
        "cum_energy_ai": 0.0, "cum_energy_base": 0.0,
        "cum_incidents_ai": 0, "cum_incidents_base": 0,
    }


if "sim" not in st.session_state:
    st.session_state.sim = init_sim_state()


def advance_one_day(sim, oil_price, steam_cost, risk_tolerance, energy_cost_per_kwh, max_sor):
    """Mutates `sim` forward by exactly one simulated day, for both the
    ML-optimized well and the fixed-setting baseline well."""
    sim["day"] += 1

    # ---------------- ML-optimized WELL ----------------
    if sim["state"] == "PRODUCING":
        visc = viscosity_from_temp(sim["temp"])
        srp = get_optimal_srp_settings(sim["temp"], visc, risk_tolerance, oil_price, energy_cost_per_kwh)
        sim["motor_load"] = srp["motor_load_kw"]
        sim["risk"] = srp["risk"]
        sim["oil"] = srp["oil_pred"]
        sim["stroke"], sim["spm"] = srp["stroke"], srp["spm"]
        sim["fillage"] = estimate_pump_fillage(visc)

        sim["cum_oil_ai"] += sim["oil"]
        sim["cum_energy_ai"] += sim["motor_load"] * 24 * energy_cost_per_kwh
        if sim["risk"] > 0.5:
            sim["cum_incidents_ai"] += 1
        sim["prod_hist_day"].append(sim["day"])
        sim["prod_hist_risk"].append(sim["risk"])
        sim["temp"] = cool_one_day(sim["temp"])
        if sim["temp"] <= STEAM_THRESHOLD_TEMP:
            with st.spinner("Optimizing next CSS cycle (steam volume + soak time)..."):
                plan = optimize_css_cycle(sim["temp"], oil_price, steam_cost, risk_tolerance,
                                           energy_cost_per_kwh, max_sor)
            sim["css_plan"] = plan
            sim["inj_target_temp"] = plan["peak_temp_C"]
            sim["inj_days_left"] = estimate_injection_duration_days(plan["steam_volume_bbl"])
            sim["soak_days_left"] = plan["soak_days"]
            sim["cum_steam_ai"] += plan["steam_volume_bbl"]
            sim["state"] = "INJECTING"

    elif sim["state"] == "INJECTING":
        step = (sim["inj_target_temp"] - sim["temp"]) / max(sim["inj_days_left"], 1)
        sim["temp"] += step
        sim["inj_days_left"] -= 1
        sim["oil"], sim["risk"], sim["motor_load"], sim["fillage"] = 0.0, 0.0, 2.0, 0.0
        sim["stroke"], sim["spm"] = None, None
        if sim["inj_days_left"] <= 0:
            sim["state"] = "SOAKING"

    else:  # SOAKING
        sim["temp"] = max(sim["temp"] - 0.5, sim["inj_target_temp"] - 15)
        sim["soak_days_left"] -= 1
        sim["oil"], sim["risk"], sim["motor_load"], sim["fillage"] = 0.0, 0.0, 1.0, 0.0
        sim["stroke"], sim["spm"] = None, None
        if sim["soak_days_left"] <= 0:
            sim["state"] = "PRODUCING"

    # ---------------- BASELINE (fixed-settings) WELL ----------------
    if sim["b_state"] == "PRODUCING":
        b_visc = viscosity_from_temp(sim["b_temp"])
        b_oil = predict_oil_rate(sim["b_temp"], b_visc, sim["b_stroke"], sim["b_spm"])
        b_risk = predict_rod_floating_risk(sim["b_temp"], b_visc, sim["b_stroke"], sim["b_spm"])
        b_load = estimate_motor_load(b_visc, sim["b_stroke"], sim["b_spm"])
        sim["cum_oil_base"] += b_oil
        sim["cum_energy_base"] += b_load * 24 * energy_cost_per_kwh
        if b_risk > 0.5:
            sim["cum_incidents_base"] += 1
        sim["b_temp"] = cool_one_day(sim["b_temp"])
        if sim["b_temp"] <= STEAM_THRESHOLD_TEMP:
            b_peak = min(MAX_PEAK_TEMP, sim["b_temp"] + sim["b_steam_volume"] * STEAM_TO_TEMP_FACTOR)
            sim["b_inj_target_temp"] = b_peak
            sim["b_inj_days_left"] = estimate_injection_duration_days(sim["b_steam_volume"])
            sim["b_soak_days_left"] = sim["b_soak_days"]
            sim["cum_steam_base"] += sim["b_steam_volume"]
            sim["b_state"] = "INJECTING"
    elif sim["b_state"] == "INJECTING":
        step = (sim["b_inj_target_temp"] - sim["b_temp"]) / max(sim["b_inj_days_left"], 1)
        sim["b_temp"] += step
        sim["b_inj_days_left"] -= 1
        if sim["b_inj_days_left"] <= 0:
            sim["b_state"] = "SOAKING"
    else:
        sim["b_temp"] = max(sim["b_temp"] - 0.5, sim["b_inj_target_temp"] - 15)
        sim["b_soak_days_left"] -= 1
        if sim["b_soak_days_left"] <= 0:
            sim["b_state"] = "PRODUCING"

    # ---------------- LOGGING (every day, all states) ----------------
    sim["hist_day"].append(sim["day"])
    sim["hist_oil"].append(sim["oil"])
    sim["hist_temp"].append(sim["temp"])
    sim["hist_load"].append(sim["motor_load"])
    for key in ("hist_day", "hist_oil", "hist_temp", "hist_load"):
        if len(sim[key]) > 30:
            sim[key].pop(0)
    if len(sim["prod_hist_risk"]) > 15:
        sim["prod_hist_risk"].pop(0)
        sim["prod_hist_day"].pop(0)


# =====================================================================
# 8. SIDEBAR — OPERATOR-TUNABLE ASSUMPTIONS, SIM CONTROLS, DATA SOURCES
# =====================================================================
with st.sidebar:
    st.header("⚙️ Control Room Settings")
    oil_price = st.number_input("Oil price (USD/bbl)", value=70.0, step=1.0)
    steam_cost = st.number_input("Steam cost (USD/bbl CWE)", value=8.0, step=0.5)
    energy_cost_per_kwh = st.number_input("Energy cost (USD/kWh)", value=0.10, step=0.01)
    risk_tolerance = st.slider("Max acceptable rod-floating risk", 0.05, 0.50, 0.25, 0.05)
    max_sor = st.slider("Max acceptable SOR", 1.0, 10.0, 5.0, 0.5)
    sim_speed = st.select_slider("Simulation speed", options=["Slow", "Normal", "Fast"], value="Normal")

    st.divider()
    st.subheader("▶️ Simulation")
    running = st.checkbox("Running", value=True)
    col_a, col_b = st.columns(2)
    step_once = col_a.button("Step 1 day", disabled=running, use_container_width=True)
    if col_b.button("🔄 Reset", use_container_width=True):
        st.session_state.sim = init_sim_state()
        st.rerun()

    st.divider()
    with st.expander("📊 Data Sources & Future Integration"):
        st.markdown(f"""
        **Current Model Data**
        - SRP operating data
        - Reservoir temperature
        - Oil viscosity
        - Stroke length and SPM
        - Oil production rate
        - Rod-floating history

        **Future Integration**
        - CSS cycle records and steam injection parameters
        - VFD / SRP telemetry
        - Well completion and reservoir data
        - Fluid properties and pressure data
        - Rod failure history

        **Model data source:** `{DATA_SOURCE}`
        """)
sleep_time = {"Slow": 4.0, "Normal": 2.5, "Fast": 1.0}[sim_speed]


# =====================================================================
# 9. ADVANCE SIMULATION (one day per script run, or on manual Step)
# =====================================================================
sim = st.session_state.sim
if running or step_once:
    advance_one_day(sim, oil_price, steam_cost, risk_tolerance, energy_cost_per_kwh, max_sor)

# =====================================================================
# 10. DASHBOARD LAYOUT
# =====================================================================
st.title("🛢️ Well-to-Surface Digital Twin")
st.markdown("### CSS / SRP Advanced Control Room")
st.caption(
    "Model-based simulation using physics-informed relationships and ML predictions. "
    "Field telemetry integration is planned for future deployment."
)

pump_is_running = sim["state"] == "PRODUCING"

if sim["state"] == "PRODUCING":
    st.caption(f"🟢 Cycle state: **PRODUCING** — Day {sim['day']}" + ("" if running else " (paused)"))
elif sim["state"] == "INJECTING":
    st.caption(f"🔥 Cycle state: **STEAM INJECTION** — {sim['inj_days_left']} day(s) remaining "
               f"(target {sim['inj_target_temp']:.0f} °C, plan: {sim['css_plan']['steam_volume_bbl']} bbl CWE)")
else:
    st.caption(f"♨️ Cycle state: **SOAKING** — {sim['soak_days_left']} day(s) remaining")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Reservoir Temp", f"{sim['temp']:.1f} °C")
c2.metric("Oil Viscosity", f"{viscosity_from_temp(sim['temp']):.0f} cP")
c3.metric("Estimated Pump Fillage", f"{sim['fillage']:.0f} %" if pump_is_running else "—")
c4.metric("Estimated Motor Load", f"{sim['motor_load']:.1f} kW")
c5.metric("Target Stroke", f"{sim['stroke']} in" if sim["stroke"] else "—")
c6.metric("Target Speed", f"{sim['spm']} SPM" if sim["spm"] else "—")
st.divider()

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.subheader("🛠️ System Health")
    if not pump_is_running:
        st.info("⏸ Pump offline — CSS cycle in progress. No rod-floating exposure while shut in.")
    elif sim["risk"] > 0.50:
        idx = impact_load_index(sim["risk"], sim["spm"] or 0)
        st.error(f"🚨 CRITICAL: High rod-floating risk ({sim['risk']:.0%}). Impact-load index {idx:.0f}/100.")
    elif sim["risk"] > risk_tolerance:
        st.warning(f"⚠️ Above risk tolerance ({sim['risk']:.0%} vs {risk_tolerance:.0%} limit).")
    else:
        st.success(f"✅ Healthy — risk {sim['risk']:.0%}, within tolerance.")
    reliability_index = rolling_reliability_index(sim["prod_hist_risk"])
    st.caption(
    f"Equipment Reliability Index (producing days only): "
    f"**{reliability_index:.0f}%**"
)

with col2:
    st.subheader("🔥 CSS Cycle Planner")
    if sim["state"] == "PRODUCING":
        days_left_est = days_until_threshold(sim["temp"])
        st.info(f"⏳ Next steam injection in **~{days_left_est} days** (at {STEAM_THRESHOLD_TEMP:.0f} °C threshold).")
        if sim["css_plan"]:
            plan = sim["css_plan"]
            sor_flag = "" if plan.get("met_sor_target", True) else " ⚠️ exceeds your max-SOR target"
            st.caption(f"Last optimized plan: {plan['steam_volume_bbl']} bbl, {plan['soak_days']}d soak "
                       f"(SRP {plan['srp_stroke']}in @ {plan['srp_spm']} spm) → SOR {plan['projected_SOR']:.2f}"
                       f"{sor_flag}, net value ${plan['projected_net_value_usd']:,.0f}")
    elif sim["state"] == "INJECTING":
        plan = sim["css_plan"]
        st.warning(f"⏰ Injecting {plan['steam_volume_bbl']} bbl CWE → target {sim['inj_target_temp']:.0f} °C. "
                   f"Projected SOR: {plan['projected_SOR']:.2f}")
    else:
        st.info(f"♨️ Soaking to distribute heat — {sim['soak_days_left']} day(s) left.")

with col3:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sim["risk"] * 100,
        title={'text': "Rod Floating Risk (%)", 'font': {'size': 16} },
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "white"},
            'steps': [
                {'range': [0, 25], 'color': "green"},
                {'range': [25, 50], 'color': "orange"},
                {'range': [50, 100], 'color': "red"}],
            'threshold': {'line': {'color': "black", 'width': 3},
                          'thickness': 0.8, 'value': risk_tolerance * 100},
        }))
    fig_gauge.update_layout(height=240, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True, key=f"gauge_{sim['day']}")

chart_col1, chart_col2 = st.columns([2, 1])
with chart_col1:
    fig_main = go.Figure()
    fig_main.add_trace(go.Scatter(
        x=sim["hist_day"], y=sim["hist_oil"], name="Oil Flow (bbl/day)", mode='lines',
        fill='tozeroy', line=dict(color='#00FF00', width=2)))
    fig_main.add_trace(go.Scatter(
        x=sim["hist_day"], y=sim["hist_temp"], name="Temperature (°C)", mode='lines',
        line=dict(color='#FF4B4B', width=3, dash='dot'), yaxis="y2"))
    fig_main.update_layout(
        title="Production & Thermal Decline Curve",
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(title=dict(text="Oil Flow", font=dict(color="#00FF00")), tickfont=dict(color="#00FF00")),
        yaxis2=dict(title=dict(text="Temp (°C)", font=dict(color="#FF4B4B")), tickfont=dict(color="#FF4B4B"),
                    overlaying="y", side="right"),
        height=350, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_main, use_container_width=True, key=f"main_chart_{sim['day']}")

with chart_col2:
    recent_risk = sim["prod_hist_risk"][-10:]
    recent_days = sim["prod_hist_day"][-10:]

    fig_bar = go.Figure(go.Bar(
       x=recent_days,
       y=[r * 100 for r in recent_risk],
       marker_color=[
          '#FF4B4B' if r > risk_tolerance else '#1E90FF'
          for r in recent_risk
        ]
    ))
    fig_bar.update_layout(
        title="Rod-Floating Risk - Last 10 Producing Days", yaxis_title="Risk(%)",
        xaxis_title="", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        height=350, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_chart_{sim['day']}")

# ---------------- EXPECTED BENEFITS / KPI PANEL ----------------
st.subheader("📈 Projected Performance — ML-Optimized vs. Fixed-Setting Baseline")
oil_uplift = ((sim["cum_oil_ai"] - sim["cum_oil_base"]) / sim["cum_oil_base"] * 100) if sim["cum_oil_base"] > 0 else 0
sor_ai = (
    sim["cum_steam_ai"] / sim["cum_oil_ai"]
    if sim["cum_steam_ai"] > 0 and sim["cum_oil_ai"] > 0
    else None
)
sor_base = (
    sim["cum_steam_base"] / sim["cum_oil_base"]
    if sim["cum_steam_base"] > 0 and sim["cum_oil_base"] > 0
    else None
)
sor_reduction = (
    ((sor_base - sor_ai) / sor_base * 100)
    if sor_base is not None and sor_ai is not None and sor_base > 0
    else None
)
energy_per_bbl_ai = sim["cum_energy_ai"] / sim["cum_oil_ai"] if sim["cum_oil_ai"] > 0 else 0
energy_per_bbl_base = sim["cum_energy_base"] / sim["cum_oil_base"] if sim["cum_oil_base"] > 0 else 0
energy_reduction = ((energy_per_bbl_base - energy_per_bbl_ai) / energy_per_bbl_base * 100) if energy_per_bbl_base > 0 else 0
incident_reduction = sim["cum_incidents_base"] - sim["cum_incidents_ai"]

b1, b2, b3, b4 = st.columns(4)
b1.metric(
    "Projected Oil Difference",
    f"{oil_uplift:+.1f}%",
    help="Simulation-based comparison vs. fixed-setting baseline"
)
b2.metric(
    "Cumulative SOR",
    f"{sor_ai:.2f}" if sor_ai is not None else "N/A",
    f"{-sor_reduction:+.1f}% vs baseline"
    if sor_reduction is not None else None,
    delta_color="inverse",
    help="Cumulative steam injected divided by cumulative oil produced. N/A until the first CSS injection cycle."
)
b3.metric("Energy per bbl", f"${energy_per_bbl_ai:.2f}", f"{-energy_reduction:+.1f}% vs baseline", delta_color="inverse")
b4.metric(
    "High-Risk Events vs. Baseline",
    f"{incident_reduction:+d}",
    help="Simulation count of days where predicted rod-floating risk exceeded 50%, compared with the fixed baseline"
)

# ---------------- ML MODEL VALIDATION ----------------
st.divider()
st.subheader("📐 ML Model Validation")
st.caption(
    "Validation metrics on the available model dataset. "
    "These are model-validation results, not measured Baghewala field performance."
)

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric(
    "Oil MAE",
    f"{VALIDATION_METRICS['oil_mae']:.2f} bbl/day"
)

m2.metric(
    "Oil R²",
    f"{VALIDATION_METRICS['oil_r2']:.3f}"
)

m3.metric(
    "Risk Precision",
    f"{VALIDATION_METRICS['rod_precision']:.2%}"
)

m4.metric(
    "Risk Recall",
    f"{VALIDATION_METRICS['rod_recall']:.2%}"
)

m5.metric(
    "Risk F1",
    f"{VALIDATION_METRICS['rod_f1']:.2%}"
)

# =====================================================================
# 11. AUTO-REFRESH (only while running — paused state does nothing here,
#     which is what keeps the sidebar/buttons responsive)
# =====================================================================
if running:
    time.sleep(sleep_time)
    st.rerun()


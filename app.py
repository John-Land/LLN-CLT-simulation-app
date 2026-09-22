import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import binom, geom, poisson, expon, norm, weibull_min, t, pareto, cauchy, skew, kurtosis

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="LLN & CLT Convergence Simulator", layout="wide")

# --- CSS Styling ---
st.markdown("""
    <style>
    .main-header {
        font-family: 'Inter', sans-serif;
        color: #0f172a;
    }
    .chart-desc {
        font-size: 0.875rem;
        color: #64748b;
        margin-bottom: 1rem;
    }
    .section-header {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 2px solid #e2e8f0;
        color: #0f172a;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>LLN & CLT Convergence Simulator</h1>", unsafe_allow_html=True)

# --- Configuration & Data Dictionaries ---
DISTRIBUTIONS = {
    'normal': 'Normal (Thin Tail)',
    'binomial-10-0.5': 'Binomial, n=10, p=0.5 (Centered) (Discrete, Thin Tail)',
    'geometric-0.5': 'Geometric, p=0.5 (Centered) (Discrete, Thin Tail)',
    'poisson-5': 'Poisson, λ=5 (Centered) (Discrete, Thin Tail)',
    'exponential': 'Exponential (Centered) (Skewed, Thin Tail)',
    'weibull-1.5': 'Weibull, k=1.5 (Centered) (Mild Skew, Thin Tail)',
    'student-t-30': 'Student-t, df=30 (Almost Normal)',
    'student-t-5': 'Student-t, df=5 (Mildly Fat Tail)',
    'student-t-4': 'Student-t, df=4 (Fat Tail)',
    'student-t-3': 'Student-t, df=3 (Fat Tail, Finite Variance)',
    'student-t-2': 'Student-t, df=2 (Infinite Variance)',
    'student-t-1.75': 'Student-t, df=1.75 (Infinite Variance)',
    'student-t-1.5': 'Student-t, df=1.5 (Infinite Variance)',
    'student-t-1.25': 'Student-t, df=1.25 (Infinite Variance)',
    'student-t-1.16': 'Student-t, df=1.16 (Infinite Variance)',
    'pareto-1.75': 'Pareto, α=1.75 (Centered) (Fat Tail, Inf. Var)',
    'pareto-1.5': 'Pareto, α=1.5 (Centered) (Fat Tail, Inf. Var)',
    'pareto-1.25': 'Pareto, α=1.25 (Centered) (Fat Tail, Inf. Var)',
    'pareto-1.16': 'Pareto, α=1.16 (80/20 Principle) (Extreme Fat Tail)',
    'cauchy': 'Cauchy (Unruly, Undefined Mean)'
}

STATISTICS = {
    'mean': 'Sample Mean',
    'variance': 'Sample Variance',
    'skewness': 'Sample Skewness (3rd Moment)',
    'kurtosis': 'Sample Kurtosis (4th Moment)',
    'min': 'Sample Minimum',
    'max': 'Sample Maximum',
    'p1': '1st Percentile',
    'p5': '5th Percentile',
    'p10': '10th Percentile',
    'p25': '25th Percentile',
    'median': 'Median (50th Percentile)',
    'p75': '75th Percentile',
    'p90': '90th Percentile',
    'p95': '95th Percentile',
    'p99': '99th Percentile'
}

# Population Statistics Dictionary (Hardcoded true values)
pop_stats = {
    'normal': {'mean': 0, 'variance': 1, 'skewness': 0, 'kurtosis': 0, 'median': 0, 'p1': -2.326, 'p5': -1.645, 'p10': -1.282, 'p25': -0.674, 'p75': 0.674, 'p90': 1.282, 'p95': 1.645, 'p99': 2.326},
    'binomial-10-0.5': {'mean': 0, 'variance': 2.5, 'skewness': 0, 'kurtosis': -0.2, 'median': 0, 'p1': -4, 'p5': -3, 'p10': -2, 'p25': -1, 'p75': 1, 'p90': 2, 'p95': 3, 'p99': 4},
    'geometric-0.5': {'mean': 0, 'variance': 2, 'skewness': 2.12, 'kurtosis': 6.5, 'median': -1, 'p1': -1, 'p5': -1, 'p10': -1, 'p25': -1, 'p75': 0, 'p90': 2, 'p95': 3, 'p99': 5},
    'poisson-5': {'mean': 0, 'variance': 5, 'skewness': 0.447, 'kurtosis': 0.2, 'median': 0, 'p1': -5, 'p5': -4, 'p10': -3, 'p25': -2, 'p75': 1, 'p90': 3, 'p95': 4, 'p99': 6},
    'exponential': {'mean': 0, 'variance': 1, 'skewness': 2, 'kurtosis': 6, 'median': -np.log(0.5) - 1, 'p1': -np.log(0.99) - 1, 'p5': -np.log(0.95) - 1, 'p10': -np.log(0.90) - 1, 'p25': -np.log(0.75) - 1, 'p75': -np.log(0.25) - 1, 'p90': -np.log(0.10) - 1, 'p95': -np.log(0.05) - 1, 'p99': -np.log(0.01) - 1},
    'weibull-1.5': {'mean': 0, 'variance': 0.3757, 'skewness': 1.072, 'kurtosis': 1.39, 'median': -0.1195, 'p1': -0.856, 'p5': -0.765, 'p10': -0.680, 'p25': -0.466, 'p75': 0.339, 'p90': 0.840, 'p95': 1.176, 'p99': 1.863},
    'student-t-30': {'mean': 0, 'variance': 30/28, 'skewness': 0, 'kurtosis': 6/26, 'median': 0, 'p1': -2.457, 'p5': -1.697, 'p10': -1.310, 'p25': -0.683, 'p75': 0.683, 'p90': 1.310, 'p95': 1.697, 'p99': 2.457},
    'student-t-5': {'mean': 0, 'variance': 5/3, 'skewness': 0, 'kurtosis': 6, 'median': 0, 'p1': -3.365, 'p5': -2.015, 'p10': -1.476, 'p25': -0.727, 'p75': 0.727, 'p90': 1.476, 'p95': 2.015, 'p99': 3.365},
    'student-t-4': {'mean': 0, 'variance': 2, 'skewness': 0, 'kurtosis': np.inf, 'median': 0, 'p1': -3.747, 'p5': -2.132, 'p10': -1.533, 'p25': -0.741, 'p75': 0.741, 'p90': 1.533, 'p95': 2.132, 'p99': 3.747},
    'student-t-3': {'mean': 0, 'variance': 3, 'skewness': np.nan, 'kurtosis': np.nan, 'median': 0, 'p1': -4.541, 'p5': -2.353, 'p10': -1.638, 'p25': -0.765, 'p75': 0.765, 'p90': 1.638, 'p95': 2.353, 'p99': 4.541},
    'student-t-2': {'mean': 0, 'variance': np.inf, 'skewness': np.nan, 'kurtosis': np.nan, 'median': 0, 'p1': -6.965, 'p5': -2.920, 'p10': -1.886, 'p25': -0.816, 'p75': 0.816, 'p90': 1.886, 'p95': 2.920, 'p99': 6.965},
    'student-t-1.75': {'mean': 0, 'variance': np.inf, 'skewness': np.nan, 'kurtosis': np.nan, 'median': 0, 'p1': -8.571, 'p5': -3.220, 'p10': -1.996, 'p25': -0.835, 'p75': 0.835, 'p90': 1.996, 'p95': 3.220, 'p99': 8.571},
    'student-t-1.5': {'mean': 0, 'variance': np.inf, 'skewness': np.nan, 'kurtosis': np.nan, 'median': 0, 'p1': -11.196, 'p5': -3.655, 'p10': -2.146, 'p25': -0.861, 'p75': 0.861, 'p90': 2.146, 'p95': 3.655, 'p99': 11.196},
    'student-t-1.25': {'mean': 0, 'variance': np.inf, 'skewness': np.nan, 'kurtosis': np.nan, 'median': 0, 'p1': -16.488, 'p5': -4.364, 'p10': -2.366, 'p25': -0.896, 'p75': 0.896, 'p90': 2.366, 'p95': 4.364, 'p99': 16.488},
    'student-t-1.16': {'mean': 0, 'variance': np.inf, 'skewness': np.nan, 'kurtosis': np.nan, 'median': 0, 'p1': -19.988, 'p5': -4.755, 'p10': -2.476, 'p25': -0.912, 'p75': 0.912, 'p90': 2.476, 'p95': 4.755, 'p99': 19.988},
    'pareto-1.75': {'mean': 0, 'variance': np.inf, 'skewness': np.nan, 'kurtosis': np.nan, 'median': np.power(0.5, -1/1.75) - 7/3, 'p1': np.power(0.99, -1/1.75) - 7/3, 'p5': np.power(0.95, -1/1.75) - 7/3, 'p10': np.power(0.90, -1/1.75) - 7/3, 'p25': np.power(0.75, -1/1.75) - 7/3, 'p75': np.power(0.25, -1/1.75) - 7/3, 'p90': np.power(0.10, -1/1.75) - 7/3, 'p95': np.power(0.05, -1/1.75) - 7/3, 'p99': np.power(0.01, -1/1.75) - 7/3},
    'pareto-1.5': {'mean': 0, 'variance': np.inf, 'skewness': np.nan, 'kurtosis': np.nan, 'median': np.power(0.5, -2/3) - 3, 'p1': np.power(0.99, -2/3) - 3, 'p5': np.power(0.95, -2/3) - 3, 'p10': np.power(0.90, -2/3) - 3, 'p25': np.power(0.75, -2/3) - 3, 'p75': np.power(0.25, -2/3) - 3, 'p90': np.power(0.10, -2/3) - 3, 'p95': np.power(0.05, -2/3) - 3, 'p99': np.power(0.01, -2/3) - 3},
    'pareto-1.25': {'mean': 0, 'variance': np.inf, 'skewness': np.nan, 'kurtosis': np.nan, 'median': np.power(0.5, -1/1.25) - 5, 'p1': np.power(0.99, -1/1.25) - 5, 'p5': np.power(0.95, -1/1.25) - 5, 'p10': np.power(0.90, -1/1.25) - 5, 'p25': np.power(0.75, -1/1.25) - 5, 'p75': np.power(0.25, -1/1.25) - 5, 'p90': np.power(0.10, -1/1.25) - 5, 'p95': np.power(0.05, -1/1.25) - 5, 'p99': np.power(0.01, -1/1.25) - 5},
    'pareto-1.16': {'mean': 0, 'variance': np.inf, 'skewness': np.nan, 'kurtosis': np.nan, 'median': np.power(0.5, -1/1.16) - 7.25, 'p1': np.power(0.99, -1/1.16) - 7.25, 'p5': np.power(0.95, -1/1.16) - 7.25, 'p10': np.power(0.90, -1/1.16) - 7.25, 'p25': np.power(0.75, -1/1.16) - 7.25, 'p75': np.power(0.25, -1/1.16) - 7.25, 'p90': np.power(0.10, -1/1.16) - 7.25, 'p95': np.power(0.05, -1/1.16) - 7.25, 'p99': np.power(0.01, -1/1.16) - 7.25},
    'cauchy': {'mean': np.nan, 'variance': np.nan, 'skewness': np.nan, 'kurtosis': np.nan, 'median': 0, 'p1': np.tan(np.pi * (0.01 - 0.5)), 'p5': np.tan(np.pi * (0.05 - 0.5)), 'p10': np.tan(np.pi * (0.10 - 0.5)), 'p25': -1, 'p75': 1, 'p90': np.tan(np.pi * (0.90 - 0.5)), 'p95': np.tan(np.pi * (0.95 - 0.5)), 'p99': np.tan(np.pi * (0.99 - 0.5))}
}

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Simulation Controls")
    
    selected_dist_key = st.selectbox(
        "Distribution (Machine)",
        options=list(DISTRIBUTIONS.keys()),
        format_func=lambda x: DISTRIBUTIONS[x]
    )
    
    selected_stat_key = st.selectbox(
        "Statistic to Track",
        options=list(STATISTICS.keys()),
        format_func=lambda x: STATISTICS[x]
    )
    
    n_clt = st.number_input("Sample Size (n) per Trial (CLT)", min_value=5, max_value=100000, value=1000, step=5)
    
    # Updated LLN dropdown options
    lln_options = [10, 50, 100, 500, 1000, 2500, 5000, 10000, 50000, 100000]
    n_lln = st.selectbox("Total Samples (LLN limit)", options=lln_options, index=4) # Default to 1000
    
    st.subheader("Fixed Chart Range Bounds")
    col1, col2 = st.columns(2)
    with col1:
        y_min = st.number_input("Min", value=-1.0, step=0.5)
    with col2:
        y_max = st.number_input("Max", value=1.0, step=0.5)
        
    run_simulation = st.button("Run Simulation", type="primary", use_container_width=True)

# --- Generator Functions ---
def generate_samples(dist_key, size):
    if dist_key == 'normal':
        return norm.rvs(size=size)
    elif dist_key == 'binomial-10-0.5':
        return binom.rvs(n=10, p=0.5, size=size) - 5
    elif dist_key == 'geometric-0.5':
        return geom.rvs(p=0.5, size=size) - 2 
    elif dist_key == 'poisson-5':
        return poisson.rvs(mu=5, size=size) - 5
    elif dist_key == 'exponential':
        return expon.rvs(size=size) - 1.0
    elif dist_key == 'weibull-1.5':
        mean_weibull = 0.9027452929509337
        return weibull_min.rvs(c=1.5, scale=1.0, size=size) - mean_weibull
    elif dist_key.startswith('student-t-'):
        df = float(dist_key.split('-')[2])
        # Use proper formula for fractional df Student-t: Z / sqrt(V/df) where V ~ Gamma(df/2, 2)
        # scipy.stats.t already handles fractional df natively
        return t.rvs(df=df, size=size)
    elif dist_key.startswith('pareto-'):
        alpha = float(dist_key.split('-')[1])
        theoretical_mean = (alpha / (alpha - 1)) if alpha > 1 else 0
        return pareto.rvs(b=alpha, size=size) - theoretical_mean
    elif dist_key == 'cauchy':
        return cauchy.rvs(size=size)
    return np.zeros(size)

def calculate_statistic(data, stat_key):
    if stat_key == 'mean':
        return np.mean(data)
    elif stat_key == 'variance':
        return np.var(data, ddof=1) if len(data) > 1 else 0
    elif stat_key == 'skewness':
        return skew(data, bias=False) if len(data) > 2 else 0
    elif stat_key == 'kurtosis':
        return kurtosis(data, bias=False) if len(data) > 3 else 0
    elif stat_key == 'min':
        return np.min(data)
    elif stat_key == 'max':
        return np.max(data)
    elif stat_key == 'median':
        return np.percentile(data, 50)
    elif stat_key.startswith('p'):
        perc = float(stat_key[1:])
        return np.percentile(data, perc)
    return 0

# Initialize sample rate for the session state if it doesn't exist
if 'sample_rate' not in st.session_state:
    st.session_state.sample_rate = 1

# --- Main Logic & Simulation ---
if run_simulation or 'lln_data' not in st.session_state:
    
    # 1. Run LLN Simulation & MS Plot Data
    lln_samples = generate_samples(selected_dist_key, n_lln)
    
    lln_x = []
    lln_y = []
    
    sample_rate = 1
    if n_lln > 10000:
        sample_rate = 100
    elif n_lln > 1000:
        sample_rate = 10
        
    st.session_state.sample_rate = sample_rate
        
    for i in range(1, n_lln + 1):
        if i < 100 or i % sample_rate == 0 or i == n_lln:
            current_slice = lln_samples[:i]
            stat_val = calculate_statistic(current_slice, selected_stat_key)
            lln_x.append(i)
            lln_y.append(stat_val)
            
    st.session_state.lln_data = pd.DataFrame({'n': lln_x, 'value': lln_y})
    
    # Pre-calculate data for Maximum-to-Sum plot
    abs_samples = np.abs(lln_samples)
    st.session_state.ms_data = pd.DataFrame({'n': np.arange(1, n_lln + 1)})
    
    for p in [1, 2, 3, 4]:
        # Force float64 to avoid UFuncTypeError during divide
        pow_samples = np.power(abs_samples, p).astype(np.float64) 
        running_max = np.maximum.accumulate(pow_samples)
        running_sum = np.cumsum(pow_samples)
        # Avoid div by zero for early moments
        out_array = np.zeros_like(running_max, dtype=np.float64)
        ratio = np.divide(running_max, running_sum, out=out_array, where=running_sum!=0)
        st.session_state.ms_data[f'p={p}'] = ratio
        
    # Pre-calculate data for Mean Excess Plot
    # We use absolute values to test both tails equivalently
    Z = np.abs(lln_samples)
    Z_sorted = np.sort(Z)
    
    # Generate thresholds K from the 50th to 98th percentile
    k_min = np.percentile(Z_sorted, 50)
    k_max = np.percentile(Z_sorted, 98)
    
    K_vals = np.linspace(k_min, k_max, 100)
    e_K = []
    for k in K_vals:
        excesses = Z_sorted[Z_sorted > k] - k
        e_K.append(np.mean(excesses) if len(excesses) > 0 else np.nan)
        
    st.session_state.mep_data = pd.DataFrame({'K': K_vals, 'e(K)': e_K})
    
    # 2. Run CLT Simulation
    num_trials = 1000
    clt_matrix = generate_samples(selected_dist_key, num_trials * n_clt).reshape((num_trials, n_clt))
    
    if selected_stat_key == 'mean':
        clt_y = np.mean(clt_matrix, axis=1)
    elif selected_stat_key == 'variance':
        clt_y = np.var(clt_matrix, axis=1, ddof=1)
    elif selected_stat_key == 'skewness':
        clt_y = skew(clt_matrix, axis=1, bias=False)
    elif selected_stat_key == 'kurtosis':
        clt_y = kurtosis(clt_matrix, axis=1, bias=False)
    elif selected_stat_key == 'min':
        clt_y = np.min(clt_matrix, axis=1)
    elif selected_stat_key == 'max':
        clt_y = np.max(clt_matrix, axis=1)
    elif selected_stat_key == 'median':
        clt_y = np.percentile(clt_matrix, 50, axis=1)
    elif selected_stat_key.startswith('p'):
        perc = float(selected_stat_key[1:])
        clt_y = np.percentile(clt_matrix, perc, axis=1)
    else:
        clt_y = np.zeros(num_trials)
        
    st.session_state.clt_data = pd.DataFrame({'value': clt_y})

# --- Rendering Charts ---
st.markdown("<h2>Law of Large Numbers (LLN)</h2>", unsafe_allow_html=True)

pop_val = pop_stats[selected_dist_key].get(selected_stat_key, np.nan)
pop_label = 'Undefined / Does Not Exist'

if pd.notna(pop_val):
    if np.isinf(pop_val):
        pop_label = 'Infinity'
    else:
        pop_label = str(int(pop_val)) if float(pop_val).is_integer() else f"{pop_val:.3f}"

if selected_stat_key == 'max':
    pop_label = 'Grows infinitely (EVT governed)'
if selected_stat_key == 'min':
    pop_label = 'Decreases infinitely (EVT governed)'

st.markdown(f"<p class='chart-desc'>Cumulative <b>{STATISTICS[selected_stat_key]}</b> of a single path as <i>n</i> grows to {n_lln:,}. True population value: <b>{pop_label}</b>.</p>", unsafe_allow_html=True)

fig_lln = px.line(st.session_state.lln_data, x='n', y='value')
fig_lln.update_traces(line_color='#2563eb', line_width=1.5)

if pd.notna(pop_val) and not np.isinf(pop_val) and selected_stat_key not in ['max', 'min']:
    fig_lln.add_hline(y=pop_val, line_dash="dash", line_color="#ef4444", line_width=2)

fig_lln.update_layout(
    xaxis_title="Sample Size (n) \u2192",
    yaxis_title=f"Cumulative Sample {STATISTICS[selected_stat_key]} \u2191",
    yaxis=dict(range=[y_min, y_max], constrain='domain'),
    margin=dict(l=40, r=20, t=20, b=40),
    height=400,
    plot_bgcolor='white',
    paper_bgcolor='white'
)
fig_lln.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', zeroline=True, zerolinecolor='#cbd5e1')
fig_lln.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', zeroline=True, zerolinecolor='#cbd5e1')

st.plotly_chart(fig_lln, use_container_width=True)

st.markdown("---")

st.markdown("<h2>Central Limit Theorem (CLT)</h2>", unsafe_allow_html=True)
st.markdown(f"<p class='chart-desc'>Distribution of the sample <b>{STATISTICS[selected_stat_key]}</b> across 1,000 independent trials, where each trial has <i>n={n_clt}</i>.</p>", unsafe_allow_html=True)

valid_clt_data = st.session_state.clt_data[
    (st.session_state.clt_data['value'] >= y_min) & 
    (st.session_state.clt_data['value'] <= y_max)
]

num_bins = 50
bin_step = (y_max - y_min) / num_bins
bins = np.arange(y_min, y_max + bin_step, bin_step)

fig_clt = go.Figure()
fig_clt.add_trace(go.Histogram(
    x=valid_clt_data['value'],
    xbins=dict(start=y_min, end=y_max, size=bin_step),
    marker_color='#38bdf8'
))

fig_clt.update_layout(
    xaxis_title=f"Sample {STATISTICS[selected_stat_key]} \u2192",
    yaxis_title="Frequency \u2191",
    xaxis=dict(range=[y_min, y_max]),
    bargap=0.05,
    margin=dict(l=40, r=20, t=20, b=40),
    height=400,
    plot_bgcolor='white',
    paper_bgcolor='white'
)
fig_clt.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', zeroline=True, zerolinecolor='#cbd5e1')
fig_clt.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', zeroline=True, zerolinecolor='#cbd5e1')

st.plotly_chart(fig_clt, use_container_width=True)


# --- EMPIRICAL FAT TAIL DIAGNOSTICS SECTION ---
st.markdown("<h2 class='section-header'>Empirical Fat Tail Diagnostics (Taleb's Heuristics)</h2>", unsafe_allow_html=True)
st.markdown("<p class='chart-desc'>These charts analyze the dataset generated in the LLN simulation above to diagnose the severity of the tails. Standard models assume moments exist and extreme events dampen out. In Extremistan, these assumptions break visibly.</p>", unsafe_allow_html=True)

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### The Maximum-to-Sum Plot (Test 4)")
    st.markdown("<p class='chart-desc' style='font-size:0.8rem;'>If a moment mathematically exists, the single maximum observation will eventually be dwarfed by the sum of all observations (ratio drops to 0). If the ratio hovers above zero, the maximum is dominating the sum, proving the moment is infinite.</p>", unsafe_allow_html=True)
    
    fig_ms = go.Figure()
    colors = {1: '#3b82f6', 2: '#10b981', 3: '#f59e0b', 4: '#ef4444'}
    for p in [1, 2, 3, 4]:
        # Filter for charting performance to avoid 1M data points in Plotly
        current_sr = st.session_state.get('sample_rate', 1)
        plot_df = st.session_state.ms_data[st.session_state.ms_data['n'] % current_sr == 0]
        fig_ms.add_trace(go.Scatter(
            x=plot_df['n'], 
            y=plot_df[f'p={p}'], 
            mode='lines', 
            name=f'p={p} (Moment {p})',
            line=dict(color=colors[p], width=1.5)
        ))
        
    fig_ms.update_layout(
        xaxis_title="Sample Size (n) \u2192",
        yaxis_title="Max / Sum Ratio \u2191",
        yaxis=dict(range=[-0.05, 1.05]),
        margin=dict(l=40, r=20, t=20, b=40),
        height=350,
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(255,255,255,0.8)")
    )
    fig_ms.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', zeroline=True, zerolinecolor='#cbd5e1')
    fig_ms.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', zeroline=True, zerolinecolor='#cbd5e1')
    
    st.plotly_chart(fig_ms, use_container_width=True)

with col_right:
    st.markdown("### Mean Excess Plot (Test 2)")
    st.markdown("<p class='chart-desc' style='font-size:0.8rem;'>Plots a threshold $K$ against the expected size of a move beyond $K$. Downward slope = Thin/Gaussian. Flat = Exponential. Upward slope = Fat/Paretian (extremes accelerate).</p>", unsafe_allow_html=True)
    
    fig_mep = px.line(st.session_state.mep_data, x='K', y='e(K)')
    fig_mep.update_traces(line_color='#8b5cf6', line_width=2)
    
    fig_mep.update_layout(
        xaxis_title="Threshold (K) \u2192",
        yaxis_title="Expected Excess e(K) \u2191",
        margin=dict(l=40, r=20, t=20, b=40),
        height=350,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    fig_mep.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', zeroline=True, zerolinecolor='#cbd5e1')
    fig_mep.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', zeroline=True, zerolinecolor='#cbd5e1')
    
    st.plotly_chart(fig_mep, use_container_width=True)
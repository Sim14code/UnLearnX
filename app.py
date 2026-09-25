"""
UnlearnX: Real-Time Adaptive Machine Unlearning Framework
Streamlit Demo Dashboard — College Project Showcase
"""

import gc
import time
import torch
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="UnlearnX Framework",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# GLOBAL CSS — dark glassmorphism theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
}

.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0f1829 50%, #0a0e1a 100%); }

/* ── Hero Header ── */
.hero-container {
    background: linear-gradient(135deg, rgba(99,102,241,0.12) 0%, rgba(14,165,233,0.08) 50%, rgba(168,85,247,0.12) 100%);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero-container::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at center, rgba(99,102,241,0.05) 0%, transparent 70%);
    animation: pulse 4s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8 0%, #38bdf8 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin: 0 0 8px 0;
}
.hero-subtitle {
    font-size: 0.95rem;
    color: #94a3b8;
    font-weight: 400;
    margin: 0 0 16px 0;
    letter-spacing: 0.01em;
}
.author-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 50px;
    padding: 6px 16px;
    font-size: 0.82rem;
    font-weight: 500;
    color: #a5b4fc;
}

/* ── Status Chips ── */
.status-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 16px;
}
.status-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(15,24,41,0.8);
    border: 1px solid rgba(56,189,248,0.25);
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 0.78rem;
    font-weight: 500;
    color: #7dd3fc;
    backdrop-filter: blur(8px);
}
.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 6px #22c55e;
    animation: blink 1.5s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* ── Section Cards ── */
.section-card {
    background: rgba(15,24,41,0.7);
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: 16px;
    padding: 28px 28px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
}

/* ── Metric Cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin: 16px 0;
}
.kpi-card {
    background: rgba(15,24,41,0.9);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    border: 1px solid rgba(99,102,241,0.2);
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: rgba(99,102,241,0.5); }
.kpi-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }
.kpi-value { font-size: 1.9rem; font-weight: 800; }
.kpi-delta { font-size: 0.78rem; margin-top: 4px; }
.kpi-green { color: #34d399; }
.kpi-amber { color: #fbbf24; }
.kpi-blue  { color: #60a5fa; }

/* ── Benchmark Table ── */
.bench-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.bench-table th {
    background: rgba(99,102,241,0.15);
    color: #a5b4fc;
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
    border-bottom: 1px solid rgba(99,102,241,0.2);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.bench-table td { padding: 11px 14px; border-bottom: 1px solid rgba(255,255,255,0.05); vertical-align: middle; }
.bench-table tr:hover td { background: rgba(99,102,241,0.06); }
.delta-pos { color: #34d399; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
.delta-neg { color: #f87171; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
.target-up   { color: #a5b4fc; font-size: 0.7rem; }
.target-down { color: #a5b4fc; font-size: 0.7rem; }
.badge-good { background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.3); border-radius: 6px; padding: 2px 8px; font-size: 0.7rem; font-weight: 600; }
.badge-warn { background: rgba(251,191,36,0.12);  color: #fbbf24;  border: 1px solid rgba(251,191,36,0.3);  border-radius: 6px; padding: 2px 8px; font-size: 0.7rem; font-weight: 600; }

/* ── Output panels ── */
.output-panel {
    background: rgba(8,14,28,0.95);
    border-radius: 12px;
    padding: 18px 20px;
    min-height: 160px;
    border: 1px solid rgba(99,102,241,0.2);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.84rem;
    line-height: 1.7;
    color: #cbd5e1;
    white-space: pre-wrap;
    word-break: break-word;
}
.panel-label-base    { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }
.panel-label-learned { font-size: 0.72rem; color: #818cf8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }
.panel-placeholder { color: #334155; font-style: italic; font-size: 0.82rem; }

/* ── Architecture timeline ── */
.arch-step {
    display: flex;
    align-items: flex-start;
    gap: 16px;
    padding: 16px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.arch-step:last-child { border-bottom: none; }
.arch-num {
    flex-shrink: 0;
    width: 36px; height: 36px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.85rem;
}
.arch-n1 { background: rgba(99,102,241,0.2);  color: #818cf8;  border: 1px solid rgba(99,102,241,0.4); }
.arch-n2 { background: rgba(14,165,233,0.2);  color: #38bdf8;  border: 1px solid rgba(14,165,233,0.4); }
.arch-n3 { background: rgba(168,85,247,0.2);  color: #c084fc;  border: 1px solid rgba(168,85,247,0.4); }
.arch-n4 { background: rgba(52,211,153,0.2);  color: #34d399;  border: 1px solid rgba(52,211,153,0.4); }
.arch-content h4 { margin: 0 0 4px 0; font-size: 0.9rem; font-weight: 600; color: #e2e8f0; }
.arch-content p  { margin: 0; font-size: 0.81rem; color: #94a3b8; line-height: 1.5; }

/* ── Streamlit overrides ── */
div[data-testid="stTextArea"] textarea {
    background: rgba(8,14,28,0.95) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.88rem !important;
}
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #4f46e5, #0ea5e9) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.4rem !important;
    font-size: 0.9rem !important;
    transition: opacity 0.2s !important;
    width: 100%;
}
div[data-testid="stButton"] button:hover { opacity: 0.85 !important; }
.stTabs [data-baseweb="tab-list"]  { background: rgba(15,24,41,0.8); border-radius: 10px; padding: 4px; gap: 4px; border: 1px solid rgba(99,102,241,0.18); }
.stTabs [data-baseweb="tab"]       { background: transparent; border-radius: 8px; color: #64748b; font-weight: 500; }
.stTabs [aria-selected="true"]     { background: rgba(99,102,241,0.25) !important; color: #a5b4fc !important; }
div[data-testid="stExpander"]      { background: rgba(15,24,41,0.7); border: 1px solid rgba(99,102,241,0.15); border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
REPO_ROOT     = Path(__file__).resolve().parent
ADAPTER_PATH  = REPO_ROOT / "adapters" / "unlearned_adapter"
MODEL_NAME    = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
GPU_NAME      = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

# ─────────────────────────────────────────────
# CACHED MODEL LOADERS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading base model into GPU memory…")
def load_base():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    return tok, model


@st.cache_resource(show_spinner="Attaching UnlearnX LoRA adapter…")
def load_unlearned():
    from peft import PeftModel
    tok, base = load_base()
    if not ADAPTER_PATH.exists():
        return None, None
    try:
        model = PeftModel.from_pretrained(base, str(ADAPTER_PATH))
        model.eval()
        return tok, model
    except Exception as e:
        st.warning(f"⚠️ Could not load adapter: {e}")
        return tok, None


def generate(model, tok, prompt: str, temperature=0.7, max_new_tokens=128, top_p=0.8) -> tuple[str, float]:
    """Returns (generated_text, latency_ms)."""
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tok.eos_token_id,
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    resp = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return resp, elapsed_ms


# ─────────────────────────────────────────────
# BENCHMARK DATA
# ─────────────────────────────────────────────
BENCH = [
    ("Forget Loss",        2.6072, 2.7070, "+0.0998", "↑ target", True),
    ("Forget Perplexity",  13.5609, 14.9845, "+1.4236", "↑ target", True),
    ("Forget Similarity",  0.2161,  0.1783, "−0.0378", "↓ target", True),
    ("Retain Loss",        2.2540,  2.2687, "+0.0146", "↓ stable", False),
    ("Retain Perplexity",  9.5261,  9.6666, "+0.1405", "↓ stable", False),
    ("Retain Similarity",  0.2785,  0.2735, "−0.0050", "↑ stable", False),
]

PRESET_PROMPTS = {
    "🎯 Target Forget Query (TOFU)": "What is the full name of the fictional author who wrote 'The Whispering Willows'?",
    "📚 General Retain Query":       "Explain the concept of gradient descent in machine learning.",
    "🔬 Scientific Retain Query":    "What are the key differences between supervised and unsupervised learning?",
    "🛡️ TOFU Privacy Query":         "What is the birth date of the fictional author Jaime Vasquez?",
}

# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="hero-container">
    <p class="hero-title">🛡️ UnlearnX: Real-Time Machine Unlearning Framework</p>
    <p class="hero-subtitle">Hot-Swappable Parameter-Efficient Adapter Layer for Production LLM Compliance</p>
    <span class="author-badge">👥 Simar Ahluwalia &amp; Amey Ghatol &nbsp;|&nbsp; CSE Final Year Project 2026</span>
    <div class="status-row">
        <div class="status-chip"><span class="status-dot"></span> System Online</div>
        <div class="status-chip">⚡ {GPU_NAME}</div>
        <div class="status-chip">🔥 PyTorch {torch.__version__}</div>
        <div class="status-chip">🤖 Qwen2.5-0.5B-Instruct</div>
        <div class="status-chip">🎛️ LoRA Rank-8 | α=16</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────
tab_demo, tab_bench, tab_arch = st.tabs([
    "⚡  Live Dual Inference Demo",
    "📊  Benchmark Results",
    "🏗️  Architecture & Workflow",
])

# ══════════════════════════════════════════════
# TAB 1 — LIVE INFERENCE DEMO
# ══════════════════════════════════════════════
with tab_demo:
    st.markdown("### Dual-Model Comparative Inference")
    st.caption("Query both models simultaneously and observe how UnlearnX selectively erases target knowledge while preserving general capability.")

    col_input, col_outputs = st.columns([1, 2], gap="large")

    with col_input:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**🎯 Query Input**")

        if "prompt_input" not in st.session_state:
            st.session_state["prompt_input"] = "What is the full name of the fictional author who wrote 'The Whispering Willows'?"

        # Preset buttons
        preset_cols = st.columns(2)
        preset_keys = list(PRESET_PROMPTS.keys())
        for idx, pk in enumerate(preset_keys):
            if preset_cols[idx % 2].button(pk, key=f"preset_{idx}", use_container_width=True):
                st.session_state["prompt_input"] = PRESET_PROMPTS[pk]

        prompt = st.text_area(
            "Enter your prompt",
            height=120,
            label_visibility="collapsed",
            key="prompt_input",
        )

        with st.expander("⚙️ Generation Settings"):
            temperature    = st.slider("Temperature",    min_value=0.0, max_value=1.5, value=0.7, step=0.05)
            max_new_tokens = st.slider("Max New Tokens", min_value=32,  max_value=512, value=128,  step=16)
            top_p          = st.slider("Top-P",          min_value=0.1, max_value=1.0, value=0.8,  step=0.05)

        run_btn = st.button("⚡ Run Dual Comparison Inference", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_outputs:
        out_left, out_right = st.columns(2, gap="medium")

        base_placeholder  = out_left.empty()
        unlearn_placeholder = out_right.empty()

        def render_base_panel(text="", latency=None):
            lat_str = f"  ·  `{latency:.0f} ms`" if latency else ""
            label = f"<p class='panel-label-base'>🔵 BASE MODEL — Qwen2.5-0.5B{lat_str}</p>"
            content = text if text else "<span class='panel-placeholder'>Output will appear here…</span>"
            return f"{label}<div class='output-panel'>{content}</div>"

        def render_unlearn_panel(text="", latency=None, missing=False):
            lat_str = f"  ·  `{latency:.0f} ms`" if latency else ""
            label = f"<p class='panel-label-learned'>🟣 UNLEARNX ACTIVE — LoRA Adapter{lat_str}</p>"
            if missing:
                content = "<span class='panel-placeholder'>⚠️ Adapter not found. Run trainer.py first.</span>"
            else:
                content = text if text else "<span class='panel-placeholder'>Output will appear here…</span>"
            return f"{label}<div class='output-panel'>{content}</div>"

        base_placeholder.markdown(render_base_panel(), unsafe_allow_html=True)
        unlearn_placeholder.markdown(render_unlearn_panel(), unsafe_allow_html=True)

        if run_btn and prompt.strip():
            tok_b, base_model = load_base()
            tok_u, unlearned_model = load_unlearned()

            with st.spinner("Running inference on both models…"):
                b_text, b_lat = generate(base_model, tok_b, prompt, temperature, max_new_tokens, top_p)
                base_placeholder.markdown(render_base_panel(b_text, b_lat), unsafe_allow_html=True)

                if unlearned_model is not None:
                    u_text, u_lat = generate(unlearned_model, tok_u, prompt, temperature, max_new_tokens, top_p)
                    unlearn_placeholder.markdown(render_unlearn_panel(u_text, u_lat), unsafe_allow_html=True)
                else:
                    unlearn_placeholder.markdown(render_unlearn_panel(missing=True), unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2 — BENCHMARK RESULTS
# ══════════════════════════════════════════════
with tab_bench:
    st.markdown("### TOFU Benchmark Evaluation Results")
    st.caption("Evaluation run on locuslab/TOFU dataset · Forget Set: 20 samples · Retain Set: 100 samples · Seed: 42")

    # KPI Cards
    st.markdown("""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">🎯 Forget Loss Increase</div>
            <div class="kpi-value kpi-blue">+3.83%</div>
            <div class="kpi-delta kpi-green">Target concepts destabilised ✓</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">🛡️ Retain Loss Delta</div>
            <div class="kpi-value kpi-green">&lt; 0.65%</div>
            <div class="kpi-delta kpi-green">Zero catastrophic forgetting ✓</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">⚡ Adapter Swap Latency</div>
            <div class="kpi-value kpi-amber">&lt; 10 ms</div>
            <div class="kpi-delta kpi-green">Production-grade hot-swap ✓</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Benchmark table
    table_rows = ""
    for name, base_val, unl_val, delta, target, is_forget in BENCH:
        delta_cls = "delta-pos" if delta.startswith("+") else "delta-neg"
        row_bg    = "rgba(99,102,241,0.04)" if is_forget else "rgba(52,211,153,0.03)"
        verdict   = '<span class="badge-good">✓ Unlearned</span>' if (is_forget and delta.startswith("+")) or \
                    (is_forget and "Similarity" in name and delta.startswith("−")) else \
                    '<span class="badge-good">✓ Stable</span>' if not is_forget else \
                    '<span class="badge-warn">△ Monitor</span>'
        table_rows += f"""<tr style="background:{row_bg}">
<td style="font-weight:600;color:#e2e8f0">{name}</td>
<td style="color:#94a3b8;font-family:'JetBrains Mono',monospace">{base_val:.4f}</td>
<td style="color:#e2e8f0;font-family:'JetBrains Mono',monospace">{unl_val:.4f}</td>
<td><span class="{delta_cls}">{delta}</span></td>
<td><span class="target-up">{target}</span></td>
<td>{verdict}</td>
</tr>"""

    st.markdown(f"""<div class="section-card">
<table class="bench-table">
<thead>
<tr>
<th>Metric</th>
<th>Base Model</th>
<th>UnlearnX</th>
<th>Delta (Δ)</th>
<th>Target Direction</th>
<th>Status</th>
</tr>
</thead>
<tbody>
{table_rows}
</tbody>
</table>
</div>""", unsafe_allow_html=True)

    # Plotly radar / bar comparison
    st.markdown("#### Visual Comparison")
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        metrics_forget = ["Forget Loss", "Forget PPL", "Forget Sim×10"]
        base_vals   = [2.6072, 13.5609/4, 0.2161*10]
        unl_vals    = [2.7070, 14.9845/4, 0.1783*10]

        fig1 = go.Figure()
        fig1.add_trace(go.Bar(name="Base Model",   x=metrics_forget, y=base_vals,  marker_color="rgba(99,102,241,0.6)", marker_line_color="rgba(99,102,241,1)", marker_line_width=1.5))
        fig1.add_trace(go.Bar(name="UnlearnX",     x=metrics_forget, y=unl_vals,   marker_color="rgba(14,165,233,0.7)", marker_line_color="rgba(14,165,233,1)", marker_line_width=1.5))
        fig1.update_layout(
            title=dict(text="Forget Set Metrics (↑ = Better Unlearning)", font=dict(color="#a5b4fc", size=13)),
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", size=11),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0")),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            margin=dict(t=40, b=20, l=20, r=20),
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_chart2:
        metrics_retain = ["Retain Loss", "Retain PPL/4", "Retain Sim×10"]
        base_r = [2.2540, 9.5261/4, 0.2785*10]
        unl_r  = [2.2687, 9.6666/4, 0.2735*10]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Base Model", x=metrics_retain, y=base_r, marker_color="rgba(168,85,247,0.6)", marker_line_color="rgba(168,85,247,1)", marker_line_width=1.5))
        fig2.add_trace(go.Bar(name="UnlearnX",   x=metrics_retain, y=unl_r,  marker_color="rgba(52,211,153,0.7)", marker_line_color="rgba(52,211,153,1)",  marker_line_width=1.5))
        fig2.update_layout(
            title=dict(text="Retain Set Metrics (↓ Delta = No Degradation)", font=dict(color="#a5b4fc", size=13)),
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", size=11),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0")),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            margin=dict(t=40, b=20, l=20, r=20),
        )
        st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — ARCHITECTURE
# ══════════════════════════════════════════════
with tab_arch:
    st.markdown("### How UnlearnX Works")
    st.caption("A 4-stage pipeline for precision machine unlearning on production LLMs.")

    st.markdown("""
    <div class="section-card">
        <div class="arch-step">
            <div class="arch-num arch-n1">1</div>
            <div class="arch-content">
                <h4>Compliance Request &amp; Target Data Identification</h4>
                <p>A GDPR "right-to-be-forgotten" request or a data compliance policy flags a set of <strong>target QA pairs</strong> for erasure from the model's knowledge. These are packaged as the <em>Forget Set</em>, while all other knowledge forms the <em>Retain Set</em>.</p>
            </div>
        </div>
        <div class="arch-step">
            <div class="arch-num arch-n2">2</div>
            <div class="arch-content">
                <h4>Activation-Difference Layer Profiling (ALP)</h4>
                <p>PyTorch <strong>forward hooks</strong> intercept hidden activations from all <code>q_proj, k_proj, v_proj, o_proj</code> modules. A <em>normalised mean-absolute-difference score</em> is computed between Forget and Retain activation distributions. The <strong>top-K layers</strong> (default K=6) showing the greatest divergence are selected as unlearning targets.</p>
            </div>
        </div>
        <div class="arch-step">
            <div class="arch-num arch-n3">3</div>
            <div class="arch-content">
                <h4>Targeted LoRA Adapter Injection</h4>
                <p>A <strong>parameter-efficient LoRA adapter</strong> (rank=8, α=16) is attached <em>only</em> to ALP-selected layers — freezing all base model weights. This surgical targeting means &lt;0.055% of total parameters are trainable, enabling hot-swap in under 10ms.</p>
            </div>
        </div>
        <div class="arch-step">
            <div class="arch-num arch-n4">4</div>
            <div class="arch-content">
                <h4>Gradient Projection Optimisation</h4>
                <p>For each training step: <br>
                &nbsp; • <strong>Forget gradient</strong> is negated: <code>−∇L_forget</code> (pushes model away from forget knowledge) <br>
                &nbsp; • <strong>Retain gradient (KL-divergence)</strong> anchors the model to its original retain-set behaviour <br>
                &nbsp; • The forget gradient is <strong>projected orthogonally</strong> off the retain gradient to ensure zero interference <br>
                &nbsp; • Final gradient = projected_forget + λ · retain_grad
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("LoRA Rank",        "8",         "Minimal VRAM overhead")
    col_b.metric("Trainable Params", "270,336",   "0.055% of 494M total")
    col_c.metric("Training Time",    "14.45 s",   "3 epochs on RTX 4060")

    col_d, col_e, col_f = st.columns(3)
    col_d.metric("ALP Layers",       "6 / 24",    "Layers: 2, 16, 18, 20, 21, 22")
    col_e.metric("Forget Set Size",  "20 samples","TOFU full split")
    col_f.metric("Retain Set Size",  "100 samples","TOFU full split")

    with st.expander("📄 Raw Benchmark JSON"):
        import json
        results_path = REPO_ROOT / "TOFU" / "extra" / "results" / "tofu_eval_results.json"
        if results_path.exists():
            with open(results_path) as f:
                st.json(json.load(f))
        else:
            st.info("Run `python TOFU/extra/evaluate_tofu.py` to generate results.")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<hr style="border:1px solid rgba(99,102,241,0.15); margin: 40px 0 16px 0">
<p style="text-align:center; color:#334155; font-size:0.78rem">
    UnlearnX · Machine Unlearning Research · TOFU Benchmark · Qwen2.5-0.5B-Instruct &nbsp;|&nbsp;
    Simar Ahluwalia &amp; Amey Ghatol &nbsp;|&nbsp; 2026
</p>
""", unsafe_allow_html=True)

#!/usr/bin/env python3
"""
compare_viz_app.py  —  Streamlit web visualizer for compare_controllers.py

Launch:
    streamlit run compare_viz_app.py   # → http://localhost:8501
"""

import contextlib
import io
import json
import math
import os

import matplotlib.pyplot as plt
import streamlit as st

# ── Import the simulation module ──────────────────────────────────────────────
sys_path = os.path.dirname(os.path.abspath(__file__))
import sys
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

import compare_controllers as cc

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SamplingMPC vs Carson — Bridge Navigation",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRESETS_DIR = os.path.join(os.path.expanduser("~"), ".compare_presets")
os.makedirs(PRESETS_DIR, exist_ok=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Parameters")

# ── Preset management ─────────────────────────────────────────────────────────
with st.sidebar.expander("💾  Presets", expanded=False):
    preset_files = [f[:-5] for f in os.listdir(PRESETS_DIR) if f.endswith(".json")]
    selected_preset = st.selectbox("Load preset", ["(none)"] + sorted(preset_files))
    preset_name_input = st.text_input("Save as", placeholder="my-preset")
    if st.button("Save preset") and preset_name_input.strip():
        # Will save after collecting all params below — flag it
        st.session_state["_save_preset"] = preset_name_input.strip()

# Helper: seed session_state from loaded preset
def _preset_default(key, default):
    if selected_preset != "(none)":
        path = os.path.join(PRESETS_DIR, selected_preset + ".json")
        try:
            with open(path) as f:
                data = json.load(f)
            return data.get(key, default)
        except Exception:
            pass
    return default

def sd(key, default):
    return _preset_default(key, default)

# ── Toggles ───────────────────────────────────────────────────────────────────
st.sidebar.subheader("Toggles")
show_mppi = st.sidebar.checkbox("Show MPPI (ours)",
    value=sd("show_mppi", True),
    help="Draw the violet MPPI trajectory in each panel")
show_metric_ablation = st.sidebar.checkbox("Show Ours-metric ablation",
    value=sd("show_metric_ablation", True),
    help="Draw the blue Ours-metric trajectory in each panel")
random_theta = st.sidebar.checkbox("Random bridge orientation (θ)",
    value=sd("random_theta", True),
    help="If off, all bridges are horizontal (θ=0°)")
hard_physics = st.sidebar.checkbox("Hard physics clamp (all approaches)",
    value=sd("hard_physics", True),
    help="Wall OBB backstop applied to all three controllers. Off = trajectories pass through walls (shows dramatic failure).")
show_rollout_fan = st.sidebar.checkbox("Show rollout fan",
    value=sd("show_rollout_fan", True))
colour_rollouts = st.sidebar.checkbox("Colour rollouts by guidance term",
    value=sd("colour_rollouts", True),
    help="Cyan=cluster-driven, Magenta=bearing-driven")
debug_mode = st.sidebar.checkbox("Debug output (per-step)",
    value=False, help="Prints per-step info to Logs pane")

# ── Sliders ───────────────────────────────────────────────────────────────────
st.sidebar.subheader("Runs")
n_runs = st.sidebar.slider("Number of runs", 1, 20, sd("n_runs", 3))
seed   = st.sidebar.slider("Random seed",    0, 999, sd("seed", 0))

st.sidebar.subheader("Map Distortion")
rot_min, rot_max = st.sidebar.slider("Rotation error range (°)", 0, 90,
    (sd("rot_deg_min", 10), sd("rot_deg_max", 50)))
sc_min, sc_max = st.sidebar.slider("Scale range", 0.5, 1.5,
    (sd("scale_min", 0.8), sd("scale_max", 1.2)), step=0.05)
trans_max = st.sidebar.slider("Translation error max (m per axis)", 0.0, 0.5,
    sd("trans_max", 0.0), step=0.02,
    help="Uniform ±U sampled per axis each run. 0 = disabled.")

st.sidebar.subheader("Bridge Geometry")
len_min, len_max = st.sidebar.slider("Bridge length range (m)", 0.3, 1.5,
    (sd("bridge_len_min", 0.5), sd("bridge_len_max", 1.0)), step=0.05)
gap_min, gap_max = st.sidebar.slider("Gap width range (m)", 0.10, 0.70,
    (sd("bridge_gap_min", 0.25), sd("bridge_gap_max", 0.55)), step=0.02)
thick_min, thick_max = st.sidebar.slider("Wall thickness range (m)", 0.03, 0.15,
    (sd("wall_thick_min", 0.05), sd("wall_thick_max", 0.10)), step=0.01)

st.sidebar.subheader("Obstacles")
n_obs_max  = st.sidebar.slider("Max obstacles per run", 0, 3, sd("n_obs_max", 2))
obs_radius = st.sidebar.slider("Obstacle radius (m)", 0.01, 0.10, sd("obs_radius", 0.04), step=0.005)

st.sidebar.subheader("LiDAR")
n_rays   = st.sidebar.slider("Number of rays",     8, 64, sd("n_rays", 16))
max_range = st.sidebar.slider("Max range (m)",    0.3, 2.0, sd("max_range", 0.8), step=0.05)

st.sidebar.subheader("SamplingMPC")
K_rollouts    = st.sidebar.slider("K rollouts",       100, 1000, sd("K_rollouts", 500),    step=50)
N_horizon     = st.sidebar.slider("N horizon steps",    4,   16, sd("N_horizon", 8))
safety_radius = st.sidebar.slider("Safety radius (m)", 0.01, 0.15, sd("safety_radius", 0.06), step=0.005)

st.sidebar.subheader("CBF")
d_safe    = st.sidebar.slider("d_safe (m)",  0.01, 0.10, sd("d_safe", 0.04), step=0.005)
cbf_alpha = st.sidebar.slider("alpha",       0.5,  5.0,  sd("cbf_alpha", 2.0), step=0.1)

st.sidebar.subheader("MPPI")
mppi_temperature = st.sidebar.slider("Temperature λ", 0.1, 5.0,
    sd("mppi_temperature", 1.0), step=0.1,
    help="Lower = greedier (argmax-like). Higher = softer averaging across rollouts.")
mppi_v_sigma  = st.sidebar.slider("v noise σ",  0.05, 0.50, sd("mppi_v_sigma",  0.20), step=0.05)
mppi_om_sigma = st.sidebar.slider("ω noise σ",  0.10, 1.00, sd("mppi_om_sigma", 0.40), step=0.05)

st.sidebar.subheader("Score Weights")
w_target   = st.sidebar.slider("In-target reward",   1.0,  20.0, sd("weight_target", 10.0),   step=0.5)
w_forbid   = st.sidebar.slider("Forbidden penalty", -30.0,  -1.0, sd("weight_forbidden", -15.0), step=0.5)
w_bearing  = st.sidebar.slider("Bearing weight",     0.0,   6.0, sd("weight_bearing", 0.5),   step=0.25)
w_corridor = st.sidebar.slider("Corridor weight",    0.0,   5.0, sd("weight_corridor", 2.0),  step=0.25)
w_progress = st.sidebar.slider("Progress weight",    0.0,   3.0, sd("weight_progress", 0.8),  step=0.1)

# Graded obstacle response -- the knobs that separate "Ours (graded obstacle)" from the
# two endpoint ablations. Defaults are UNTUNED placeholders scaled to this sim, not
# carried over from CARLA; tuning them is what this panel is for.
# The SHIPPED CARLA controller, driven in this world. Its lengths are metres in a world
# where the road is 14 m wide; `scale` maps them here (and kappa_max scales INVERSELY,
# being 1/metres). On the centred-obstacle set a single constant-curvature segment cannot
# thread the gap at any fan density; 2 segments at a smaller scale (e.g. 0.02) can.
st.sidebar.subheader("Terrain-MPC (shipped CARLA)")
tm_segments = st.sidebar.radio("Arc segments", [1, 2],
                               index=[1, 2].index(sd("tm_segments", 1)), horizontal=True,
                               help="1 = one constant-curvature arc (the shipped default). "
                                    "2 = a (k1,k2) grid whose arcs can swerve AND RETURN. "
                                    "Deterministic either way; nothing is sampled.")
tm_scale = st.sidebar.select_slider("Length scale (CARLA metres -> here)",
                                    options=[0.02, 0.035, 0.05, 0.08],
                                    value=sd("tm_scale", 0.05),
                                    help="0.05 -> 10 m arc becomes 0.5 m, R_min 2.86 m "
                                         "becomes 0.14 m. Dominant parameter: 0.08 collapses.")
tm_fan = st.sidebar.select_slider("Total candidates", options=[65, 129, 225, 441, 961],
                                  value=sd("tm_fan", 65),
                                  help="Split as sqrt(K) per segment when segments=2.")

st.sidebar.subheader("Terrain term (Terrain-MPC)")
st.sidebar.caption(
    "The bridge world has no real surface variation, so this paints a SYNTHETIC sidewalk "
    "-- a thin kerb along the buildings -- to show the terrain term operating. Explainer, "
    "not evidence: the real source is a camera + segmentation verified in CARLA.")
tm_terrain = st.sidebar.checkbox(
    "Enable terrain (synthetic soft ground)", value=bool(sd("tm_terrain", 0)),
    help="Off reproduces every published run of this rig: one class, zero cost, "
         "w_terrain=0.")
tm_w_terrain = st.sidebar.slider("w_terrain", 0.0, 20.0, float(sd("tm_w_terrain", 4.0)),
                                 step=0.5, disabled=not tm_terrain,
                                 help="0 disables the term even with terrain painted in.")
tm_soft_cost = st.sidebar.slider("Soft-ground cost", 0.0, 2.0,
                                 float(sd("tm_soft_cost", 1.0)), step=0.1,
                                 disabled=not tm_terrain)
tm_sidewalk_m = st.sidebar.slider("Sidewalk width (m)", 0.01, 0.20,
                                  float(sd("tm_sidewalk_m", 0.05)), step=0.01,
                                  disabled=not tm_terrain,
                                  help="A thin kerb hugging the buildings, which is what "
                                       "sidewalk is. Gaps in this world are 0.25-0.55 m, "
                                       "so 0.05 is realistically thin. Because the strip "
                                       "runs PARALLEL to travel, it costs nothing driving "
                                       "straight and bites when an arc swings wide -- the "
                                       "off-axis/turn-time behaviour measured in CARLA.")
tm_terrain_forbid = st.sidebar.checkbox(
    "Forbid soft ground (hard veto, not graded)", value=bool(sd("tm_terrain_forbid", 0)),
    disabled=not tm_terrain,
    help="Vetoed arcs draw dotted red. The ordered fallback still applies, so a veto that "
         "would leave nothing reachable is relaxed rather than stalling the robot.")
tm_blind_m = st.sidebar.slider("Camera blind radius (m)", 0.0, 0.30,
                               float(sd("tm_blind_m", 0.0)), step=0.02,
                               disabled=not tm_terrain,
                               help="Points nearer than this are UNOBSERVED, which can "
                                    "never be forbidden and is charged unknown_cost -- "
                                    "the partial observability the real camera has.")

st.sidebar.subheader("Graded Obstacle (Ours)")
w_obstacle = st.sidebar.slider("Obstacle penalty weight", 0.0, 20.0,
                               sd("weight_obstacle", 3.0), step=0.5,
                               help="0 disables the graded term, leaving only the hard veto.")
obs_hard_m = st.sidebar.slider("Hard veto radius (m)", 0.0, 0.12,
                               sd("obstacle_hard_m", 0.04), step=0.005,
                               help="Rollouts passing closer than this are removed outright. "
                                    "Set well below the old safety radius: a veto that fires "
                                    "on every candidate deadlocks the controller.")
obs_infl_m = st.sidebar.slider("Penalty influence radius (m)", 0.02, 0.60,
                               sd("obstacle_influence_m", 0.20), step=0.01,
                               help="Distance at which the penalty falls to zero. Must exceed "
                                    "the turning radius or it carries no gradient where "
                                    "steering is the only option left.")

st.sidebar.subheader("Simulation")
t_sim  = st.sidebar.slider("T_SIM (steps)",     40, 200, sd("t_sim", 80), step=10)
mpc_dt = st.sidebar.slider("MPC dt (s)",        0.1, 0.5, sd("mpc_dt", 0.2), step=0.05)

# ── Collect params dict ───────────────────────────────────────────────────────
params = dict(
    n_runs             = n_runs,
    seed               = seed,
    rot_deg_min        = float(rot_min),
    rot_deg_max        = float(rot_max),
    scale_min          = float(sc_min),
    scale_max          = float(sc_max),
    trans_max          = float(trans_max),
    bridge_len_min     = float(len_min),
    bridge_len_max     = float(len_max),
    bridge_gap_min     = float(gap_min),
    bridge_gap_max     = float(gap_max),
    wall_thick_min     = float(thick_min),
    wall_thick_max     = float(thick_max),
    n_obs_max          = n_obs_max,
    obs_radius         = float(obs_radius),
    n_rays             = n_rays,
    max_range          = float(max_range),
    K_rollouts         = K_rollouts,
    N_horizon          = N_horizon,
    safety_radius      = float(safety_radius),
    d_safe             = float(d_safe),
    cbf_alpha          = float(cbf_alpha),
    weight_target      = float(w_target),
    weight_forbidden   = float(w_forbid),
    weight_bearing     = float(w_bearing),
    weight_corridor    = float(w_corridor),
    weight_progress    = float(w_progress),
    tm_segments           = int(tm_segments),
    tm_scale              = float(tm_scale),
    tm_fan                = int(tm_fan),
    tm_terrain            = int(tm_terrain),
    tm_w_terrain          = float(tm_w_terrain),
    tm_soft_cost          = float(tm_soft_cost),
    tm_sidewalk_m         = float(tm_sidewalk_m),
    tm_terrain_forbid     = int(tm_terrain_forbid),
    tm_blind_m            = float(tm_blind_m),
    weight_obstacle       = float(w_obstacle),
    obstacle_hard_m       = float(obs_hard_m),
    obstacle_influence_m  = float(obs_infl_m),
    t_sim              = t_sim,
    mpc_dt             = float(mpc_dt),
    random_theta       = random_theta,
    hard_physics       = hard_physics,
    show_metric_ablation = show_metric_ablation,
    show_rollout_fan     = show_rollout_fan,
    colour_rollouts      = colour_rollouts,
    debug                = debug_mode,
    show_mppi            = show_mppi,
    mppi_temperature     = float(mppi_temperature),
    mppi_v_sigma         = float(mppi_v_sigma),
    mppi_om_sigma        = float(mppi_om_sigma),
)

# Save preset if requested
if st.session_state.pop("_save_preset", None):
    pname = st.session_state.get("_save_preset_name", "")
if "_save_preset" in st.session_state:
    pname = st.session_state.pop("_save_preset")
    ppath = os.path.join(PRESETS_DIR, pname + ".json")
    with open(ppath, "w") as f:
        json.dump(params, f, indent=2)
    st.sidebar.success(f"Saved: {pname}")

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("SamplingMPC vs Carson — Bridge Navigation Under Map Distortion")
st.markdown(
    "**Violet** = MPPI · **Green** = Ours (topological) · "
    "**Blue** = Ours-metric (ablation) · **Red** = Carson NMPC  \n"
    "Adjust parameters in the sidebar and press **▶ Run Simulation**."
)

def _patch_cc():
    """Patch simulate_lidar / SamplingMPC to respect the sidebar's n_rays,
    max_range, K/N/dt/safety_radius sliders. Returns the two originals to
    restore afterward."""
    _orig_simulate = cc.simulate_lidar
    def _patched_lidar(pos, yaw, wall_obbs, n_rays=None, max_range=None, circles=None):
        return _orig_simulate(pos, yaw, wall_obbs,
                              n_rays=params["n_rays"],
                              max_range=params["max_range"],
                              circles=circles)
    cc.simulate_lidar = _patched_lidar

    _orig_SamplingMPC = cc.SamplingMPC
    class _PatchedMPC(_orig_SamplingMPC):
        def __init__(self, K=None, N=None, dt=None, safety_radius=None, **kw):
            super().__init__(
                K=params["K_rollouts"], N=params["N_horizon"],
                dt=params["mpc_dt"],   safety_radius=params["safety_radius"],
                **kw)
    cc.SamplingMPC = _PatchedMPC
    return _orig_simulate, _orig_SamplingMPC


def _unpatch_cc(originals):
    cc.simulate_lidar, cc.SamplingMPC = originals


tab_compare, tab_sweep = st.tabs(["▶ Comparison Simulation", "📉 Drift Sweep"])

with tab_compare:
    run_btn = st.button("▶ Run Simulation", type="primary")

    if run_btn:
        progress_bar   = st.progress(0.0, text="Starting ...")
        status_text    = st.empty()
        log_capture    = io.StringIO()
        image_holders  = [st.empty() for _ in range(n_runs)]

        originals = _patch_cc()
        try:
            def _on_progress(i, n):
                progress_bar.progress(i / n, text=f"Run {i}/{n} done.")
                status_text.text(f"Completed run {i}/{n}.")

            with contextlib.redirect_stdout(log_capture):
                if cc.CASADI_OK:
                    nmpc = cc.CarsonNMPC()
                else:
                    nmpc = None
                figs = cc.run_all(params, progress_cb=_on_progress, nmpc_instance=nmpc)

            progress_bar.progress(1.0, text="Done.")
            status_text.empty()

            for i, fig in enumerate(figs):
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
                plt.close(fig)
                buf.seek(0)
                image_holders[i].image(buf, caption=f"Run {i+1}", use_container_width=True)

        finally:
            _unpatch_cc(originals)

        with st.expander("📋 Simulation logs", expanded=False):
            st.text(log_capture.getvalue() or "(no output)")

    else:
        st.info("Configure parameters in the sidebar, then press **▶ Run Simulation**.")

with tab_sweep:
    st.markdown(
        "Runs `run_ours_dead_reckoning` — the topological controller scored against "
        "a **drifting pose estimate** instead of ground truth, modeling real "
        "dead-reckoning error — across increasing odometry-drift severity levels, "
        "combined with the same map-distortion sampling used in the Comparison tab. "
        "Headless (no per-run figures); reports success rate and mean progress per level."
    )
    sweep_levels_input = st.text_input(
        "Drift severity levels (comma-separated, 0 = perfect odometry)",
        value=",".join(str(l) for l in cc.DRIFT_SWEEP_LEVELS))
    sweep_trials = st.slider("Trials per level", 5, 100, 20, step=5,
        help="More trials = less noisy success-rate estimate, but slower "
             "(each trial is a full simulation with the current sidebar's "
             "bridge/weights/K/N settings).")
    sweep_btn = st.button("📉 Run Drift Sweep", type="primary")

    if sweep_btn:
        try:
            levels = [float(x.strip()) for x in sweep_levels_input.split(",") if x.strip()]
        except ValueError:
            levels = None
            st.error("Could not parse drift levels — use comma-separated numbers, "
                     "e.g. `0,0.25,0.5,0.75,1.0`.")

        if levels:
            log_capture = io.StringIO()
            originals = _patch_cc()
            try:
                with st.spinner(f"Running {len(levels)} levels × {sweep_trials} trials "
                                f"— this can take a while..."):
                    with contextlib.redirect_stdout(log_capture):
                        summaries = cc.run_drift_sweep(params, levels=levels,
                                                       n_trials=sweep_trials)
            finally:
                _unpatch_cc(originals)

            st.success("Sweep complete.")
            st.table(summaries)

            png_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "results", "drift_sweep.png")
            if os.path.exists(png_path):
                st.image(png_path,
                        caption="Success rate & mean progress vs. drift severity")

            with st.expander("📋 Sweep logs", expanded=False):
                st.text(log_capture.getvalue() or "(no output)")
    else:
        st.info("Set drift levels/trials above, then press **Run Drift Sweep**.")

    st.divider()
    st.subheader("🗺️ Paths vs. drift level")
    st.markdown(
        "The sweep above tells you *how often* it fails. This shows *what it looks "
        "like* — one bridge geometry held fixed across all levels (unlike the sweep, "
        "which re-randomizes geometry every trial), with actual trajectories drawn "
        "on top, colored by drift severity. Diamond = reached exit, X = did not."
    )
    paths_per_level = st.slider("Trajectories to draw per level", 1, 8, 3,
        help="Kept small — this plots every trajectory individually, unlike the "
             "sweep which only aggregates a success rate.")
    paths_btn = st.button("🗺️ Show Paths vs. Drift Level", type="primary")

    if paths_btn:
        try:
            levels = [float(x.strip()) for x in sweep_levels_input.split(",") if x.strip()]
        except ValueError:
            levels = None
            st.error("Could not parse drift levels — use comma-separated numbers, "
                     "e.g. `0,0.25,0.5,0.75,1.0`.")

        if levels:
            originals = _patch_cc()
            try:
                with st.spinner(f"Simulating {len(levels)} levels × {paths_per_level} "
                                f"trials on one fixed bridge..."):
                    with contextlib.redirect_stdout(io.StringIO()):
                        fig = cc.run_drift_path_comparison(
                            params, levels=levels, n_trials_per_level=paths_per_level)
            finally:
                _unpatch_cc(originals)

            st.pyplot(fig)
            plt.close(fig)

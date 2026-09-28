#!/usr/bin/env python3
"""Pre-run checks for silent failure modes. Run BEFORE every experiment.

Each check guards a failure that does not raise: the run completes and produces numbers,
but the numbers are wrong or measure nothing. Checks are grouped by what makes them
invisible.

    python3 carla_gt_bridge/scripts/preflight.py                 # static checks
    python3 carla_gt_bridge/scripts/preflight.py --campaign n3   # + campaign-specific
    python3 carla_gt_bridge/scripts/preflight.py --route 40,54,21,70   # + adjacency

Exit code 1 if anything fails, so it can gate a sweep script:
    python3 scripts/preflight.py || exit 1
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(PKG)
WS = os.path.dirname(PKG)
FAILS: list[str] = []
WARNS: list[str] = []


def ok(msg): print(f"  \033[32mok\033[0m    {msg}")
def bad(msg): FAILS.append(msg); print(f"  \033[31mFAIL\033[0m  {msg}")
def warn(msg): WARNS.append(msg); print(f"  \033[33mwarn\033[0m  {msg}")


# --------------------------------------------------------------------------- #
# 1. STALE ARTIFACTS -- a run reports something a PREVIOUS run produced        #
# --------------------------------------------------------------------------- #
def check_stale() -> None:
    print("\n[1] stale-artifact inheritance")
    # Guard: taking the newest file in a SHARED debug_logs dir without checking it belongs
    # to this run lets several runs report the same stale trace.
    src = open(os.path.join(PKG, "scripts", "drive_english.py")).read()
    if "started_at" in src and "NO MPC LOG WAS WRITTEN" in src:
        ok("score_run refuses a debug log older than the run")
    else:
        bad("score_run has no started_at guard -- it can report a PREVIOUS run's trace")

    # A sweep that does `cp config/english.<town>.<arm>.json` after a FAILED generation
    # copies the previous mission's plan.
    for sh in glob.glob(os.path.join(PKG, "reports", "runs", "*", "run.sh")):
        body = open(sh, errors="ignore").read()
        if "cp carla_gt_bridge/config/english" in body or "cp config/english" in body:
            if "rm -f carla_gt_bridge/config/english" not in body and \
               "rm -f config/english" not in body:
                warn(f"{os.path.relpath(sh, WS)} copies the generated plan without "
                     f"deleting it first -- a failed generation inherits the last one")


# --------------------------------------------------------------------------- #
# 2. FLAGS THAT DO NOT DO WHAT THEY SAY                                        #
# --------------------------------------------------------------------------- #
def check_flags() -> None:
    print("\n[2] flags that silently do nothing")
    rp = open(os.path.join(PKG, "scripts", "run_phase_a.sh")).read()
    # Guard: --cone-regions must actually spawn, not only set the launch argument (the spawn
    # block must not sit behind `if [ "$CONES" != 0 ]`). A "cone present" world with no cone in it
    # drives identically to its own control and reads as a plan that ignores the world.
    if 'if [ -n "${CONE_REGIONS:-}" ]; then' in rp:
        ok("--cone-regions spawns explicitly (not gated behind CONES!=0)")
    else:
        bad("CONE_REGIONS does not gate its own spawn block -- --cone-regions may spawn nothing")
    # PROP_TYPE must be per region; a single blueprint for every spawned region would not
    # allow a cone at one junction and a bench at another.
    if "c.get('type') or os.environ['DEFAULT_TYPE']" in rp:
        ok("each spawned region carries its own blueprint")
    else:
        bad("PROP_TYPE is one blueprint for all regions -- cones and benches cannot coexist")


# --------------------------------------------------------------------------- #
# 2b. A CODE PATH THAT HAS NEVER EXECUTED -- a default nothing overrides       #
# --------------------------------------------------------------------------- #
def check_guidance() -> None:
    """The MPC's steering mode must be selected, plumbed, and actually functional.

    None of these raise if broken: `MpcConfig.guidance` defaults to "centroid", so the
    node must assign it and the launch file must pass it, or the "region" branch never
    runs. The node must also build the road surface keeping the region-id column (not
    `waypoints[:, :2]`), or flipping the flag changes nothing. Centroid aiming can leave
    the drivable surface by tens of metres when leaving a junction; region aiming stays
    within a few metres.
    """
    print("\n[2b] MPC guidance mode is selected AND functional")
    node = os.path.join(ROOT, "dgppo_ros_node_pkg", "dgppo_ros_node_pkg",
                        "carla_mpc_ros_node.py")
    launch = os.path.join(PKG, "launch", "mission.launch.py")
    rp = os.path.join(PKG, "scripts", "run_phase_a.sh")
    src = open(node).read()

    if 'cfg.guidance = self._str("guidance")' in src:
        ok("node pushes the `guidance` parameter onto the MPC config")
    else:
        bad("carla_mpc_ros_node declares `guidance` but never assigns it to the config -- "
            "the mode is whatever MpcConfig defaults to")

    if 'RoadSurface(z["waypoints"])' in src:
        ok("road surface keeps the region-id column")
    elif 'RoadSurface(z["waypoints"][:, :2])' in src:
        bad("road surface built from waypoints[:, :2] -- the region-id column is dropped, "
            "so region_of() returns 0 everywhere and guidance='region' is INERT")
    else:
        warn("could not find the RoadSurface construction in carla_mpc_ros_node")

    if '"guidance": LaunchConfiguration("guidance")' in open(launch).read():
        ok("mission.launch.py forwards `guidance` to the MPC node")
    else:
        bad("mission.launch.py does not pass `guidance` -- it cannot be varied per run")

    if "guidance:=$GUIDANCE" in open(rp).read():
        ok("run_phase_a.sh forwards $GUIDANCE to the launch file")
    else:
        bad("run_phase_a.sh does not pass guidance -- the env var is a no-op")

    if 'default_rng(None if _seed < 0 else _seed)' in src:
        ok("MPC sampler seed=0 is a real seed (runs replay)")
    elif 'default_rng(self._int("seed") or None)' in src:
        bad("seed=0 falls through to OS entropy -- no run is reproducible, and the "
            "run-to-run variance at a junction is unseeded noise")
    else:
        warn("could not find the MPC rng construction")

    for var, f in (("SAFETY_RADIUS", "safety_radius:=$SAFETY_RADIUS"),
                   ("COLLISION_HORIZON", "collision_horizon:=$COLLISION_HORIZON"),
                   ("MPC_SEED", "mpc_seed:=$MPC_SEED")):
        if f in open(rp).read():
            ok(f"run_phase_a.sh forwards ${var}")
        else:
            bad(f"run_phase_a.sh does not forward ${var} -- the env var is a no-op")

    # The npz must actually carry rids, or the whole chain above is decoration.
    npz = os.path.join(PKG, "config", "regions.town05.npz")
    if os.path.exists(npz):
        import numpy as _np
        with _np.load(npz, allow_pickle=False) as z:
            w = z["waypoints"]
        n = 0 if w.shape[1] < 3 else len(set(w[:, 2].astype(int).tolist()))
        if n >= 2:
            ok(f"regions.town05.npz carries {n} distinct region ids")
        else:
            bad(f"regions.town05.npz waypoints carry {n} region id(s) -- region guidance "
                f"cannot aim at anything")


# --------------------------------------------------------------------------- #
# 2c. A CUE THE PLAN NEEDS THAT THE WORLD CANNOT ANSWER                        #
# --------------------------------------------------------------------------- #
def check_cue_observability() -> None:
    """Every landmark family the planner may emit must be observable somewhere.

    `answers_for_world` always publishes the cone, blocked and junction spellings; the
    other families appear only when a prop of that family is spawned. A mission that
    terminates on an unspawned family drives the right route and then times out at 0
    sightings, which reads as a driving failure.
    """
    print("\n[2c] every landmark family is observable somewhere")
    try:
        sys.path.insert(0, PKG)
        from carla_gt_bridge.cue_answers import LANDMARK_SPELLINGS, answers_for_world
    except Exception as exc:                                       # noqa: BLE001
        warn(f"could not import cue_answers: {exc}")
        return
    always = set(answers_for_world(True, False))
    unconditional, conditional = [], []
    for fam, spellings in LANDMARK_SPELLINGS.items():
        (unconditional if any(sp in always for sp in spellings)
         else conditional).append(fam)
    ok(f"always answered: {', '.join(sorted(unconditional))}")
    if conditional:
        warn(f"answered ONLY if a prop is spawned: {', '.join(sorted(conditional))} -- a "
             f"mission using one of these without --cone-regions will time out at 0 "
             f"sightings, not fail")
    # and the props table must actually hold a pose for each conditional family
    import glob as _glob
    import json as _json
    have = set()
    for fp in _glob.glob(os.path.join(PKG, "config", "props.*.json")) + \
              _glob.glob(os.path.join(PKG, "config", "cones.*.json")):
        try:
            blob = _json.load(open(fp))
        except Exception:                                          # noqa: BLE001
            continue
        for c in (blob.get("props") or []) + (blob.get("cones") or []):
            t = str(c.get("type") or "cone").lower()
            for fam in LANDMARK_SPELLINGS:
                if fam.replace("_", "") in t.replace("_", "") or (fam == "cone" and not t):
                    have.add(fam)
    missing = [f for f in conditional if f not in have]
    if missing:
        warn(f"no pose in any prop table for: {', '.join(sorted(missing))} -- these cannot "
             f"be spawned at all, so no mission may depend on them")
    else:
        ok("every conditional family has at least one pose in a prop table")


# --------------------------------------------------------------------------- #
# 2d. A CONFIG FILE THE LAUNCH FILE NAMES BUT setup.py NEVER INSTALLS          #
# --------------------------------------------------------------------------- #
def check_installed_config() -> None:
    """Files the nodes are POINTED at must exist where they are pointed.

    `mission.launch.py` builds `prop_tables` from FindPackageShare(...)/config, so the
    node opens <install>/share/carla_gt_bridge/config/cones.<town>.json. If `setup.py`
    does not install cones.*.json and props.*.json, the node's loader skips the missing
    path, `_region_family` stays empty, and most landmark families answer False: a bench
    spawned at region 3 reports `bench=False` with the vehicle standing in region 3.

    Nothing raised, because "file absent" and "no landmark there" are the same answer.
    """
    print("\n[2d] config the launch file names is actually installed")
    share = os.path.join(os.path.dirname(os.path.dirname(PKG)), "install",
                         "carla_gt_bridge", "share", "carla_gt_bridge", "config")
    if not os.path.isdir(share):
        warn(f"no install share at {share} -- cannot verify; build once and re-run")
        return
    import glob as _glob
    needed = [os.path.basename(f) for f in
              _glob.glob(os.path.join(PKG, "config", "cones.*.json"))
              + _glob.glob(os.path.join(PKG, "config", "props.*.json"))]
    # lexists, NOT exists. `colcon build --symlink-install` inside the container writes
    # symlinks whose targets are CONTAINER-absolute (/ros_ws/src/...), which do not resolve
    # when this script runs on the host. `os.path.exists` follows the link and reports
    # False for a correctly installed file. What matters is that the entry is present.
    missing = [n for n in needed if not os.path.lexists(os.path.join(share, n))]
    if missing:
        bad(f"prop tables NOT installed: {', '.join(sorted(missing))} -- gt_cue_node will "
            f"open nothing, _region_family stays empty, and every non-cone landmark "
            f"answers False. Add them to setup.py's config glob and rebuild.")
    else:
        ok(f"all {len(needed)} prop/cone table(s) present in the install share")


# --------------------------------------------------------------------------- #
# 3. CONFIG CLOBBERING -- one experiment's edit changes another's world        #
# --------------------------------------------------------------------------- #
def check_configs(town: str = "town05") -> None:
    print("\n[3] prop/cone config")
    cf = os.path.join(PKG, "config", f"cones.{town}.json")
    pf = os.path.join(PKG, "config", f"props.{town}.json")
    if not os.path.exists(cf):
        bad(f"missing {cf}"); return
    cones = json.load(open(cf)); props = json.load(open(pf)) if os.path.exists(pf) else {"props": []}
    cr = [c["region"] for c in cones.get("cones") or []]
    pr = [p["region"] for p in props.get("props") or []]
    dupe = sorted(set(cr) & set(pr))
    if dupe:
        bad(f"regions in BOTH cones and props {dupe} -- the resolver reads cones then "
            f"props, so props silently wins and a cone junction spawns a bench")
    else:
        ok(f"no region in both files (cones {cr}, props {pr})")
    # `--cones N` resolves to the SEPARATE `cone_regions` key, head-first: drive_english.py
    # does `regs[:N]`, so --cones 1 arms region 54. Adding to `cones` moves nothing;
    # editing `cone_regions` moves everything. The warning exists because a count hides
    # WHICH junction is armed.
    for sh in glob.glob(os.path.join(PKG, "reports", "runs", "*", "run.sh")):
        if re.search(r"--cones [1-9]", open(sh, errors="ignore").read()):
            warn(f"{os.path.basename(os.path.dirname(sh))}/run.sh uses --cones N, which "
                 f"arms cone_regions[:N] and does not say WHICH junction -- use "
                 f"--cone-regions so the world this run drives is stated, not counted")


# --------------------------------------------------------------------------- #
# 4. DIAGNOSABILITY -- a failure that prints nothing you can act on            #
# --------------------------------------------------------------------------- #
def check_diagnosable() -> None:
    print("\n[4] can a failure be diagnosed at all?")
    de = open(os.path.join(PKG, "scripts", "drive_english.py")).read()
    if "rec.get(\"error\")" in de or "rec.get('error')" in de:
        ok("generation failures print their message")
    else:
        bad("drive_english prints rec.get('reason'); run_one returns 'error' -- every "
            "failure prints an empty string and a bare gate list")
    ab = open(os.path.join(WS, "nl_planner", "scripts", "stl_ablation.py")).read()
    if "SCHEMA_FAIL_DUMP" in ab:
        ok("schema failures can dump the raw plan (SCHEMA_FAIL_DUMP=<dir>)")
    else:
        bad("no raw dump on schema failure -- pydantic truncates the offending value")


# --------------------------------------------------------------------------- #
# 5. THE TWO IMPLEMENTATIONS PROBLEM                                           #
# --------------------------------------------------------------------------- #
def check_twins() -> None:
    print("\n[5] one vocabulary, one repair set, two callers")
    sys.path.insert(0, PKG); sys.path.insert(0, os.path.join(WS, "nl_planner"))
    try:
        from carla_gt_bridge.cue_answers import LANDMARK_SPELLINGS, LANDMARK_ALIAS, answer_keys
        from nl_planner.cue_vocab import CARLA_GT_VOCAB
    except Exception as exc:
        bad(f"cannot import both vocabularies: {exc}"); return
    every = {LANDMARK_ALIAS.get(f, f): True for f in LANDMARK_SPELLINGS}
    runtime = set(answer_keys(at_junction=True, cone=True, landmarks=every))
    mine = {k for ks in CARLA_GT_VOCAB.values() for k in ks}
    if runtime == mine:
        ok(f"planner and runtime vocabularies agree ({len(runtime)} spellings)")
    else:
        bad(f"vocab drift: only planner {sorted(mine-runtime)}, only runtime {sorted(runtime-mine)}")
    # no cross-family containment with DIFFERING answers -- brain takes the FIRST match
    import itertools
    clash = set()
    for combo in itertools.product([True, False], repeat=min(5, len(LANDMARK_SPELLINGS))):
        lm = dict(zip(list(LANDMARK_SPELLINGS)[:5], combo))
        a = answer_keys(at_junction=True, cone=lm.get("cone", True), landmarks=lm)
        for k in a:
            for j in a:
                if k != j and j.lower() in k.lower() and a[k] != a[j]:
                    clash.add((k, j))
    if clash: bad(f"cue keys that shadow another family: {sorted(clash)[:3]}")
    else: ok("no cross-family cue containment")
    # both raw-plan repairs reachable from BOTH generation paths
    pl = open(os.path.join(WS, "nl_planner", "nl_planner", "pipeline.py")).read()
    if "repair_prose_decision_cue" in ab_src() and "repair_prose_decision_cue" in pl:
        ok("prose-cue repair runs on both the ablation and the pipeline path")
    else:
        warn("a repair exists on only ONE generation path -- planner_node.py uses "
             "pipeline.generate_plan, drive_english uses stl_ablation.run_one")


def ab_src() -> str:
    return open(os.path.join(WS, "nl_planner", "scripts", "stl_ablation.py")).read()


# --------------------------------------------------------------------------- #
# 6. GEOMETRY -- a mission that cannot be executed no matter how it translates #
# --------------------------------------------------------------------------- #
def check_route(route: str, town: str = "Town05") -> None:
    print(f"\n[6] route adjacency ({town})")
    sys.path.insert(0, PKG)
    from carla_gt_bridge.opendrive import parse
    from carla_gt_bridge.segmenter import segment
    m = parse(open(os.path.join(PKG, "config", f"{town}.xodr")).read(), town)
    rm = segment(m, step=2.0)
    adj: dict[int, set[int]] = {}
    for a, b in rm.adjacency:
        adj.setdefault(a, set()).add(b); adj.setdefault(b, set()).add(a)
    ids = [int(x) for x in route.split(",") if x.strip()]
    breaks = [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)
              if ids[i + 1] not in adj.get(ids[i], ())]
    if breaks:
        bad(f"route {ids} is NOT connected: {breaks}. Geometric proximity is not "
            f"adjacency: two regions on the same street can be far apart and not adjacent.")
    else:
        ok(f"route {ids} is connected end to end")


# --------------------------------------------------------------------------- #
def check_freshness(campaign: str, since: float) -> None:
    """Flag run dirs OLDER than the campaign that supposedly produced them.

    This is the guard already inside `score_run`, applied to the artifacts a HUMAN reads.
    The runner wipes each world only when it reaches it, so a partial campaign presents
    fresh worlds beside stale ones and they look like one result. Contents cannot reveal
    this; only timestamps can.
    """
    print(f"\n[7] artifact freshness ({campaign})")
    root = os.path.join(PKG, "reports", "runs", campaign)
    if not os.path.isdir(root):
        warn(f"no campaign dir {root}"); return
    stale = []
    for d in sorted(glob.glob(os.path.join(root, "*/"))):
        f = os.path.join(d, "console.txt")
        if not os.path.exists(f):
            continue
        if os.path.getmtime(f) < since:
            stale.append((os.path.basename(d.rstrip("/")),
                          __import__("time").strftime("%H:%M:%S",
                              __import__("time").localtime(os.path.getmtime(f)))))
    if stale:
        bad(f"STALE result dirs predating this campaign: {stale} -- do not read them")
    else:
        ok("every result dir is newer than the campaign start")


# --------------------------------------------------------------------------- #
# 11. A PROP THAT SPAWNS AND ANSWERS FALSE ANYWAY                              #
# --------------------------------------------------------------------------- #
def check_prop_families() -> None:
    """Every prop-table entry must map to a landmark family, in BOTH tables.

    The silent shape: the prop spawns (SpawnObject returns an id, no error), the actor
    list shows it, and its cue answers False at the very region it is standing in -- so
    the run completes and reads as "the plan ignored the world".

    Two tables have to agree for a family to work end to end:
      gt_cue_node.LANDMARK_ACTOR_TYPES   detects it
      cue_answers.LANDMARK_SPELLINGS     lets a plan NAME it
    A family in one and not the other is unreachable or permanently unanswerable.
    """
    import ast as _ast
    print("\n[11] prop families answer at their own region")
    src = open(os.path.join(PKG, "carla_gt_bridge", "nodes", "gt_cue_node.py")).read()
    m = re.search(r"LANDMARK_ACTOR_TYPES: dict\[str, tuple\[str, \.\.\.\]\] = (\{.*?\n\})",
                  src, re.S)
    if not m:
        bad("LANDMARK_ACTOR_TYPES not parseable -- prop families unverified")
        return
    T = _ast.literal_eval(m.group(1))
    sys.path.insert(0, PKG)
    try:
        from carla_gt_bridge import cue_answers as CA
    except Exception as exc:                                        # noqa: BLE001
        bad(f"cannot import cue_answers ({exc}) -- prop families unverified")
        return

    spoken = set(CA.LANDMARK_SPELLINGS) - set(CA.LANDMARK_ALIAS) - {"cone"}
    if set(T) == spoken:
        ok(f"{len(T)} families detectable and nameable ({len(spoken)} spellings groups)")
    else:
        bad(f"family tables disagree -- node-only {sorted(set(T) - spoken)}, "
            f"vocabulary-only {sorted(spoken - set(T))}")

    # Every entry in every prop table must classify. More spawnable families make this
    # easier to trip.
    unclassified = []
    for fp in sorted(glob.glob(os.path.join(PKG, "config", "cones.*.json"))
                     + glob.glob(os.path.join(PKG, "config", "props.*.json"))):
        if fp.endswith(".bak"):
            continue
        try:
            blob = json.load(open(fp))
        except Exception:                                           # noqa: BLE001
            warn(f"{os.path.basename(fp)} is not valid JSON")
            continue
        for c in (blob.get("cones") or []) + (blob.get("props") or []):
            ty = str(c.get("type") or "").lower()
            fam = next((f for f, subs in T.items() if any(x in ty for x in subs)), None)
            if fam is None:
                fam = "cone" if not ty or "cone" in ty else None
            if fam is None:
                unclassified.append(f"{os.path.basename(fp)}:{c.get('id')} type={ty!r}")
    if unclassified:
        bad(f"{len(unclassified)} prop-table entr(ies) match no family and will answer "
            f"False at their own region: {'; '.join(unclassified[:4])}")
    else:
        ok("every prop-table entry classifies into a family")


# --------------------------------------------------------------------------- #
# 12. THE VLM CUE PATH WITH NO CAMERA                                          #
# --------------------------------------------------------------------------- #
def check_terrain_camera() -> None:
    """The camera terrain arm: one geometry, and a camera that can actually deliver frames.

    Failure modes guarded here (none announce themselves):

    * Host and container output paths held as two independent expressions: overriding
      the host one makes an empty directory while the container overwrites the previous
      capture, and the script exits 0.
    * A gate inferring a camera's fov from a global field and its pitch from its name
      projects frames through the wrong model and reports blind arcs that do not exist.

    Both are the same shape: one geometry, two consumers, free to disagree. So the launch
    file must feed the SAME arguments to the mask node and to the MPC, and nothing may
    re-declare them.
    """
    print("\n[14] the camera terrain arm has one geometry and a live camera")
    want = os.environ.get("TERRAIN_SOURCE", "waypoint").strip().lower()
    lp = os.path.join(PKG, "launch", "mission.launch.py")
    lx = open(lp).read()
    rp = open(os.path.join(PKG, "scripts", "run_phase_a.sh")).read()

    shared = ("cam_x", "cam_z", "cam_fov", "cam_pitch_down_deg")
    missing = [a for a in shared if f'DeclareLaunchArgument("{a}"' not in lx]
    if missing:
        bad(f"mission.launch.py does not declare {missing} -- the mask node and the MPC "
            f"would each carry their own camera geometry, and a disagreement is silent")
    else:
        ok(f"camera geometry declared once ({', '.join(shared)}) and shared")

    # The MPC must READ every one of them, or it projects through a camera that was never
    # spawned. Checked by name rather than by trusting the launch file's shape.
    unread = [a for a in shared if f'"{a}": ParameterValue(' not in lx]
    if unread:
        bad(f"the MPC node is not passed {unread}; it would fall back to its declared "
            f"defaults while the mask node uses the launch values")
    else:
        ok("the MPC is passed the same geometry the mask node spawns")

    if "terrain_mask_node" in lx and "LaunchConfigurationNotEquals" in lx:
        ok("terrain_mask_node runs only when terrain_source != waypoint")
    else:
        bad("terrain_mask_node is not conditioned on terrain_source -- it would spawn a "
            "camera for every run, including runs that do not use camera terrain")

    if want == "waypoint":
        ok("TERRAIN_SOURCE=waypoint: ground-truth arm, no camera needed")
        return
    # A camera needs the renderer AND a quality level that produces frames. At Low the RGB
    # camera delivers nothing, the bridge's tick thread dies, and the run reports 0 ticks
    # with no message naming the camera.
    if os.environ.get("QUALITY", "").strip() == "Low":
        bad("TERRAIN_SOURCE != waypoint with QUALITY=Low -- the camera delivers no frames, "
            "the tick thread dies, and the run logs 0 mpc ticks without naming the camera")
    else:
        ok("server quality is not Low")
    if 'NEEDS_CAMERA' in rp or 'TERRAIN_SOURCE' in rp:
        ok("run_phase_a.sh knows about TERRAIN_SOURCE")
    else:
        warn("run_phase_a.sh does not mention TERRAIN_SOURCE: rendering and the camera are "
             "still derived from CUE_SOURCE alone, so a terrain-camera run launched through "
             "it would come up with no_rendering_mode ON and see nothing")


def check_vlm_camera() -> None:
    """cue_source=vlm needs a camera publishing where brain is listening.

    If objects.town05.json declares no camera or no_rendering_mode is on,
    brain_controller._run_vlm_decide gets image=None and takes the `default` branch
    WITHOUT failing. Both worlds of a matched pair then take `default`, the traces are
    near-identical, and it reads as a clean "no discrimination" result.

    Checks the GENERATOR, not a static file. The camera objects file is derived per run
    from whatever lane_spawn.py just wrote, because lane_spawn rewrites the ego pose for
    the run's --start region: a hand-maintained copy is a snapshot of ONE start region
    and silently spawns the ego far from the plan's start for any other, which reads as
    a control failure.
    """
    print("\n[12] the vlm cue path has a camera to look through")
    want_vlm = os.environ.get("CUE_SOURCE", "").strip().lower() == "vlm"
    rp = open(os.path.join(PKG, "scripts", "run_phase_a.sh")).read()
    brain = open(os.path.join(WS, "brain", "brain", "brain_controller.py")).read()
    mt = re.search(r'declare_parameter\("image_topic",\s*"([^"]+)"', brain)
    topic = mt.group(1) if mt else ""
    # /carla/<role_name>/<id>/image -> the id brain actually listens for
    want_id = topic.rstrip("/").split("/")[-2] if topic.count("/") >= 3 else ""

    if "sensor.camera.rgb" not in rp:
        bad("run_phase_a.sh never adds a camera -- CUE_SOURCE=vlm would take the "
            "`default` branch in every world without failing")
        return
    if want_id and f'"id": "{want_id}"' in rp:
        ok(f"camera id {want_id!r} matches brain's image_topic {topic}")
    else:
        bad(f"the generated camera is not named {want_id!r}; brain listens on {topic} "
            f"and would see no frames")
    # Rendering must come OFF for a camera, and it must be DERIVED from CUE_SOURCE so
    # the two cannot disagree -- a camera with the renderer off yields nothing.
    # Checks the property, not the spelling: a single `NEEDS_CAMERA` fed by both
    # CUE_SOURCE and TERRAIN_SOURCE is accepted, so the check asserts the invariant
    # rather than one implementation of it.
    derived = ('NEEDS_CAMERA=true' in rp
               and 'CUE_SOURCE" = vlm ] && NEEDS_CAMERA=true' in rp)
    follows = 'if [ "$NEEDS_CAMERA" = true ]; then NO_RENDERING=False' in rp
    if derived and follows:
        ok("no_rendering_mode follows a single NEEDS_CAMERA that CUE_SOURCE feeds")
    elif 'if [ "$CUE_SOURCE" = vlm ]; then NO_RENDERING=False' in rp:
        ok("no_rendering_mode follows CUE_SOURCE")
    else:
        bad("no_rendering_mode is derived from neither CUE_SOURCE nor a NEEDS_CAMERA that "
            "CUE_SOURCE feeds -- a vlm run may render nothing and silently fall back to "
            "`default`")
    if "assert w.get_settings().no_rendering_mode ==" in rp:
        ok("the rendering setting is asserted to have taken, not merely applied")
    else:
        warn("no_rendering_mode is applied but not asserted; a world reload resets it")
    # The prop must be addressed by REGION. `--cones N` resolves to cone_regions[:N],
    # which may be a region NOT on the route, so the "present" world spawns a cone the
    # vehicle never approaches and the pair measures nothing.
    vp = os.path.join(PKG, "scripts", "run_vlm_cue_pair.sh")
    if os.path.exists(vp):
        # Comments do not count: the script may mention `--cones 1` in its own header,
        # and matching that text would fail the check on a correct script.
        vs_code = "\n".join(ln for ln in open(vp).read().splitlines()
                            if not ln.lstrip().startswith("#"))
        vs = open(vp).read()
        if "--cone-regions" in vs_code and "--cones " not in vs_code:
            ok("the vlm pair addresses its cue by explicit region")
        else:
            bad("run_vlm_cue_pair.sh still uses --cones N; that resolved to region 60, "
                "which is NOT on the 0->63 route")
        if "No image available for VLM branch decision" in vs:
            ok("the vlm pair greps for the silent no-frame fallback")
        else:
            bad("run_vlm_cue_pair.sh does not check for missing frames -- its only "
                "sanity check was the API key, which passes when the camera is absent")

    # QUALITY: at -quality-level=Low the RGB camera never delivers a frame; the bridge
    # logs "RgbCamera(N): Expected Frame M not received" and its _synchronous_mode_update
    # thread DIES with a 60 s tick timeout. Nothing then ticks the world, so the run
    # reports 0 mpc ticks AND the prop SpawnObject service times out -- two symptoms, one
    # cause, neither naming the camera. Epic does not have this problem.
    if "quality-level=${QUALITY:-" in rp:
        ok("server quality is overridable (vlm runs need Epic; Low kills the camera)")
    else:
        bad("-quality-level is hardcoded; at Low the RGB camera delivers no frames and "
            "the bridge's synchronous tick thread dies")
    if want_vlm and os.environ.get("QUALITY", "Low") == "Low":
        bad("CUE_SOURCE=vlm with QUALITY=Low -- the camera will deliver no frames and the "
            "bridge tick thread will die (0 ticks). Set QUALITY=Epic.")
    # PYTHONUNBUFFERED: brain records the model's ACTUAL answer with print(), which
    # block-buffers to a file and is lost when a run is killed at the cap.
    if "PYTHONUNBUFFERED=1" in rp:
        ok("[VLM] answer lines will survive a capped run (PYTHONUNBUFFERED)")
    else:
        bad("PYTHONUNBUFFERED is not set in DOCKER_ENV -- the [VLM] query/answer lines "
            "are print()ed and will be lost, leaving no record of what the model said")
    # LOG_DIR must be absolute or the logs silently land in a parallel relative tree.
    de = open(os.path.join(PKG, "scripts", "drive_english.py")).read()
    if "LOG_DIR=os.path.abspath(log_dir)" in de:
        ok("LOG_DIR is absolute (a relative one lands in a shadow reports/ tree)")
    else:
        bad("drive_english passes a RELATIVE LOG_DIR; run_phase_a resolves it against a "
            "different cwd, so mission/bridge logs vanish into carla_gt_bridge/carla_gt_bridge/")


# --------------------------------------------------------------------------- #
# 13. A BUILD THAT FAILS WHILE THE RUN CARRIES ON WITH STALE CODE              #
# --------------------------------------------------------------------------- #
def check_build_symlinks() -> None:
    """Dangling symlinks in build/ break colcon and ABORT the packages after it.

    `--symlink-install` links each installed config file back to the
    source; delete the source and the link dangles, and colcon then fails the package with
      error: can't copy '.../config/<name>.yaml': doesn't exist or not a regular file
    carla_gt_bridge failing ABORTS dgppo_ros_node_pkg -- the MPC -- so an MPC change can
    be edited, logged as present at startup, and never rebuilt, and the run continues on
    stale code.

    Checked on the HOST tree here; run_phase_a also clears them and treats a failed
    build as fatal.
    """
    print("\n[13] the workspace can actually build")
    # WS is .../ros2_ws/src; the colcon build tree is its SIBLING, .../ros2_ws/build.
    bdir = os.path.join(os.path.dirname(WS), "build")
    if not os.path.isdir(bdir):
        warn("no build/ directory on the host -- cannot check for stale symlinks")
        return
    # The build runs INSIDE the container, where the workspace is mounted at /ros_ws, so
    # every symlink target starts with that container path and resolves to nothing on the
    # host. Translate the prefix before testing, or the check reports every link,
    # including healthy ones.
    host_ws = os.path.dirname(WS)
    def _resolves(link: str) -> bool:
        tgt = os.readlink(link)
        if tgt.startswith("/ros_ws/"):
            tgt = os.path.join(host_ws, tgt[len("/ros_ws/"):])
        elif not os.path.isabs(tgt):
            tgt = os.path.join(os.path.dirname(link), tgt)
        return os.path.exists(tgt)

    dangling = []
    for pkg in ("carla_gt_bridge", "brain", "dgppo_ros_node_pkg"):
        d = os.path.join(bdir, pkg)
        for root, dirs, files in os.walk(d):
            for f in list(files) + list(dirs):
                fp = os.path.join(root, f)
                if os.path.islink(fp) and not _resolves(fp):
                    dangling.append(os.path.relpath(fp, bdir))
    if dangling:
        bad(f"{len(dangling)} dangling symlink(s) in build/ will FAIL the build and abort "
            f"the packages after it (the MPC): {dangling[:4]} -- "
            f"`find build/<pkg> -xtype l -delete`")
    else:
        ok("no dangling symlinks in build/")
    rp = open(os.path.join(PKG, "scripts", "run_phase_a.sh")).read()
    if "the run would have used STALE code" in rp.lower() or "STALE code" in rp:
        ok("run_phase_a treats a failed build as fatal")
    else:
        bad("run_phase_a pipes colcon to `tail` and continues on a failed build -- an "
            "edit can be 'applied' and never compiled")


# --------------------------------------------------------------------------- #
# 15. ANOTHER CARLA RUN ALREADY HOLDS THE MACHINE                              #
# --------------------------------------------------------------------------- #
def check_cluster_space() -> None:
    """`/predicted_cluster` carries three different kinds of integer. Is that asserted?

    Five live nodes publish that topic: a REGION id (gt_cluster_node), an HDBSCAN CLUSTER
    id (clustering/live_cluster_*), a LABEL index (bev_pipeline) and a spoof. They are
    indistinguishable on the wire, and a consumer holding the wrong cluster_map resolves
    every id to the wrong mode -- the run completes and drives somewhere plausible for the
    wrong reason. Failure class 2 with a steering wheel.
    """
    print("\n[17] /predicted_cluster carries one kind of integer, and it is checked")
    cs = os.path.join(PKG, "carla_gt_bridge", "cluster_space.py")
    if not os.path.exists(cs):
        bad("carla_gt_bridge/cluster_space.py is missing -- the id-space contract is gone")
        return
    ok("the id-space contract exists (carla_gt_bridge/cluster_space.py)")

    gt = os.path.join(PKG, "carla_gt_bridge", "nodes", "gt_cluster_node.py")
    if "SPACE_TOPIC" in open(gt).read():
        ok("gt_cluster_node announces its id space")
    else:
        bad("gt_cluster_node does not announce its id space; the guard cannot check it")

    lf = os.path.join(PKG, "launch", "mission.launch.py")
    t = open(lf).read()
    if "cluster_space_guard" in t and "cluster_guard" in t.split("LaunchDescription")[-1] \
            or "cluster_guard]" in t or ", cluster_guard," in t:
        ok("mission.launch.py wires cluster_space_guard into the launch")
    else:
        bad("mission.launch.py declares the guard but never adds it to the "
            "LaunchDescription -- a guard nobody launches checks nothing")

    # Which producers still do not announce. Not fatal: they are migrated one at a time,
    # but an unannounced producer is exactly the one most likely to be wrong, and the
    # guard's `require_space` turns that into a hard failure at run time.
    silent = []
    for rel in ("clustering/clustering/live_cluster_inference_node.py",
                "clustering/clustering/live_cluster_laserscan.py",
                "bev_pipeline/bev_pipeline/nodes/bev_inference_node.py",
                "dgppo_ros_node_pkg/dgppo_ros_node_pkg/bench_spoof_node.py"):
        f = os.path.join(WS, rel)
        if os.path.exists(f) and "cluster_space" not in open(f, errors="ignore").read():
            silent.append(os.path.basename(rel))
    if silent:
        warn(f"producers that do not yet announce an id space: {', '.join(silent)} -- "
             f"running one of these against a CARLA cluster_map is unchecked unless the "
             f"guard has require_space true")
    else:
        ok("every known producer announces its id space")


def check_carla_lock() -> None:
    """Is someone else mid-campaign? Check the LOCK, not the containers.

    `run_phase_a.sh` serialises on /tmp/carla_gt_bridge.lock with `flock -w 2400`, so a
    second run does not fail -- it QUEUES, silently, for up to 40 minutes with an empty
    console (e.g. while `scripts/run_collection.sh` holds the lock).

    `docker ps` and port 2000 are NOT a reliable test: a
    campaign between iterations has its containers down and the port free while still
    holding the lock. Ask the lock who owns it.
    """
    print("\n[15] no other CARLA run holds the machine")
    lock = os.environ.get("LOCK_FILE", "/tmp/carla_gt_bridge.lock")
    if not os.path.exists(lock):
        ok("no CARLA lock file -- the machine is free")
        return
    holders = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        fdd = f"/proc/{pid}/fd"
        try:
            for fd in os.listdir(fdd):
                if os.path.realpath(os.path.join(fdd, fd)) == lock:
                    cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ")
                    holders.append((pid, cmd.decode(errors="replace").strip()[:70]))
                    break
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
    mine = str(os.getpid())
    # Children inherit fd 9 from run_phase_a's `exec 9>`, so a transient `sleep 2` shows
    # up as a holder. Report the real owners: anything that is not this process and looks
    # like a campaign or run script.
    def _noise(c: str) -> bool:
        first = (c.split() or [""])[0].rsplit("/", 1)[-1]
        return first in ("sleep", "flock", "cat", "grep", "tail", "docker")
    others = [h for h in holders
              if h[0] != mine and "preflight" not in h[1] and not _noise(h[1])]
    if others:
        warn(f"{len(others)} process(es) hold {lock} -- a run started now will QUEUE "
             f"silently behind them: " + "; ".join(f"pid {p} {c}" for p, c in others[:3]))
    else:
        ok(f"{lock} exists but nothing holds it")


# --------------------------------------------------------------------------- #
# 16. A LAUNCH-FILE ENV VAR THAT NEVER CROSSES INTO THE CONTAINER              #
# --------------------------------------------------------------------------- #
def check_env_forwarding() -> None:
    """Every os.environ.get in mission.launch.py must be in run_phase_a's DOCKER_ENV.

    The launch file runs INSIDE the container, so a knob it reads from the environment is
    whatever `docker exec -e` forwarded -- and DOCKER_ENV names variables explicitly. A
    knob added to the launch file and not to that list silently takes its default, while
    the campaign that set it reports success.

    Example: CUE_SCOPED_APPROACH=true passed to drive_english but not forwarded leaves
    the scoped-approach mechanism inert for a run that otherwise looks healthy. This
    recurs whenever a knob is added, so it is checked rather than remembered.
    """
    print("\n[16] launch-file env knobs cross into the container")
    lf = open(os.path.join(PKG, "launch", "mission.launch.py")).read()
    rp = open(os.path.join(PKG, "scripts", "run_phase_a.sh")).read()
    # Match the tolerant readers too -- _envf/_envb exist precisely because an empty
    # forwarded var used to kill the launch, and a check that only knows os.environ.get
    # would go quietly blind the moment a knob is converted.
    names = sorted(set(re.findall(
        r'(?:os\.environ\.get|_envf|_envb)\(\s*"([A-Z0-9_]+)"', lf)))
    if not names:
        warn("no os.environ.get knobs found in mission.launch.py")
        return
    missing = [n for n in names if f"-e {n}=" not in rp]
    if missing:
        bad(f"{len(missing)} launch knob(s) never forwarded by DOCKER_ENV, so they "
            f"silently take their defaults: {missing}")
    else:
        ok(f"all {len(names)} launch env knob(s) forwarded")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--town", default="town05")
    ap.add_argument("--route", default=None, help="comma-separated region ids to verify")
    ap.add_argument("--campaign", default=None,
                    help="campaign under reports/runs to check for stale result dirs")
    ap.add_argument("--since", default=None,
                    help="epoch seconds; result dirs older than this are stale")
    a = ap.parse_args(argv)

    print("PREFLIGHT — checks for known silent failure modes")
    check_stale()
    check_flags()
    check_guidance()
    check_cue_observability()
    check_installed_config()
    check_configs(a.town)
    check_diagnosable()
    check_twins()
    check_prop_families()
    check_vlm_camera()
    check_terrain_camera()
    check_build_symlinks()
    check_cluster_space()
    check_carla_lock()
    check_env_forwarding()
    if a.route:
        check_route(a.route, a.town.capitalize().replace("town", "Town"))
    if a.campaign and a.since:
        check_freshness(a.campaign, float(a.since))

    print()
    if FAILS:
        print(f"\033[31m{len(FAILS)} FAILED\033[0m, {len(WARNS)} warnings")
        for f in FAILS: print(f"   - {f}")
        return 1
    print(f"\033[32mall checks passed\033[0m ({len(WARNS)} warnings)")
    for w in WARNS: print(f"   - {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

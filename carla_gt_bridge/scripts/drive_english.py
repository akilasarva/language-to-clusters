#!/usr/bin/env python3
"""English in, CARLA out: generate a plan from a sentence and drive it, in one command.

This connects the two halves: the planner turns English into a validated brain tree, and
`run_phase_a.sh` drives an arbitrary tree in CARLA via `PLAN_FILE=`. This script generates
the tree and hands it to run_phase_a.sh. At one control command per world tick the simulator runs faster than real
time, so a mission costs seconds of driving inside ~90 s of Docker bring-up.

GENERATION GOES THROUGH THE ABLATION'S OWN PATH (`stl_ablation.run_one`) rather than a
second copy of it, so the CARLA path and the offline ablation path cannot drift: `--arm`
compares arms when DRIVEN using exactly the offline generator. One generator, five prompts.

    # one sentence, driven
    python3 drive_english.py --english "turn right at the intersection" --start 0

    # the same sentence through a different arm
    python3 drive_english.py --english "..." --start 0 --arm flat

    # generate only, look at the plan before spending a CARLA run on it
    python3 drive_english.py --english "..." --start 0 --dry
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import time
import sys

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.dirname(PKG)
for _p in (PKG, os.path.join(PKG, "scripts"), os.path.join(WS, "nl_planner"),
           os.path.join(WS, "nl_planner", "scripts"), os.path.join(WS, "brain"),
           os.path.join(WS, "dgppo_ros_node_pkg")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: Where run_phase_a.sh's container sees the config directory. It must not be a host
#: path: load_mission_node runs INSIDE the bridge container and cannot see $PWD.
CFG_IN_CONTAINER = "/ros_ws/src/carla_gt_bridge/config"


# ONE SWITCH FOR THE MODE VOCABULARY, because it takes TWO files to change it and they must
# not drift: the cluster map declares the modes the generator may emit, the regions npz decides
# which of them any cluster can actually BE. Declaring a mode with no regions behind it is
# A branch that can never execute -- so `resolve_vocab` asserts every declared
# mode is present in the table's labels and refuses otherwise.
#
# Example: with the default `base` vocabulary (path, junction only) the generator can render
# "do not go on the sidewalk" as forbid_modes=['junction'], because `sidewalk` is not an
# available mode and the prompt instructs it to "pick the closest available one". The
# sidewalk vocabulary makes that mission expressible instead of silently wrong.
VOCABS = {
    "base":      ("cluster_map.carla_{town}.yaml",            "regions.{town}.npz"),
    "sidewalks": ("cluster_map.carla_{town}_sidewalks.yaml",  "regions.{town}.sidewalks.npz"),
    "approach":  ("cluster_map.carla_{town}.approach.yaml",   "regions.{town}.approach.npz"),
    "terrain":   ("cluster_map.carla_{town}_terrain.yaml",    "regions.{town}.sidewalks.npz"),
}


def resolve_vocab(vocab: str, town: str):
    """(cluster_map, regions_npz) for a named vocabulary, checked against each other."""
    import numpy as _np
    import yaml as _yaml
    try:
        cm, rz = VOCABS[vocab]
    except KeyError:
        sys.exit(f"--vocab {vocab!r} unknown; choose from {sorted(VOCABS)}")
    cmap = os.path.join(PKG, "config", cm.format(town=town))
    npz = os.path.join(PKG, "config", rz.format(town=town))
    for f in (cmap, npz):
        if not os.path.exists(f):
            sys.exit(f"--vocab {vocab}: missing {f}")
    declared = set((_yaml.safe_load(open(cmap)) or {}).get("modes") or [])
    have = set(_np.load(npz, allow_pickle=True)["labels"].tolist())
    orphan = declared - have - {"asphalt"}      # asphalt is a surface alias for path/junction
    if orphan:
        sys.exit(f"--vocab {vocab}: {sorted(orphan)} declared in {os.path.basename(cmap)} but "
                 f"NO region carries that label in {os.path.basename(npz)} "
                 f"(labels present: {sorted(have)}). A mode with no regions can never execute.")
    return cmap, npz


def generate(english: str, arm: str, town: str, model: str, retries: int, cmap: str):
    """English -> brain tree, through the ablation's generator so the arms stay comparable."""
    import stl_ablation as A
    from nl_planner.branch_materializer import to_brain_tree
    from nl_planner.schemas import NavPlan
    from nl_planner.taxonomy import load_taxonomy

    tax = load_taxonomy(cmap)
    r = A.run_one(english, arm, tax, model, retries)
    if not r.get("ok"):
        return None, r
    plan = NavPlan.model_validate(r["plan"])
    # MATERIALIZATION IS ITS OWN FAILURE MODE, and it is the one the ungrounded arms
    # hit: `untaxed` and `free` may invent mode names, so there is nothing to resolve them
    # to cluster ids and to_brain_tree raises. A plan that cannot be materialized cannot
    # be driven, so it is reported rather than crashing the run and looking like a
    # harness bug.
    try:
        # stl= only bites for arms that produce a formula; for the rest it is a no-op.
        tree = to_brain_tree(plan, tax, stl=r.get("stl") or None)
    except Exception as exc:                                       # noqa: BLE001
        r = dict(r, ok=False, stage="materialize",
                 reason=f"{type(exc).__name__}: {exc}")
        return None, r
    return tree, r


def spawn_args(town: str, start: int, toward: int | None, cmap: str, npz: str):
    import missions
    _, _, _, x, y, yaw = missions.spawn_seed(
        cluster_map=cmap, regions_npz=npz, start_region=start, start_toward=toward)
    return (f"--town {town} --region {start} "
            f"--x {x:.2f} --y {y:.2f} --yaw {yaw:.1f}")


def score_run(log_dir: str, tree: dict, started_at: float | None = None):
    """What the drive did, from the MPC JSONL -- the ROS log lines are throttled ~80x.

    `started_at` GUARDS AGAINST SCORING ANOTHER RUN. debug_logs is a single shared
    directory. When a run dies during bring-up it writes no log, and the newest file is
    then the PREVIOUS run's -- so taking the newest file unconditionally makes the failed
    run silently report the previous run's ticks, path and region sequence. Pass the
    wall-clock time the drive started and a log older than that is refused.
    """
    import numpy as np
    js = sorted(glob.glob(os.path.join(
        WS, "dgppo_ros_node_pkg", "dgppo_ros_node_pkg", "debug_logs",
        "carla_mpc_*.jsonl")), key=os.path.getmtime)
    if started_at is not None:
        js = [f for f in js if os.path.getmtime(f) >= started_at]
    rows = []
    if js:
        for line in open(js[-1]):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    # COMPLETION IS READ FROM WHICHEVER FILE ACTUALLY EXISTS. The harness redirects the
    # launch straight into `console.txt`; the container writes /tmp/mission.log inside
    # itself and it never lands here. Reading only `mission.log` would report "did not
    # finish" even for runs whose console carries `Navigation plan complete`.
    complete = 0
    for cand in ("mission.log", "console.txt"):
        f = os.path.join(log_dir, cand)
        if os.path.exists(f):
            complete = sum(1 for l in open(f, errors="ignore") if "plan complete" in l)
            if complete:
                break
    out = {"ticks": len(rows), "completed": complete > 0}
    if rows:
        xy = np.array([(r["x"], r["y"]) for r in rows if r.get("x") is not None])
        t = [r["t"] for r in rows if "t" in r]
        ts = [r["t_sim"] for r in rows if "t_sim" in r]
        seen = []
        for r in rows:
            c = r.get("gt_cluster")
            if c is not None and (not seen or seen[-1] != c):
                seen.append(c)
        out.update(
            regions=seen,
            path_m=round(float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))), 1)
            if len(xy) > 1 else 0.0,
            wall_s=round(max(t) - min(t), 1) if t else 0,
            sim_s=round(max(ts) - min(ts), 1) if ts else 0,
            jsonl=os.path.basename(js[-1]) if js else None)
    # PIN THE TRACE TO THE RUN. debug_logs is one shared directory and the association is
    # knowable only HERE, where `started_at` has already selected the file. Downstream,
    # `score_maneuvers._nearest_log` has to guess from mtimes within 600 s, which can hand
    # two closely spaced runs the same trace. Content cannot disambiguate either, because
    # clean missions drive the same corridor. So copy it.
    if js:
        try:
            shutil.copyfile(js[-1], os.path.join(log_dir, "run.jsonl"))
        except OSError:
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--english", required=True, help="the mission, in plain English")
    ap.add_argument("--start", type=int, required=True, help="start region id")
    ap.add_argument("--toward", type=int, default=None,
                    help="region to face at spawn (default: the corridor's first junction)")
    ap.add_argument("--arm", default="full",
                    choices=["full", "none", "flat", "untaxed", "free"],
                    help="which generator arm writes the plan")
    ap.add_argument("--town", default="town05")
    ap.add_argument("--vocab", default="base", choices=sorted(VOCABS),
                    help="which MODE vocabulary the generator may use. `base` is path/junction "
                         "only -- with it, a mission about surfaces ('do not go on the "
                         "sidewalk') is silently rendered into the nearest topological mode. "
                         "`sidewalks` adds sidewalk/asphalt and makes those missions "
                         "expressible. Sets the cluster map AND the regions npz together.")
    ap.add_argument("--cones", type=int, default=0, choices=[0, 1, 2],
                    help="how many cones, taken in order from cones.<town>.json")
    ap.add_argument("--prop", default=None,
                    help="CARLA blueprint to spawn at --cone-regions, e.g. "
                         "static.prop.bench01. A TERMINAL prop is what makes a completion "
                         "claim checkable: 'stop at the bench after the third "
                         "intersection' can be verified; 'stop at the far end' cannot.")
    ap.add_argument("--cone-regions", default=None,
                    help="place cones at THESE region ids (comma-separated), overriding "
                         "--cones. A matched pair that moves the cone between two named "
                         "junctions needs this: --cones only controls HOW MANY, in a "
                         "fixed order, so it cannot express 'the cone is at 60, not 63'.")
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--run-seconds", type=int, default=45)
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--dry", action="store_true",
                    help="generate and print the plan; do not touch CARLA")
    ap.add_argument("--cue-place-first", action="store_true",
                    help="publish cue answers place-key-first -- the ordering that made "
                         "an ambiguous branch cue answer the wrong predicate. For "
                         "demonstrating that the ordering matters.")
    ap.add_argument("--plan", default=None,
                    help="drive an EXISTING plan json instead of generating one. "
                         "REQUIRED for a matched pair -- the pair isolates the world, so "
                         "the plan has to be identical across both runs.")
    a = ap.parse_args()
    # resolved ONCE and passed down, so the generator's mode list and the spawner's region
    # table can never come from different vocabularies.
    CMAP, NPZ = resolve_vocab(a.vocab, a.town)

    print(f'mission : "{a.english}"')
    print(f"arm     : {a.arm}   start region {a.start}   town {a.town}   cones {a.cones}\n")

    if a.plan:
        # A MATCHED PAIR MUST REUSE ONE PLAN. Generating per invocation varies the plan
        # AND the world at once, so the pair stops isolating the cone: the absent and
        # present runs could be given different plans for the same sentence.
        tree = json.load(open(a.plan))
        rec = {"attempts": 0, "stl": tree.get("stl_formula"), "reused": a.plan}
        print(f"  reusing plan {os.path.basename(a.plan)} (not regenerating)")
    else:
        tree, rec = generate(a.english, a.arm, a.town, a.model, a.retries, CMAP)
    if tree is None:
        stage = rec.get("stage", "generate")
        # run_one returns `error`, not `reason`; reading only `reason` prints an empty
        # string and a bare gate list with no way to see WHICH rule was violated. Read
        # both, and never let a diagnosis depend on a key spelling.
        why = rec.get("error") or rec.get("reason") or "(no message captured)"
        print(f"  {stage.upper()} FAILED after {rec.get('attempts')} attempts: "
              f"gates={rec.get('gates')}\n    last error: {why}")
        return 1
    steps = tree.get("steps") or []
    print(f"  generated in {rec['attempts']} attempt(s), {len(steps)} step(s)")
    for st in steps:
        print(f"    {st.get('start_mode')} -> {st.get('goal_mode')}"
              f"  cue={st.get('transition_cue')}  trigger={st.get('trigger')}"
              f"  goal_cluster={st.get('goal_cluster')}"
              + (f"  [{len(st['branches'])} branches]" if st.get("branches") else ""))
    if rec.get("stl"):
        print(f"    STL: {rec['stl']}")
    for f in ("forbid_clusters", "require_clusters"):
        if tree.get(f):
            print(f"    {f}: {tree[f]}")

    name = f"english.{a.town}.{a.arm}"
    host_path = os.path.join(PKG, "config", f"{name}.json")
    with open(host_path, "w") as fh:
        json.dump(tree, fh, indent=2)
    print(f"\n  plan -> config/{name}.json")
    if a.dry:
        print("  --dry: stopping before CARLA")
        return 0

    # NOT /tmp: it can be cleaned out from under long runs, losing results and generated
    # plans with the only symptom an empty directory.
    log_dir = a.log_dir or os.path.join(
        PKG, "reports", "runs", f"drive_english_{a.arm}")
    os.makedirs(log_dir, exist_ok=True)
    # CONE_REGIONS is what actually makes the cue answerable. gt_cue_node prefers the
    # region-scoped ground-truth answer ("the cone is where it was placed") and only falls
    # back to scanning /carla/actor_list, which does not list static props in every
    # bridge build. Without this the `present` run can report "0 cone actors known" and
    # answer cone=False -- a matched pair with no discriminator in it.
    cone_regions = ""
    if a.cone_regions:
        # explicit placement. Validated here rather than at run time: an unknown region
        # silently places no cone, and a pair with no cone in either world looks like a
        # plan that ignores the world.
        want = [int(x) for x in str(a.cone_regions).split(",") if x.strip()]
        # look in BOTH: cones.json holds the DECISION set that --cones indexes into,
        # props.json holds terminal markers. Keep terminals out of cones.json: appending
        # them changes what `--cones N` resolves to, spawning the cone away from the
        # decision junction so the cue answers False all run and the plan looks broken.
        known, poses = set(), {}
        for fn in (f"cones.{a.town}.json", f"props.{a.town}.json"):
            fp = os.path.join(PKG, "config", fn)
            if not os.path.exists(fp):
                continue
            blob = json.load(open(fp))
            for c in (blob.get("cones") or []) + (blob.get("props") or []):
                known.add(c["region"]); poses[c["region"]] = c
        missing = [r for r in want if r not in known]
        if missing:
            sys.exit(f"cones.{a.town}.json has no pose for region(s) {missing}; "
                     f"known: {sorted(known)}. Add them before using --cone-regions.")
        cone_regions = ",".join(str(r) for r in want)
    elif a.cones:
        cj = os.path.join(PKG, "config", f"cones.{a.town}.json")
        if os.path.exists(cj):
            regs = json.load(open(cj)).get("cone_regions") or []
            cone_regions = ",".join(str(r) for r in regs[:a.cones])

    env = dict(os.environ,
               CUE_PLACE_FIRST="true" if a.cue_place_first else "false",
               CONE_REGIONS=cone_regions,
               **({"PROP_TYPE": a.prop} if a.prop else {}),
               PLAN_FILE=f"{CFG_IN_CONTAINER}/{name}.json",
               LANE_SPAWN_ARGS=spawn_args(a.town, a.start, a.toward, CMAP, NPZ),
               TOWN=a.town.capitalize().replace("town", "Town"),
               # SYNC IS OVERRIDABLE, and cue_source=vlm needs it OFF.
               # With a camera attached and synchronous mode on, the stack comes up and
               # loads the plan, then produces ZERO mpc ticks. The deadlock is the one
               # run_phase_a.sh's header describes: the server waits for a vehicle
               # control command, the bridge waits for the camera to deliver, and the
               # camera needs a world tick. The same contention times out the prop
               # SpawnObject service after 60 s while the ego spawns fine.
               # Async free-runs at -fps 10, so nothing waits on anything.
               # Default stays "True" (synchronous).
               CONES=str(a.cones),
               SYNC=os.environ.get("SYNC", "True"),
               WAIT_FOR_CONTROL=os.environ.get("WAIT_FOR_CONTROL", "True"),
               # ABSOLUTE. run_phase_a.sh runs with its own cwd (the package dir), so a
               # RELATIVE log dir resolves one level too deep, silently putting
               # mission/bridge/twist/spawn.log in
               #   src/carla_gt_bridge/carla_gt_bridge/reports/runs/<run>/
               # instead of src/carla_gt_bridge/reports/runs/<run>/.
               RUN_SECONDS=str(a.run_seconds), LOG_DIR=os.path.abspath(log_dir))
    print(f"  driving in CARLA (~{a.run_seconds}s + bring-up)...")
    _t0 = time.time()
    # THE TIMEOUT MUST COVER THE LOCK WAIT, NOT JUST THE DRIVE. `run_phase_a.sh` blocks on
    # `flock -w $LOCK_WAIT` (default 1800 s) before it starts anything. A timeout shorter
    # than that kills a run queued behind another CARLA job WHILE STILL WAITING, and the
    # `TimeoutExpired` traceback reads as "the run hung" when the run had not begun.
    _lock_wait = int(os.environ.get("LOCK_WAIT", "1800"))
    _budget = _lock_wait + a.run_seconds + 300          # + bring-up and teardown
    console = os.path.join(log_dir, "console.txt")
    try:
        subprocess.run(["bash", os.path.join(PKG, "scripts", "run_phase_a.sh"), "-", "drive"],
                       cwd=PKG, env=env, stdout=open(console, "w"),
                       stderr=subprocess.STDOUT, timeout=_budget)
    except subprocess.TimeoutExpired:
        # Say WHICH of the two it was. run_phase_a echoes "[lock] held" the instant it
        # acquires, so its absence is proof the run never started.
        got_lock = False
        try:
            got_lock = "[lock] held" in open(console, errors="ignore").read()
        except OSError:
            pass
        if got_lock:
            print(f"  TIMED OUT after {_budget}s WHILE DRIVING (lock was acquired). "
                  f"The run itself hung.")
        else:
            print(f"  NEVER GOT THE CARLA LOCK within {_lock_wait}s — another session's "
                  f"run holds /tmp/carla_gt_bridge.lock. Nothing was driven; retry later.")
        return 3

    res = score_run(log_dir, tree, started_at=_t0)
    if res["ticks"] == 0:
        print("  NO MPC LOG WAS WRITTEN BY THIS RUN — it did not drive. "
              "Not falling back to an older log.")
    print(f"\n  RESULT  {'COMPLETED' if res['completed'] else 'did not finish'}")
    print(f"    ticks {res['ticks']}   wall {res.get('wall_s')}s   sim {res.get('sim_s')}s"
          f"   path {res.get('path_m')} m")
    print(f"    regions visited: {res.get('regions')}")
    print(f"    logs: {log_dir}")
    return 0 if res["completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

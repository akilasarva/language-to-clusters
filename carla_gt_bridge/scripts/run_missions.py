#!/usr/bin/env python3
"""Drive the missions in config/missions.yaml in CARLA, for one of two arms.

    python3 scripts/run_missions.py --arm ours --only M3 --reps 1
    python3 scripts/run_missions.py --arm llm  --only M3 --reps 1 --model gpt-5.6-terra
    python3 scripts/run_missions.py --arm ours --list          # show the runs, drive nothing

Arms
  ours  English -> plan tree (nl_planner generator, prompt GENERATOR_PROMPT) -> brain -> MPC.
        The plan is generated ONCE per mission (drive_english.py --dry) and frozen, so every
        world and repetition of a mission drives the same tree. --regenerate makes a new
        one each run instead.
  llm   the LLM is the executor: BRAIN_MODE=llm replaces brain with llm_brain_node, which
        asks the model for a decision at every region change. The MPC gets a neutral
        geometry-only plan (plan_geometry.neutral_plan), so no mission constraint leaks in.

Both arms use ground-truth perception (gt_cluster_node, gt_cue_node) and the same worlds,
spawns and props. Every run is driven through drive_english.py -> run_phase_a.sh, one at a
time (CARLA is single-tenant). Needs OPENAI_API_KEY.

Output: <out>/<arm>/<mission>/<world>_s<seed>/ (logs) and <out>/<arm>/results.jsonl, one
row per run with the raw outcome (result, regions visited, path length, ticks, branch).
Nothing is scored here.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

import yaml

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.dirname(PKG)
sys.path.insert(0, os.path.join(PKG, "scripts"))


def container_path(host_path: str) -> str:
    """run_phase_a.sh mounts the workspace at /ros_ws, with this repo at /ros_ws/src."""
    return "/ros_ws/src/" + os.path.relpath(os.path.abspath(host_path), WS)


def base_env(prompt: str) -> dict:
    e = dict(os.environ)
    e["PYTHONPATH"] = os.pathsep.join([WS, os.path.join(WS, "brain"),
                                       os.path.join(WS, "nl_planner"), e.get("PYTHONPATH", "")])
    e["GENERATOR_PROMPT"] = prompt
    e.setdefault("GUIDANCE", "region")
    e["WS"] = os.path.dirname(WS)
    return e


def drive(english: str, town: str, world: dict, plan: str | None, logdir: str,
          seed: int, env: dict, secs: int, dry: bool = False,
          model: str | None = None) -> dict:
    """One drive_english.py run; returns its raw outcome parsed from the log."""
    args = ["--english", english, "--start", str(world["start"]),
            "--toward", str(world["toward"]), "--town", town, "--arm", "full",
            "--run-seconds", str(secs), "--log-dir", logdir]
    props = world.get("props") or []
    args += ["--cone-regions", ",".join(map(str, props))] if props else ["--cones", "0"]
    if plan:
        args += ["--plan", plan]
    if dry:
        args += ["--dry"]
    if model:
        args += ["--model", model]
    e = dict(env)
    e["MPC_SEED"] = str(seed)
    if world.get("extra_landmarks"):
        e["EXTRA_LANDMARKS"] = world["extra_landmarks"]
    os.makedirs(logdir, exist_ok=True)
    out = os.path.join(logdir, "out.txt")
    with open(out, "w") as f:
        proc = subprocess.run(
            [sys.executable, os.path.join(PKG, "scripts", "drive_english.py")] + args,
            cwd=WS, env=e, stdout=f, stderr=subprocess.STDOUT, timeout=1800)
    text = open(out, errors="ignore").read()
    logs = "".join(open(os.path.join(logdir, x), errors="ignore").read()
                   for x in os.listdir(logdir) if x.endswith(".log"))

    def grab(pattern, default=None):
        m = re.search(pattern, text)
        return m.group(1) if m else default

    branches = re.findall(r"branch \d+ \('([^']*)'\)", logs)
    return dict(seed=seed, result=grab(r"RESULT\s+(\w+)", "NOSTART"),
                path_m=float(grab(r"path ([0-9.]+) m", 0) or 0),
                regions=grab(r"regions visited: (\[[^\]]*\])", "[]"),
                ticks=int(grab(r"ticks (\d+)", 0) or 0),
                branch=branches[-1] if branches else None,
                gen_failed=bool(re.search(r"\b[A-Z_]+ FAILED after", text)),
                exit_code=proc.returncode)


def frozen_plan(mission: dict, town: str, world: dict, out_dir: str, env: dict,
                model: str | None) -> str:
    """Generate the mission's plan once (drive_english --dry) and keep it."""
    dst = os.path.join(out_dir, "plan.json")
    if os.path.exists(dst):
        return dst
    staged = os.path.join(PKG, "config", f"english.{town}.full.json")
    if os.path.exists(staged):
        os.remove(staged)                      # never pick up a previous mission's plan
    r = drive(mission["en"], town, world, None, os.path.join(out_dir, "generate"), 0, env,
              0, dry=True, model=model)
    if r["gen_failed"] or not os.path.exists(staged):
        raise RuntimeError(f"{mission['id']}: plan generation failed; see "
                           f"{os.path.join(out_dir, 'generate', 'out.txt')}")
    os.replace(staged, dst)
    return dst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", required=True, choices=["ours", "llm"])
    ap.add_argument("--missions", default=os.path.join(PKG, "config", "missions.yaml"))
    ap.add_argument("--only", default="", help="comma-separated mission ids (default: all)")
    ap.add_argument("--worlds", default="", help="comma-separated world names (default: all)")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--secs", type=int, default=200, help="CARLA driving cap per run")
    ap.add_argument("--model", default=None,
                    help="ours: plan generator (default gpt-4.1); llm: driver (default gpt-5.6-terra)")
    ap.add_argument("--prompt", default="generator_deep",
                    help="ours: generator prompt name in nl_planner/prompts (GENERATOR_PROMPT)")
    ap.add_argument("--regenerate", action="store_true",
                    help="ours: generate a fresh plan for every run instead of once per mission")
    ap.add_argument("--out", default=None, help="default: carla_gt_bridge/reports/missions/<stamp>")
    ap.add_argument("--list", action="store_true", help="print the runs and exit")
    a = ap.parse_args(argv)

    prompt_file = os.path.join(WS, "nl_planner", "nl_planner", "prompts", f"{a.prompt}.md")
    if a.arm == "ours" and not a.list and not os.path.exists(prompt_file):
        print(f"generator prompt {a.prompt!r} not found at {prompt_file}; "
              f"add it or pass --prompt <name> (e.g. --prompt generator)")
        return 2
    doc = yaml.safe_load(open(a.missions))
    town = doc.get("town", "town05")
    only = {x for x in a.only.split(",") if x}
    wanted = {x for x in a.worlds.split(",") if x}
    runs = [(m, w) for m in doc["missions"] for w in m.get("worlds") or []
            if (not only or m["id"] in only) and (not wanted or w["name"] in wanted)]
    if a.list or not runs:
        for m, w in runs:
            print(f"{m['id']:3s} {w['name']:12s} start={w['start']} toward={w['toward']} "
                  f"props={w.get('props') or []} {w.get('extra_landmarks') or ''}")
        if not runs:
            print("no runs selected")
        return 0 if runs else 1

    out = a.out or os.path.join(PKG, "reports", "missions", time.strftime("%Y%m%d_%H%M%S"))
    arm_dir = os.path.join(out, a.arm)
    os.makedirs(arm_dir, exist_ok=True)
    env = base_env(a.prompt)
    if a.arm == "llm":
        from plan_geometry import neutral_plan
        env.update(BRAIN_MODE="llm", LLM_MODEL=a.model or "gpt-5.6-terra", DECIDE_V_MAX="0.5")
    print(f"arm={a.arm} prompt={a.prompt if a.arm == 'ours' else '-'} "
          f"model={a.model or 'default'} runs={len(runs)}x{a.reps} -> {arm_dir}", flush=True)

    results = os.path.join(arm_dir, "results.jsonl")
    for m, w in runs:
        mdir = os.path.join(arm_dir, m["id"])
        os.makedirs(mdir, exist_ok=True)
        run_env = dict(env)
        if a.arm == "llm":
            plan = os.path.join(mdir, f"{w['name']}.neutral_plan.json")
            json.dump(neutral_plan(town, w["start"]), open(plan, "w"), indent=1)
            mission_file = os.path.join(mdir, f"{w['name']}.mission.txt")
            open(mission_file, "w").write(m["en"])
            run_env["LLM_MISSION_FILE"] = container_path(mission_file)
        for seed in range(a.reps):
            if a.arm == "ours":
                plan = (None if a.regenerate
                        else frozen_plan(m, town, w, mdir, run_env, a.model))
            logdir = os.path.join(mdir, f"{w['name']}_s{seed}")
            r = drive(m["en"], town, w, plan, logdir, seed, run_env, a.secs,
                      model=a.model if a.arm == "ours" else None)
            row = {"arm": a.arm, "mission": m["id"], "world": w["name"], "plan": plan,
                   "model": a.model, "prompt": a.prompt if a.arm == "ours" else None, **r}
            with open(results, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{m['id']} {w['name']} seed {seed}: {r['result']} regions {r['regions']}",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

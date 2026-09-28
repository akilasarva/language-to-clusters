#!/usr/bin/env python3
"""Baseline: the same LLM asked for a bare sequence of regions. No plan, no cues.

This isolates the contribution of the plan REPRESENTATION: identical model, identical map,
identical mission text, and the only thing removed is the structure (branches, constraints,
cues). Unlike an external baseline (CLIP-Nav, VLMaps), it requires no reimplementation of
another method.

WHAT IT CANNOT DO, by design and not by accident:
  - branch. A sequence has one path, so a contingency has nowhere to live.
  - carry a constraint. There is no field for "never enter X".
  - respond to a cue. Regions are named up front; nothing is decided at run time.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(WS, "nl_planner"), os.path.join(WS, "nl_planner", "scripts"),
           os.path.dirname(os.path.abspath(__file__))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PROMPT = """You are routing a robot through a town described as a graph of regions.

Each region has an id and a kind ({kinds}). The robot starts in region {start}.

REGIONS AND THEIR CONNECTIONS:
{graph}

MISSION: {mission}

Reply with ONLY a JSON array of region ids, in the order the robot should visit them,
starting with {start}. No prose, no explanation, no other keys.
Example: [0, 63, 1, 60]
"""


def build(mission: str, town: str, start: int, model: str = "gpt-4.1") -> dict:
    from town_graph import TownGraph
    # The RAW openai client, which is what stl_ablation.py uses -- pydantic_ai is not
    # installed in the interpreter that has the rest of the deps, and more importantly a
    # schema-bound Agent would defeat the point: this baseline is meant to get NO schema,
    # since the structure is exactly what is being ablated.
    import openai
    g = TownGraph(town)
    lines = []
    for r in g.rids:
        ns = sorted(g.adj.get(r, []))
        lines.append(f"  {r} ({g.label(r)}) -> {ns}")
    prompt = PROMPT.format(kinds=", ".join(sorted({g.label(r) for r in g.rids})),
                           start=start, graph="\n".join(lines), mission=mission)
    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system",
                   "content": "You output only a JSON array of integers. Nothing else."},
                  {"role": "user", "content": prompt}],
    )
    raw = resp.choices[0].message.content or ""
    m = re.search(r"\[[\d,\s]+\]", raw or "")
    if not m:
        return {"ok": False, "reason": "no JSON array in the reply", "raw": (raw or "")[:200]}
    seq = json.loads(m.group(0))
    bad = [r for r in seq if r not in g.rids]
    if bad:
        return {"ok": False, "reason": f"names regions not in {town}: {bad}", "route": seq}
    # a sequence is only executable if consecutive regions are actually adjacent
    breaks = [(a, b) for a, b in zip(seq, seq[1:]) if b not in g.adj.get(a, ())]
    return {"ok": not breaks, "route": seq,
            "reason": "" if not breaks else f"non-adjacent hops: {breaks}",
            "n_regions": len(seq), "disconnected_hops": len(breaks)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--english", required=True)
    ap.add_argument("--town", default="town05")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--model", default="gpt-4.1")
    a = ap.parse_args()
    r = build(a.english, a.town, a.start, a.model)
    print(json.dumps(r, indent=2))
    return 0 if r.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Refuse a plan whose landmark cue can NEVER fire under the ORACLE, before driving it.

WHY. A step such as

    path -> junction | trigger=landmark | cue=Detect(Fountain)

requires, at one instant, BOTH (a) standing in a region labelled `junction` and
(b) the fountain cue answering True. Under `CUE_SOURCE=topic` the oracle answers landmarks by
REGION CONTAINMENT (`gt_cue_node._landmark_near`: `self._cluster in mine`), so (b) means
"the current region is one of the regions that prop occupies". If the fountain's only pose
is in a region labelled `path`, the conjunction is unsatisfiable and the vehicle searches on
step 0 until the clock runs out. That is decidable on paper in milliseconds, so check it
before driving.

THIS IS SCOPED TO THE ORACLE ON PURPOSE. Under `CUE_SOURCE=vlm` detection has RANGE -- a
camera at a junction can see a fountain tens of metres down the road -- so the same plan is
satisfiable and must NOT be refused. The check therefore takes the cue source and says
nothing when it is `vlm`.
"""
import argparse, glob, json, os, sys

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pose_regions(town: str) -> dict:
    """family -> set of regions where that family actually has a pose/actor."""
    out: dict[str, set] = {}
    for fn in (f"cones.{town}.json", f"props.{town}.json", f"landmarks.{town}.json"):
        fp = os.path.join(PKG, "config", fn)
        if not os.path.exists(fp):
            continue
        blob = json.load(open(fp))
        for key, rows in blob.items():
            if not isinstance(rows, list):
                continue
            for r in rows:
                if not isinstance(r, dict) or "region" in r is None:
                    continue
                fam = ("cone" if key == "cones" else
                       _fam_of(r.get("type", "")) if key == "props" else key)
                if fam and r.get("region") is not None:
                    out.setdefault(fam, set()).add(int(r["region"]))
    return out


def _fam_of(bp: str) -> str:
    bp = (bp or "").lower()
    for fam in ("fountain", "bench", "kiosk", "barrier", "mailbox", "haybale",
                "trash", "vending", "advertisement"):
        if fam in bp:
            return "trash_can" if fam == "trash" else (
                "vending_machine" if fam == "vending" else fam)
    return ""


def cue_family(cue: str) -> str:
    """Which vocabulary family this cue names, USING THE REAL VOCABULARY.

    Matching is driven off `CARLA_GT_VOCAB`, not a hardcoded list: a hardcoded list returns
    "" for any family it does not contain, and that cue is then SILENTLY SKIPPED (e.g. a
    `Detect(CarPark)` step would pass although `car_park` has no regions in any table).
    Driving the match off the vocabulary means a family can never be unknown to this script
    while being known to the generator.
    """
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(PKG), "nl_planner"))
        from nl_planner.cue_vocab import CARLA_GT_VOCAB, families_matching
    except Exception:                                            # noqa: BLE001
        return ""
    fams = families_matching(cue or "", CARLA_GT_VOCAB)
    # `blocked` is aliased to the cone answer and `junction` is topological, not a placed
    # thing; neither is checked for a pose.
    fams = [f for f in fams if f not in ("blocked", "junction")]
    return fams[0] if len(fams) == 1 else ""


def walk(steps):
    for s in steps or []:
        yield s
        for b in (s.get("branches") or []):
            yield from walk(b.get("sub_plan"))


def check(plan: dict, labels: dict, poses: dict) -> list:
    bad = []
    for s in walk(plan.get("steps")):
        if s.get("trigger") != "landmark":
            continue
        fam = cue_family(s.get("transition_cue"))
        if not fam:
            continue
        where = poses.get(fam)
        if not where:
            bad.append(f"step {s.get('step')}: cue {s.get('transition_cue')!r} names {fam!r}, "
                       f"which has NO POSE in this town -- it can never answer True.")
            continue
        goal = s.get("goal_mode")
        got = {labels.get(r) for r in where}
        if goal and goal not in got:
            bad.append(
                f"step {s.get('step')}: needs goal_mode={goal!r} AND {s.get('transition_cue')!r} "
                f"at the same instant, but {fam!r} only exists in region(s) {sorted(where)} "
                f"labelled {sorted(x for x in got if x)} -- never {goal!r}. UNSATISFIABLE under "
                f"the oracle (containment); needs CUE_SOURCE=vlm, where detection has range.")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--town", default="town05")
    ap.add_argument("--cue-source", default=os.environ.get("CUE_SOURCE", "topic"))
    a = ap.parse_args()
    if a.cue_source.lower() == "vlm":
        print("cue_source=vlm: detection has range, nothing to check")
        return 0
    plan = json.load(open(a.plan))
    labels = {int(k): v for k, v in (plan.get("cluster_labels") or {}).items()}
    if not labels:
        print("plan carries no cluster_labels; cannot check", file=sys.stderr)
        return 0
    bad = check(plan, labels, pose_regions(a.town))
    for b in bad:
        print(f"UNSATISFIABLE  {b}")
    print(f"{len(bad)} unsatisfiable landmark step(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

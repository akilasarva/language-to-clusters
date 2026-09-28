#!/usr/bin/env python3
"""The LLM as the executor, driven in CARLA. Replaces brain_controller when
BRAIN_MODE=llm (mission.launch.py); the default is unchanged.

Same inputs the plan-tree brain gets: /predicted_cluster (ground-truth region, gt_cluster_node) and
/cue/confirmations (ground-truth cue answers, gt_cue_node). Same output: /brain/state, the JSON the
MPC steers from (goal_label + a Bearing(...) transition_cue + complete). No plan, no translation.

On every region change the node writes a one-sentence observation (containment visibility),
sends mission + history + sentence to the LLM, and acts:
  road          CONTINUE -> keep driving to the next junction;  STOP -> plan complete
  intersection  LEFT/STRAIGHT/RIGHT -> publish that manoeuvre;   STOP -> plan complete
While a junction decision is pending the node publishes state=DECIDING, which makes the MPC creep
(decide_v_max) and pin the turn to the junction the decision started at -- the mechanism that already
exists for slow VLM decisions -- so LLM latency does not turn into overshoot.

Env: LLM_MISSION_FILE (the English, one file), LLM_MODEL (default gpt-5.6-terra). Every call is
spend-capped and logged by `carla_gt_bridge.llm_budget` (LLM_CAP_USD, LLM_LEDGER_PATH). Run as a plain
script from the launch file."""
import json, os, sys, threading
import numpy as np, yaml
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16, String

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PKG)                            # parent of the carla_gt_bridge package
from carla_gt_bridge.routing import adjacency_from_bearing_map, maneuver_toward      # noqa: E402
from carla_gt_bridge.cue_answers import LANDMARK_SPELLINGS                           # noqa: E402

# ---- prompt text ------------------------------------------------------------------------------------
SYS = ("You are driving a robot car through a road network, one decision at a time, following a mission "
       "instruction exactly. You are told what you can perceive at each point. On a road segment you may "
       "CONTINUE (drive on to the next intersection) or STOP. At an intersection you may choose one of the "
       "listed exits (LEFT, STRAIGHT, RIGHT) or STOP inside the intersection. 'Stop just after' an "
       "intersection or 'right after a turn' means STOP on the road segment you reach next. Ignore clauses "
       "about transient things such as pedestrians. Reply ONLY with JSON {\"action\": \"...\"}.")
PHRASE = {"cone": "a traffic cone in the middle of the intersection", "traffic_light": "traffic lights",
          "stop_sign": "a stop sign", "bench": "a bench", "fountain": "a fountain", "overpass": "an overpass above the road",
          "car_park": "a car park beside the road"}
def describe(fams):
    fams = [f for f in sorted(fams) if f != "blocked"]
    return ", ".join(PHRASE[f] for f in fams) if fams else "nothing notable"
def obs_text(o):
    if o["kind"] == "road":
        return "You are on a road segment between intersections. Here you can see: " + describe(o["here"]) + "."
    s = "You have entered an intersection. Available exits: " + ", ".join(e.upper() for e in o["exits"]) + "."
    s += " In this intersection you can see: " + describe(o["here"]) + "."
    if "blocked" in o["here"]: s += " The intersection is blocked: there is a traffic cone in the middle of it."
    return s
def valid_actions(o):
    return ["continue", "stop"] if o["kind"] == "road" else list(o["exits"]) + ["stop"]
# -----------------------------------------------------------------------------------------------------

FAMILIES = ("cone", "traffic_light", "stop_sign", "bench", "fountain", "overpass", "car_park")


def families_from_answers(answers: dict, kind: str) -> set:
    """gt_cue_node publishes {spelling: bool}; a family is present when any of its spellings is True.
    Traffic lights only count at an intersection, as in the text world."""
    low = {str(k).lower(): bool(v) for k, v in answers.items()}
    out = set()
    for fam in FAMILIES:
        spells = [s.lower() for s in (LANDMARK_SPELLINGS.get(fam) or ())]
        if any(low.get(s) for s in spells):
            out.add(fam)
    if "cone" in out and kind == "junction": out.add("blocked")
    if kind != "junction": out.discard("traffic_light"); out.discard("cone")
    return out


class LlmBrain(Node):
    def __init__(self):
        super().__init__("llm_brain")
        cfg = os.path.join(PKG, "config")
        self.bm = yaml.safe_load(open(os.path.join(cfg, "cluster_map.carla_town05.yaml")))["bearing_map"]
        self.adj = adjacency_from_bearing_map(self.bm)
        z = np.load(os.path.join(cfg, "regions.town05.npz"))
        self.kind = {int(r): str(l) for r, l in zip(z["rids"], z["labels"])}
        self.mission = open(os.environ["LLM_MISSION_FILE"]).read().strip()
        self.model = os.environ.get("LLM_MODEL", "") or "gpt-5.6-terra"      # forwarded ${VAR:-} is ""
        from carla_gt_bridge import llm_budget as budget       # noqa: E402  spend-capped client
        self.budget = budget
        self.cur = self.prev = None
        self.answers = {}
        self.hist = []
        self.step = 0
        self.state, self.goal_label, self.cue, self.complete = "NAVIGATING", "junction", None, False
        self.busy = False
        self.pub = self.create_publisher(String, "/brain/state", 10)
        self.create_subscription(Int16, "/predicted_cluster", self._cluster_cb, 10)
        self.create_subscription(String, "/cue/confirmations", self._cue_cb, 10)
        self.create_timer(0.2, self._publish)
        print(f"[LLM_BRAIN] model={self.model} mission={self.mission!r}", flush=True)

    def _cue_cb(self, msg):
        try: self.answers = json.loads(msg.data)
        except Exception: pass                                   # noqa: BLE001

    def _cluster_cb(self, msg):
        r = int(msg.data)
        if r < 0 or r == self.cur or self.complete: return
        self.prev, self.cur = self.cur, r
        kind = "junction" if self.kind.get(r) == "junction" else "road"
        if kind == "junction":
            ex = {}
            if self.prev is not None:
                for b in self.adj.get(r, set()) - {self.prev}:
                    m = maneuver_toward(self.bm, r, self.prev, b)
                    if m in ("left", "straight", "right"): ex[m] = b
            obs = {"kind": "junction", "exits": sorted(ex)}
            self.state, self.goal_label, self.cue = "DECIDING", "path", None      # creep + pin the junction
        else:
            obs = {"kind": "road"}
        self._publish()
        threading.Thread(target=self._decide, args=(obs, r), daemon=True).start()

    def _decide(self, obs, region):
        # gt_cue_node recomputes its answers on the same region change, so give it a moment: reading
        # self.answers immediately would describe the PREVIOUS region's cues.
        import time as _t
        _t.sleep(0.4)
        if self.cur != region: return                       # already moved on; the newer region decides
        obs["here"] = families_from_answers(self.answers, obs["kind"])
        va = valid_actions(obs)
        self.hist.append(f"Observation {len(self.hist)+1}: {obs_text(obs)} Valid actions now: " + ", ".join(x.upper() for x in va) + ".")
        base = [{"role": "system", "content": SYS},
                {"role": "user", "content": f"Mission: {self.mission}\n\nHistory so far (oldest first):\n" + "\n".join(self.hist) + "\n\nWhat is your action now?"}]
        msg, a = list(base), "stop"
        for _ in range(3):
            try:
                txt, _u = self.budget.call(self.model, msg, f"driven_llm:{self.model}", est_in=2500, max_out=3000,
                                           response_format={"type": "json_object"})
            except Exception as e:                                                # noqa: BLE001
                print(f"[LLM_BRAIN] call failed: {type(e).__name__}: {e}", flush=True); a = "stop"; break
            try: a = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])["action"].strip().lower()
            except Exception: a = "?"                                             # noqa: BLE001
            if a in va: break
            msg = base + [{"role": "assistant", "content": txt},
                          {"role": "user", "content": f"{a.upper()} is not available. Choose one of: " + ", ".join(x.upper() for x in va) + "."}]
        self.hist[-1] += f" -> you chose {a.upper()}"
        print(f"[LLM_BRAIN] region {region} {obs['kind']} {sorted(obs['here'])} exits={obs.get('exits')} -> {a.upper()}", flush=True)
        if a == "stop" or a not in va:
            self.complete, self.state = True, "COMPLETE"
            print("[LLM_BRAIN] Navigation plan complete", flush=True)
        elif obs["kind"] == "junction":
            self.step += 1
            self.state, self.goal_label = "NAVIGATING", "path"
            self.cue = f"Bearing({a.capitalize()}) completed"
        else:
            self.step += 1
            self.state, self.goal_label, self.cue = "NAVIGATING", "junction", None
        self._publish()

    def _publish(self):
        p = {"state": self.state, "current_cluster": self.cur, "plan_name": "llm_executor",
             "step": self.step, "branch_path": [], "complete": self.complete,
             "start_cluster": self.cur, "goal_cluster": 1000 + self.step, "goal_label": self.goal_label,
             "start_label": self.kind.get(self.cur, "unknown") if self.cur is not None else "unknown",
             "transition_cue": self.cue, "trigger": "llm"}
        self.pub.publish(String(data=json.dumps(p)))


def main():
    rclpy.init()
    n = LlmBrain()
    try: rclpy.spin(n)
    finally:
        n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()

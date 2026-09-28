#!/usr/bin/env python3
"""Plan-generation harness with ablation arms over the STL intermediate representation.

Also the plan-generation entry point used by `drive_english.py` (via `run_one`).

The generator emits three things in one call: a filtered command, a JSON plan tree, and
an STL formula. Only the JSON reaches the executor — `branch_materializer` contains no
reference to the formula, so the STL is consumed by the syntax check and the tripartite
verifier and then discarded.

Arms (conditions):

  flat      NO plan tree at all -- a linear list of steps, `branches` always null.
            The control for plan STRUCTURE: `full` and `none` both carry full
            structure, so neither alone says whether structure buys anything.
  full_tri  STL taught, emitted, syntax-checked AND tripartite-verified (production)
  full      STL taught, emitted, syntax-checked only
  emit      STL taught and emitted, NEITHER STL check runs
  none      STL never taught (generator_nostl.md), no formula, no checks

  full_tri vs full  isolates the TRIPARTITE check specifically — cross-reading
                    English/STL/JSON is the only gate that can see a SEMANTIC mismatch,
                    and it is the one place the formula does work no JSON validator can.
  full vs emit      isolates the syntax gate
  emit vs none   isolates the REPRESENTATION — does writing a formula improve the plan
                 even when nothing reads it? (a chain-of-thought effect)
  full vs none   with and without STL
  none vs flat   what does the plan tree itself buy?

EVERY CONDITION RUNS THE SAME JSON VALIDATORS — schema, modes, transitions, decision
cues. Those operate on the plan tree and are not part of the STL contribution; omitting
them from `none` would credit STL with catching what plain schema validation catches.

TWO DIFFERENT THINGS GET CALLED A TIER, so this file uses neither word loosely:

  PLAN SHAPE (`plan_shape`)  what the arm EMITTED -- LINEAR / BRANCHED / NESTED.
                             Tautological for arms forbidden to branch.
  MISSION TIER (elsewhere)   how hard the MISSION is -- EASY means a direct prompt
                             already handles it.

Do not confuse the two: a `flat` arm producing linear plans is not producing easy missions.
Report per mission tier, not only the aggregate, since a large EASY population can hide
differences on HARD missions.

Usage:
    python3 scripts/stl_ablation.py --conditions full none --trials 1
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
# SAMPLING TEMPERATURE. The provider default (1.0) makes plans non-reproducible between runs;
# the variation is mostly in prose and STL rendering rather than the executable plan tree, so
# sampling buys little behaviourally and costs artifact provenance. Default 0; override with
# NL_PLANNER_TEMPERATURE to restore sampling for eval diversity.
LLM_TEMPERATURE = float(os.environ.get("NL_PLANNER_TEMPERATURE", "0"))

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)
WS = os.path.dirname(PKG)
sys.path.insert(0, os.path.join(WS, "brain"))

TASKS = None   # a task YAML must be passed with --tasks
CONDITIONS = ("full_tri", "full", "twostage", "twostage_gated", "emit", "none",
              "flat", "untaxed", "free")

#: Arms whose output includes an STL formula, and which therefore get the STL
#: syntax and mode gates. Membership here is what makes validity comparable.
FORMULA_ARMS = ("full", "full_tri", "twostage", "twostage_gated")

#: Arms that get NO closed mode vocabulary: no list in the user message, no prompt
#: section teaching one, and `validate_plan_modes` skipped so there is no retry loop
#: teaching it either. The model invents place-names from the English.
#:
#: These exist because `flat` alone confounds two things. It removes branching but
#: keeps the taxonomy, so a flat-vs-tree gap could have been structure or grounding.
#: With these the arms form a 2x2 and each dimension is isolated:
#:
#:              ungrounded      grounded
#:   linear     free            flat
#:   tree       untaxed         none        (+ full = none + STL)
#:
#:   free vs flat, untaxed vs none  ->  what GROUNDING buys
#:   free vs untaxed, flat vs none  ->  what STRUCTURE buys
UNTAXED: frozenset[str] = frozenset({"untaxed", "free"})


def _steps(plan):
    return (plan.get("steps") or []) if isinstance(plan, dict) else (plan or [])


def _depth(steps, d=0):
    best = d
    for s in steps or []:
        if isinstance(s, dict):
            for br in (s.get("branches") or []):
                if isinstance(br, dict):
                    best = max(best, _depth(br.get("sub_plan"), d + 1))
    return best


def _ordinal(steps):
    for s in steps or []:
        if not isinstance(s, dict):
            continue
        if (s.get("cue_ordinal") or 1) > 1:
            return True
        for br in (s.get("branches") or []):
            if isinstance(br, dict) and _ordinal(br.get("sub_plan")):
                return True
    return False


def plan_shape(plan) -> str:
    """The SHAPE OF THE PLAN THIS ARM EMITTED -- not how hard the mission is.

    Not to be confused with the MISSION TIERS (where EASY means "a direct prompt already
    handles it"): a `flat` arm produces linear plans, not easy missions.

    It is near-tautological for some arms and must be read that way: `flat` and
    `free` are forbidden to branch, so they are LINEAR by construction and the column
    says nothing about them. It is informative only ACROSS arms allowed to branch, or
    against what the mission actually demanded -- which is what `faithful` measures.
    """
    st = _steps(plan)
    d, o = _depth(st), _ordinal(st)
    if d >= 2 or (d == 1 and o):
        return "NESTED"
    if d == 1 or o:
        return "BRANCHED"
    return "LINEAR"


def _tripartite(client, model: str, english: str, stl: str, plan) -> tuple[bool, str]:
    """The X<->Y<->Z check, driven by the raw client.

    `pipeline.generate_plan` runs this through pydantic_ai, which may not be installed
    here, so the prompt is invoked directly. Same prompt file and same three-way
    question, so the gate is the deployed one.

    This is the ONLY gate that can catch a SEMANTIC mismatch — a plan that is perfectly
    well-formed and simply does not say what the English said. No JSON validator can see
    that, because there is nothing structurally wrong to see.
    """
    from nl_planner.prompts import load_prompt
    msg = ("[X] Original English Command\n" + english.strip() +
           "\n\n[Y] STL Formula\n" + (stl or "").strip() +
           "\n\n[Z] JSON Plan\n" + plan.model_dump_json(indent=2) + "\n")
    try:
        r = client.chat.completions.create(
            model=model, temperature=LLM_TEMPERATURE,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": load_prompt("verify_tripartite")},
                      {"role": "user", "content": msg}])
        v = json.loads(r.choices[0].message.content)
        ok = bool(v.get("english_stl_aligned") and v.get("stl_json_aligned")
                  and v.get("english_json_aligned"))
        return ok, str(v.get("notes", ""))[:160]
    except Exception as exc:                                       # noqa: BLE001
        # A verifier that errors must not silently pass the plan it was meant to check.
        return False, f"verifier exception: {type(exc).__name__}: {exc}"[:160]


def _leaf_paths(plan: dict) -> int:
    """How many distinct executions this plan admits.

    THE CONTINGENCY ANALOGUE OF A DROPPED CONSTRAINT. The `flat` arm produces perfectly
    valid plans for "if the gate is shut go around" -- it just commits to one course and
    drops the other. Plan validity cannot see that, exactly as it could not see a dropped
    `forbid_modes`. A single leaf path on a mission that states a contingency IS the
    failure, and it is the only thing distinguishing the flat arm from the tree arm.
    """
    try:
        from brain.plan_navigator import count_leaf_paths
        return count_leaf_paths(plan.get("steps") or [])
    except Exception:                                              # noqa: BLE001
        return 1


def _cluster_hints(steps, out=None) -> list:
    """Every goal cluster the plan mentions, in order, across all branches."""
    out = [] if out is None else out
    for st in steps or []:
        g = st.get("goal_cluster")
        if isinstance(g, int) and g not in out:
            out.append(g)
        for b in (st.get("branches") or []):
            _cluster_hints(b.get("sub_plan"), out)
    return out


def _discriminates(plan, tax) -> bool:
    """Does the plan's EXECUTION diverge depending on what it observes?

    Replays the plan twice -- every cue answered True, then every cue False -- and
    compares the branch path taken. Complements `_leaf_paths`, which counts branches
    that exist; this checks that a cue actually selects between them, so a decision step
    whose cue is never consulted shows up as 1 rather than 2.

    WHAT IT DOES NOT CATCH: two branches leading
    to the SAME destination still report True. The branch path differs, so the execution
    genuinely diverged -- but the contingency is cosmetic. Distinguishing that needs the
    destinations compared, which a trace-driven replay cannot do, since the observed
    regions are fixed by the trace. `missions.simulate` is what closes that loop.
    """
    try:
        import replay_monitors as R
        from nl_planner.branch_materializer import to_brain_tree
        # MATERIALIZE FIRST. A raw NavPlan dump carries modes, not cluster ids --
        # `goal_cluster` and `accept_clusters` are what to_brain_tree resolves them into,
        # and they are what the monitors compare against. Replaying the un-materialized
        # plan finds no clusters and silently returns False for every plan.
        tree = to_brain_tree(plan, tax)
        steps = tree.get("steps") or []
        clusters = _cluster_hints(steps)
        if not steps or not clusters:
            return False
        seq = [c for c in clusters for _ in range(4)]
        yes = R.replay(tree, [R.Obs(c, 0.0, True) for c in seq])
        no = R.replay(tree, [R.Obs(c, 0.0, False) for c in seq])
        # Compare the BRANCH PATH, not the regions: replay is trace-driven, so the
        # observed regions are fixed and a plan cannot change where the robot went --
        # only which of its own paths it walked. Comparing regions would report every
        # plan as non-discriminating, including genuinely branching ones.
        return yes.branch_path != no.branch_path
    except Exception:                                              # noqa: BLE001
        return False


def captured_constraints(plan: dict, stl: str) -> list[str]:
    """Which CONSTRAINTS survived, from either representation.

    WHY PLAN VALIDITY IS THE WRONG METRIC FOR A CONSTRAINT MISSION. "Follow the walkway
    to the plaza and stay on it the whole way" produces a perfectly valid plan if the
    invariant is simply dropped — it becomes a plain traversal, passes every structural
    validator, and a validity metric scores it OK even though it set NEITHER
    `forbid_modes` NOR `require_modes`.

    Counted from BOTH representations so the comparison is fair: a constraint stated as a
    JSON field counts exactly as much as one stated in the formula. Anything else would
    credit STL for expressiveness the JSON also has — `forbid_modes` and `require_modes`
    are real NavPlan fields.
    """
    out: list[str] = []
    for kind in ("forbid_modes", "require_modes"):
        for m in (plan.get(kind) or []):
            out.append(f"{kind[:6]}:{m}")
    if stl:
        try:
            from nl_planner.stl_compile import compile_monitors, parse
            spec = compile_monitors(parse(stl)[0])
            out += [f"forbid:{m}" for m in spec.forbid_modes]
            out += [f"requir:{m}" for m in spec.require_modes]
            out += [f"hold:{h['hold_mode']}" for h in spec.holds]
        except Exception:                                          # noqa: BLE001
            pass
    return sorted(set(out))


#: WHAT EACH ARM ISOLATES, for reports. The short keys stay as the on-disk `condition`
#: value so existing result files keep parsing; these are the display names. Each name
#: says what was REMOVED -- "flat" and "none" say nothing about which component is tested.
ARM_LABEL: dict[str, str] = {
    "full":           "ours (tree + formula)",
    "full_tri":       "ours + LLM verifier",
    "twostage":       "ours, formula written first",
    "twostage_gated": "ours, formula first + containment gate",
    "none":           "no formula (tree only)",
    "flat":           "no branching (linear, grounded)",
    "untaxed":        "no grounding (tree, open vocabulary)",
    "free":           "no structure (linear, open vocabulary)",
    "emit":           "emit-only (no validation)",
}


def arm_label(cond: str) -> str:
    return ARM_LABEL.get(cond, cond)


#: Which prompt each arm is generated from. `flat` is the structure control: `full` and
#: `none` BOTH carry full plan-tree structure, so neither speaks to whether structure
#: buys anything.
PROMPT_FOR: dict[str, str] = {
    "full_tri": "generator",
    "full": "generator",
    # TWO-STAGE. Stage 1 writes the formula alone (`generator_stl_only`); stage 2 is the
    # ORDINARY generator, given that formula in its user message. The system prompt is
    # therefore identical to `full`, so the only variable between the two arms is whether
    # the formula was written first and supplied -- which is the question the arm exists to
    # answer. Anything else (a bespoke stage-2 prompt) would confound it.
    "twostage": "generator",
    # Same two-stage generation, plus a CONTAINMENT gate: the materialised tree is
    # checked against its own formula and a violating leaf is fed back as a retry
    # message.
    "twostage_gated": "generator",
    "emit": "generator",
    "none": "generator_nostl",
    "flat": "generator_flat",
    "untaxed": "generator_untaxed",
    "free": "generator_free",
}


def run_one(text: str, cond: str, tax, model: str, retries: int) -> dict:
    import openai
    from nl_planner.prompts import load_prompt
    from nl_planner.schemas import GeneratorOutput, NavPlan
    from nl_planner.stl_syntax import quick_syntax_check
    from nl_planner.cue_vocab import CARLA_GT_VOCAB, validate_plan_cues
    from nl_planner.taxonomy import (repair_degenerate_branches,
                                     repair_place_detect_to_traverse,
                                     repair_prose_decision_cue,
                                     validate_decision_cues, validate_plan_modes,
                                     validate_plan_transitions)

    client = openai.OpenAI()
    # PROMPT A/B (`GENERATOR_PROMPT=<name>`). Compare prompts on the same code rather than
    # against older result files, since gates change between versions. Only the generator
    # prompt is swapped; the STL-only and verifier prompts are untouched, so a two-stage
    # arm stays comparable.
    # PROMPT RESOLUTION, AND WHY IT IS PRINTED. `GENERATOR_PROMPT` applies only to arms
    # whose prompt is the plain `generator`, so that a corpus A/B does not silently clobber
    # the specialised arms. Caveat: e.g. GENERATOR_PROMPT=generator_deep reaches `full` and
    # NOT `none`, so "with STL vs without STL" becomes "generator_deep vs generator_nostl".
    # The arm is its prompt, so the resolved name is always printed, and
    # GENERATOR_PROMPT_FORCE overrides EVERY arm for a deliberate one-variable pair.
    _force = os.environ.get("GENERATOR_PROMPT_FORCE")
    _ovr = os.environ.get("GENERATOR_PROMPT")
    _name = (_force or (_ovr if (_ovr and PROMPT_FOR[cond] == "generator")
                        else PROMPT_FOR[cond]))
    print(f"[gen] arm={cond} prompt={_name}.md"
          + ("  (FORCED)" if _force else ""), flush=True)
    prompt = load_prompt(_name)
    if cond in UNTAXED:
        base = f"Mission: {text}"
    else:
        base = (f"Mission: {text}\n\nAvailable semantic modes (use these VERBATIM for "
                f"start_mode/goal_mode):\n"
                + "\n".join(f"- {m}" for m in tax.modes_for_prompt()))
        # The axis line is what lets a SURFACE constraint land on the surface axis. Without
        # it the generator can put "stay off the sidewalk" on `path` (topology). Empty for
        # taxonomies that declare no axes, so those environments are byte-identical.
        _axes = tax.axes_note()
        if _axes:
            base += "\n\n" + _axes

    stage1_stl = ""
    # AN EMPTY STAGE-1 FORMULA IS NOT A STAGE-1 FAILURE, and conflating them is the same
    # bug as `Rule 0: empty STL formula` -- a correct answer charged as an error.
    # `generator_stl_only_v2.md` teaches that most missions state no constraint and the
    # empty formula is the right return. Treating it as a failure would charge a correct
    # stage 1 to the PLAN budget on every attempt and consume its entire retry allowance.
    stage1_ok = True
    if cond in ("twostage", "twostage_gated"):
        # Stage 1: the specification, alone. A failure here is NOT fatal -- the arm then
        # degrades to plain `full`, and the result records that it did, because silently
        # falling back would report the two-stage arm's number for a one-stage run.
        try:
            r1 = client.chat.completions.create(
                model=model, temperature=LLM_TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[{"role": "system",
                           "content": load_prompt("generator_stl_only")},
                          {"role": "user", "content": base}])
            stage1_stl = str(json.loads(r1.choices[0].message.content)
                             .get("stl_formula") or "")
        except Exception:                                          # noqa: BLE001
            stage1_stl, stage1_ok = "", False
        if stage1_stl:
            if os.environ.get("TWOSTAGE_V2") == "1":
                base += (
                    "\n\nA specification for this mission has already been written:\n\n"
                    f"    {stage1_stl}\n\n"
                    "Your plan must satisfy it, and you should normally return it "
                    "unchanged as your `stl_formula`. Make the `steps` realise it: every "
                    "branch it guards must exist, every ordinal alternation it counts must "
                    "appear as separate steps, and any mode it says must HOLD must be "
                    "carried on the step that holds it rather than merely being a "
                    "destination.\n\nYOU MAY REWRITE THE FORMULA, and must do so if it is "
                    "malformed — unbalanced brackets, an invented macro, a nesting depth "
                    "above three — or if it states a constraint the mission never made. "
                    "Preserve its MEANING; its bracketing is not sacred. If the mission "
                    "states no constraint at all, an empty `stl_formula` is correct even "
                    "though stage one wrote one.")
            else:
                base += (
                    "\n\nA specification for this mission has already been written:\n\n"
                    f"    {stage1_stl}\n\n"
                    "Your plan must satisfy it. Return this formula VERBATIM as your "
                    "`stl_formula` field, and make the `steps` realise it: every branch it "
                    "guards must exist, every ordinal alternation it counts must appear as "
                    "separate steps, and any mode it says must HOLD must be carried on the "
                    "step that holds it rather than merely being a destination.")
    msg, last, gates = base, None, []
    degenerate_repairs: list[str] = []

    # PER-TARGET RETRY BUDGETS (`PER_TARGET_BUDGET=1`).
    #
    # With a SHARED budget, two independent repair targets (plan and formula) contend, and
    # the model can oscillate between them (e.g. cue_vocab -> syntax -> cue_vocab) until
    # the budget runs out, even when each target is individually repairable.
    #
    # Under this flag the plan and the formula each get `retries` attempts of their own and
    # a failure on one does not consume the other's. The run stops when EITHER target is
    # exhausted, so a plan that can never be repaired still terminates.
    #
    # CONTROL: per-target 3+3 permits up to 6 calls, so compare it against a shared budget
    # of 6, not 3; otherwise the effect of simply buying more attempts is confounded.
    _per_target = os.environ.get("PER_TARGET_BUDGET") == "1"
    _PLAN_GATES = {"schema", "modes", "transitions", "cues", "cue_vocab",
                   "stage1_call_failed"}
    _STL_GATES = {"stl_modes", "syntax", "tripartite", "containment"}
    _budget = {"plan": retries, "stl": retries}

    def _fire(name: str) -> None:
        """Record a gate and charge it to its own target's budget."""
        gates.append(name)
        if name in _PLAN_GATES:
            _budget["plan"] -= 1
        elif name in _STL_GATES:
            _budget["stl"] -= 1

    def _exhausted(n: int) -> bool:
        if _per_target:
            return _budget["plan"] <= 0 or _budget["stl"] <= 0
        return n >= retries

    attempt = 0
    while not _exhausted(attempt):
        attempt += 1
        try:
            r = client.chat.completions.create(
                model=model, temperature=LLM_TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": prompt},
                          {"role": "user", "content": msg}])
            raw = json.loads(r.choices[0].message.content)
            # REPAIR BEFORE VALIDATION. A decision step whose only branch is `default`
            # decides nothing, and the schema refuses it -- but the refusal happens inside
            # model_validate, so the post-validation repairs cannot reach it. Common on deep
            # missions, where the real decision underneath the wrapper is correctly formed.
            _pl = raw.get("json_plan") if isinstance(raw, dict) else None
            _target = _pl if isinstance(_pl, dict) else raw
            _did = repair_degenerate_branches(_target)
            # Same class, same place: a prose cue on a branching step is unanswerable and
            # validate_decision_cues' own message says to null it.
            _did += repair_prose_decision_cue(_target)
            # Place nouns belong to the CLUSTER channel, not the visual one:
            # "approach the crossroads" is an arrival the region classifier
            # reports, not a sighting the VLM answers. Detect(Crossroads) on a
            # junction step would stall the step.
            _did += repair_place_detect_to_traverse(_target)
            if _did:
                degenerate_repairs.extend(_did)
            # Fires only when the stage-1 CALL failed, so the arm silently degrading to
            # one-stage generation is still recorded. A successful stage 1 that returned
            # `""` because the mission states no constraint is a correct answer.
            if cond in ("twostage", "twostage_gated") and not stage1_ok:
                _fire("stage1_call_failed")
            if cond in ("none", "flat") or cond in UNTAXED:
                plan = NavPlan.model_validate(raw.get("json_plan") or raw)
                stl = ""
            else:
                out = GeneratorOutput.model_validate(raw)
                plan, stl = out.json_plan, out.stl_formula
        except Exception as exc:                                   # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
            _fire("schema")
            # KEEP THE ARTIFACT. pydantic truncates the offending value ("input_value=
            # {'step': 2, 'description'...e, 'branches': None}]}]}"), so a schema failure
            # would leave nothing to inspect beyond the word "schema". Dumping the raw
            # output makes structural failures diagnosable.
            dbg = os.environ.get("SCHEMA_FAIL_DUMP")
            if dbg:
                try:
                    os.makedirs(dbg, exist_ok=True)
                    fn = os.path.join(dbg, f"raw_attempt{attempt}_{os.getpid()}.json")
                    with open(fn, "w") as fh:
                        json.dump(raw, fh, indent=1)
                except Exception:
                    pass
            msg = base + "\n\nVerifier feedback:\n" + str(exc)
            continue

        # An ungrounded arm has no vocabulary to validate against -- enforcing one
        # would BE the taxonomy, and the arm would no longer be the thing under test.
        bad = [] if cond in UNTAXED else validate_plan_modes(plan, tax)
        if bad:
            last = f"modes: {bad}"; _fire("modes")
            msg = base + f"\n\nVerifier feedback:\nTAXONOMY FAIL: {bad}"
            continue
        bad = validate_plan_transitions(plan)
        if bad:
            last = f"transitions: {bad}"; _fire("transitions")
            msg = base + "\n\nVerifier feedback:\nTRANSITION FAIL:\n" + "\n".join(bad)
            continue
        bad = validate_decision_cues(plan)
        if bad:
            last = f"cues: {bad}"; _fire("cues")
            msg = base + "\n\nVerifier feedback:\nCUE FAIL:\n" + "\n".join(bad)
            continue

        # THE VOCABULARY GATE. `validate_plan_cues` exists precisely to make "a plan naming
        # something the runtime cannot answer" a RETRYABLE generation error rather than a
        # silent execution failure. Examples of cues it rejects, each of which otherwise
        # looks like a planning or control failure:
        #   Detect(Crossroads)      brain sits at step 0 even when cue answers are correct
        #   way ahead is blocked    the robot waits at the junction indefinitely
        #   bench is visible        a later decision never fires
        # ADVISORY MODE (`CUE_VOCAB_ADVISORY=1`) records the same finding WITHOUT spending
        # a retry on it. The gate conflates two questions that should be separate: whether
        # the instruction was TRANSLATED correctly, and whether THIS deployment can sense
        # what it names. Touchdown names street furniture no CARLA deployment answers, so
        # on that corpus the gate measures the prop table rather than the translator. It is
        # also a major failure source under joint generation, because the retry budget is
        # shared: repairing a cue breaks the formula and vice versa.
        # Advisory mode is only defensible because the runtime now reports an unanswerable
        # cue explicitly rather than stalling silently -- three-valued predicates, not a
        # relaxation.
        # OPEN SET UNDER THE VLM. `check_cue`'s own docstring scopes this gate to the
        # deployment, not the language: "under cue_source=vlm a model answers free text
        # and nothing here applies -- pass vocab=None". Reading CARLA_GT_VOCAB
        # unconditionally would, with CUE_SOURCE=vlm, reject a perfectly answerable
        # open-set cue ("a red fire hydrant", "Detect(Gnome)") and burn a retry for a
        # constraint the runtime does not have.
        #
        # The closed vocabulary is CORRECT for cue_source=topic: there the publisher's
        # keys ARE the answerable set, and a cue outside it cannot be answered by
        # anything -- it times out silently.
        #
        # So the gate follows the SENSOR, and the value in force is logged, because a
        # vocabulary that is silently off is indistinguishable from one that passes.
        _cue_source = os.environ.get("CUE_SOURCE", "topic").strip().lower()
        _vocab = None if _cue_source == "vlm" else CARLA_GT_VOCAB
        _advisory = os.environ.get("CUE_VOCAB_ADVISORY") == "1"
        bad_vocab = validate_plan_cues(plan, _vocab)
        if _vocab is None and "cue_vocab_openset" not in gates:
            # ONCE. This is a MODE marker -- "the closed vocabulary was not applied" --
            # not a gate that fired, and `gates` is consumed by a Counter downstream
            # (see the per-condition summary), so appending it per attempt would report
            # three openset "gates" for a three-attempt generation and inflate the very
            # comparison the gate list exists to make. The advisory gate beside it is
            # appended only when it actually suppresses a finding, which is why it does
            # not need this guard.
            gates.append("cue_vocab_openset")
        if bad_vocab and _advisory:
            gates.append("cue_vocab_advisory")
            bad_vocab = []
        if bad_vocab:
            last = f"cue vocab: {bad_vocab}"; _fire("cue_vocab")
            msg = base + ("\n\nCUE VOCABULARY FAIL: this deployment cannot answer these "
                          "cues. Use a spelling it publishes.\n"
                          + "\n".join(f"- {e}" for e in bad_vocab))
            continue

        # EVERY ARM THAT PRODUCES A FORMULA IS GATED THE SAME WAY (including `twostage`).
        # Ungated arms are not comparable to gated ones on validity.
        if cond in FORMULA_ARMS:
            # Mode-check the FORMULA too. Without this the conditions are not
            # comparable: `none` can only state a constraint in `forbid_modes`,
            # which is mode-validated, while `full` could state one in a formula
            # that was not — crediting STL with constraints that name modes outside
            # the taxonomy and would compile to monitors that never fire.
            from nl_planner.stl_compile import validate_stl_modes
            dead = validate_stl_modes(stl, tax)
            if dead:
                last = f"stl names modes not in the taxonomy: {dead}"
                _fire("stl_modes")
                continue
            _allow_empty = os.environ.get("ALLOW_EMPTY_STL") == "1"
            ok, err = (True, None) if (_allow_empty and not (stl or "").strip()) \
                else quick_syntax_check(stl, modes=tax.modes_for_prompt())
            if not ok:
                last = f"syntax: {err}"; _fire("syntax")
                msg = base + f"\n\nVerifier feedback:\nSYNTAX FAIL: {err}"
                continue

        if cond == "full_tri":
            ok, notes = _tripartite(client, model, text, stl, plan)
            if not ok:
                last = f"tripartite: {notes}"; _fire("tripartite")
                msg = base + f"\n\nVerifier feedback:\nTRIPARTITE FAIL: {notes}"
                continue

        # THE CONTAINMENT GATE. Every other gate here is structural -- it asks whether the
        # plan is well-formed, never whether it does what the mission said. This one asks
        # whether the plan the model wrote actually satisfies the specification the model
        # itself wrote one call earlier, on EVERY branch. A plan can pass every validator
        # above and still, say, collapse an ordinal count the formula spelled out -- which
        # is exactly what the around-the-block plan does.
        if cond == "twostage_gated" and stl:
            try:
                from nl_planner.branch_materializer import to_brain_tree as _tbt
                from nl_planner.containment import check_tree as _check
                _rep = _check(_tbt(plan, tax, stl=stl), stl)
            except Exception:                                      # noqa: BLE001
                _rep = None
            if _rep is not None and _rep.parsed and not _rep.contained:
                bad = [v for v in _rep.verdicts if not v.satisfied]
                detail = "\n".join(
                    f"- branch {v.path!r} walks {' -> '.join(repr(w) for w in v.word)}"
                    for v in bad[:3])
                last = f"containment: {len(bad)}/{_rep.n_paths} leaves violate the formula"
                _fire("containment")
                msg = base + (
                    "\n\nCONTAINMENT FAIL: your plan permits a path your own formula "
                    f"forbids ({len(bad)} of {_rep.n_paths} branches).\n" + detail +
                    "\n\nEach line is one branch of your plan written as the sequence of "
                    "modes it passes through, with the cues true at each. Change the STEPS "
                    "so every branch satisfies the formula -- do not weaken the formula. "
                    "The most common cause is collapsing an ordinal: if the formula counts "
                    "seen, then not-seen, then seen again, the plan needs a separate step "
                    "for each of those three.")
                continue

        d = plan.model_dump()
        _disc = _discriminates(plan, tax)
        return dict(ok=True, attempts=attempt, gates=gates, stl=stl,
                    stage1_stl=stage1_stl, stage1_ok=stage1_ok,
                    degenerate_repairs=degenerate_repairs,
                    plan=d, shape=plan_shape(d), n_steps=len(d.get("steps") or []),
                    captured=captured_constraints(d, stl),
                    leaf_paths=_leaf_paths(d),
                    discriminates=_disc,
                    **__import__("task_capture").score_plan(
                        d, stl, text, tax, valid=True, leaf_paths=_leaf_paths(d),
                        discriminates=_disc, captured=captured_constraints(d, stl)))

    return dict(ok=False, attempts=attempt, gates=gates, error=last, stl="",
                stage1_stl=stage1_stl, stage1_ok=stage1_ok,
                degenerate_repairs=degenerate_repairs,
                plan=None, shape=None, n_steps=0)


def _report(rows, conds) -> None:
    if conds:
        print("\nARMS  (each name says what was REMOVED)")
        for c in conds:
            print(f"    {c:<16} {arm_label(c)}")
    print("\n" + "=" * 78)
    print("PLAN VALIDITY BY CONDITION  (identical JSON validators in every condition)")
    print("=" * 78)
    for c in conds:
        sub = [r for r in rows if r["condition"] == c]
        ok = [r for r in sub if r["ok"]]
        att = sum(r["attempts"] for r in sub) / max(1, len(sub))
        gate = collections.Counter(g for r in sub for g in r["gates"])
        print(f"  {c:<6} valid {len(ok):3}/{len(sub):<3}  mean attempts {att:.2f}"
              f"   gate fires {dict(gate)}")

    print("\nPLANS PRODUCED, BY TIER (what a linear sequence cannot express)")
    tiers = ["LINEAR", "BRANCHED", "NESTED"]
    print("  " + "condition".ljust(10) + "".join(f"{t:>10}" for t in tiers))
    for c in conds:
        cells = []
        for t in tiers:
            n = sum(1 for r in rows
                    if r["condition"] == c and r["ok"] and r["shape"] == t)
            cells.append(str(n).rjust(10))
        print("  " + c.ljust(10) + "".join(cells))
    print("\n  A condition producing FEWER hard plans is not necessarily worse — it may")
    print("  be reading the same mission as simpler. Compare per mission id, below.")

    print("\nTHREE DIFFERENT QUESTIONS, never collapsed into one")
    print("  VALID     is it well-formed?        (structural; never reads the English)")
    print("  GROUNDED  do its names refer to anything real?")
    print("  FAITHFUL  does it say what the MISSION said?")
    print()
    print(f"  {'arm':<9}{'VALID':>10}{'GROUNDED':>11}{'gnd rate':>10}"
          f"{'FAITHFUL':>11}{'valid but NOT faithful':>25}")
    for c in conds:
        sub = [r for r in rows if r["condition"] == c and r["ok"]]
        tot = [r for r in rows if r["condition"] == c]
        if not sub:
            continue
        g = sum(1 for r in sub if r.get("grounded"))
        gr = sum(r.get("grounded_rate") or 0 for r in sub) / len(sub)
        want = [r for r in sub if (r.get("n_expected") or 0) > 0]
        f = sum(1 for r in want if r.get("faithful"))
        vbu = sum(1 for r in sub if r.get("valid_but_unfaithful"))
        print(f"  {c:<9}{len(sub)}/{len(tot):<7}{g}/{len(sub):<8}{gr:>9.2f}"
              f"{f}/{len(want):<8}{vbu:>19}/{len(sub)}")
    print("\n  A plan can be valid and still unfaithful to the instruction (e.g. a linear")
    print("  plan for a branching mission); validity alone does not show it.")

    print("\nCONTINGENCY CAPTURE — did the plan keep the alternative it was told about?")
    print("  (a flat plan for 'if the gate is shut go around' is perfectly VALID —")
    print("   it commits to one course and drops the other, which validity cannot")
    print("   see, exactly as it could not see a dropped constraint)")
    for c in conds:
        sub = [r for r in rows if r["condition"] == c and r["ok"]]
        if not sub:
            continue
        multi = [r for r in sub if (r.get("leaf_paths") or 1) > 1]
        disc = [r for r in sub if r.get("discriminates")]
        mean = sum(r.get("leaf_paths") or 1 for r in sub) / len(sub)
        print(f"  {c:<9} {len(multi):3}/{len(sub):<3} admit >1 execution   "
              f"{len(disc):3}/{len(sub):<3} actually DISCRIMINATE   "
              f"mean paths {mean:.2f}")

    print("\nCONSTRAINT CAPTURE — did the rule survive, in EITHER representation?")
    print("  (plan validity cannot see a dropped constraint: a plan that ignores")
    print("   'never cross the grass' is still a perfectly valid plan)")
    for c in conds:
        sub = [r for r in rows if r["condition"] == c and r["ok"]]
        got = [r for r in sub if r.get("captured")]
        print(f"  {c:<9} {len(got):3}/{len(sub):<3} runs captured a constraint")
    print("\nPER-MISSION SHAPE DISAGREEMENT (same mission, different plan structure)")
    by_id = collections.defaultdict(dict)
    for r in rows:
        if r["ok"]:
            by_id[r["id"]][r["condition"]] = r["shape"]
    n_dis = 0
    for mid, d in sorted(by_id.items()):
        if len(set(d.values())) > 1:
            n_dis += 1
            print(f"  {mid:<34} " + "  ".join(f"{k}={v}" for k, v in sorted(d.items())))
    print(f"  {n_dis} of {len(by_id)} missions structurally disagree across conditions")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="task YAML to run the arms over")
    ap.add_argument("--conditions", nargs="*", default=["full", "none"],
                    choices=list(CONDITIONS))
    ap.add_argument("--seed-note", default="", help="free text recorded in the output")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--taxonomy", default=os.path.join(
        WS, "carla_gt_bridge", "config", "cluster_map.carla_town01.yaml"),
        help="the archetype constraint missions need a taxonomy that HAS the modes "
             "they name — town01 has only {junction, path}, so 'never go through the "
             "alley' is unstatable there and the comparison is void")
    ap.add_argument("--ids", nargs="*", default=None,
                    help="run only these mission ids (prefix match)")
    ap.add_argument("--out", default=os.path.join(PKG, "reports", "stl_ablation.json"))
    a = ap.parse_args()

    import yaml
    from nl_planner.taxonomy import load_taxonomy
    doc = yaml.safe_load(open(a.tasks))
    tasks = doc["tasks"][: a.limit] if a.limit else doc["tasks"]
    if a.ids:
        tasks = [t for t in tasks if any(t["id"].startswith(p) for p in a.ids)]
    tax = load_taxonomy(a.taxonomy)

    rows = []
    for cond in a.conditions:
        print(f"\n===== condition: {cond}  ({len(tasks)} missions x {a.trials} trial(s))")
        for t in tasks:
            for trial in range(a.trials):
                r = run_one(t["mission"], cond, tax, a.model, a.retries)
                r.update(id=t["id"], condition=cond, trial=trial, mission=t["mission"])
                rows.append(r)
                flag = "OK  " if r["ok"] else "FAIL"
                cap = r.get("captured") or []
                print(f"  [{flag}] {t['id']:<32} {r.get('shape') or '-':<7} "
                      f"a={r['attempts']} cap={cap or '-'} gates={r['gates']}",
                      flush=True)
        json.dump(rows, open(a.out, "w"), indent=2, default=str)

    _report(rows, a.conditions)
    json.dump(rows, open(a.out, "w"), indent=2, default=str)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

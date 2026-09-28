"""Turn a tree-shaped ``NavPlan`` into a sequence of brain-compatible linear plans.

The executor walks the tree with a list-of-branch-indices ``path``:

- ``materialize_segment(plan, taxonomy, path=[])`` returns the leading linear
  chunk of the root and the next ``decision_step`` (or ``None`` for terminal).
- ``materialize_segment(plan, taxonomy, path=[2])`` returns the leading linear
  chunk of the branch chosen at the first decision (index 2), and so on.

Each returned ``brain_plan`` is a dict in exactly the shape ``brain_controller``
already expects (``plan_name`` / ``description`` / ``cluster_labels`` / ``steps``),
ready to be ``json.dumps``'d into ``/brain/incoming_plan``.

Semantic modes are resolved to scalar cluster ids via the taxonomy's
``canonical_id()`` (first id listed for the mode). The executor still
watches ``/predicted_cluster`` against the full id set for transient
diagnostic / multi-id mode matching — see ``taxonomy.resolve()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .schemas import Branch, NavPlan, PlanStep, infer_trigger
from .taxonomy import ClusterTaxonomy


def _step_common(step: PlanStep, taxonomy: ClusterTaxonomy, index: int) -> dict[str, Any]:
    """Fields shared by the linear-segment and full-tree brain encodings.

    Resolves the semantic mode to CLUSTER ID SETS here, in the planner, so brain
    never needs the taxonomy or the label vocabulary:

    ``goal_cluster``
        Unchanged scalar (``taxonomy.canonical_id``) — the MPC region logic and
        every existing consumer still read this.
    ``accept_clusters``
        Every id that counts as "in goal_mode", preferred-first. Membership in
        this set replaces the old ``current == goal_cluster`` equality test, so a
        step whose goal is ``Road: On`` is satisfied at a junction.
    ``accept_clusters_degraded``
        ``accept_clusters`` plus the coarser ids that may satisfy the step when
        the fine classifier does not fire. Brain applies this ONLY on
        ``landmark`` steps — see schemas.TRIGGERS.
    ``perception_backed``
        False when the mode is a plan-level construct in this environment, so
        brain knows not to wait for a cluster that will never arrive.
    """
    trigger = step.trigger or infer_trigger(step.transition_cue)
    out = {
        "step":            index,
        "description":     step.description,
        "start_cluster":   taxonomy.canonical_id(step.start_mode),
        "goal_cluster":    taxonomy.canonical_id(step.goal_mode),
        "start_mode":      step.start_mode,
        "goal_mode":       step.goal_mode,
        "transition_cue":  step.transition_cue,
        "notes":           step.notes,
        "trigger":         trigger,
        "cue_ordinal":     step.cue_ordinal,
        "accept_clusters": list(taxonomy.accept_clusters(step.goal_mode)),
        "accept_clusters_degraded": list(
            taxonomy.accept_clusters(step.goal_mode, degraded=True)
        ),
        "perception_backed": taxonomy.is_perception_backed(step.goal_mode),
    }
    # `Phi_X U cue`: resolved here for the same reason `accept_clusters` is —
    # brain never sees the taxonomy, so a mode name in the tree is a mode name
    # nothing downstream can act on. `hold_accept_clusters` is the STRICT set
    # (degraded=False): the coarse fallback exists so a MISSED fine detection
    # cannot strand a step, and widening an invariant's accept set does the
    # opposite of what an invariant is for — it makes leaving the mode harder to
    # notice, which is the failure the monitor exists to catch.
    #
    # Emitted ONLY when the step actually holds a mode, so steps without a hold
    # (and the checked-in `mission.*.json` files) are unchanged byte-for-byte.
    if step.hold_mode is not None:
        out["hold_mode"] = step.hold_mode
        out["until"] = step.until
        out["hold_accept_clusters"] = list(taxonomy.accept_clusters(step.hold_mode))
    return out


def _require_ids(modes, taxonomy) -> list[int]:
    """Cluster ids satisfying every requirement in ``modes``.

    UNION within an axis, INTERSECTION across axes. Two modes on one axis are
    alternatives ("a path or a junction"); two on different axes are simultaneous facts
    about one place ("a path" AND "surface: sidewalk"), and unioning those gives
    "path OR sidewalk", which a route driven entirely on the carriageway satisfies --
    the mission's opposite.
    """
    by_axis: dict[str, set[int]] = {}
    for m in modes:
        axis = (getattr(taxonomy, "axes", None) or {}).get(
            taxonomy.canonical_mode(m) or m, "_default")
        by_axis.setdefault(axis, set()).update(
            taxonomy.accept_clusters(m, degraded=False))
    ids: set[int] = set()
    for i, group in enumerate(by_axis.values()):
        ids = set(group) if i == 0 else (ids & group)
    return sorted(ids)


def _forbid_clusters(plan: NavPlan, taxonomy: ClusterTaxonomy) -> list[int] | None:
    """Resolve ``plan.forbid_modes`` to the UNION of their accept sets.

    Union, not intersection, and it has to be: "never enter the plaza OR the
    grass" forbids a region that is either. Intersection would forbid only what
    is both, i.e. usually nothing, and a negative constraint that forbids nothing
    passes silently — the exact failure mode a `[]` return is indistinguishable
    from, which is why `None` (no constraint declared) and `[]` are kept apart at
    the call sites rather than collapsed here.

    STRICT sets (``degraded=False``). The degraded fallback exists so a missed
    fine detection cannot strand a positive step; folding it into a PROHIBITION
    inverts its meaning — it would forbid every coarse region that merely
    subsumes the banned one, and on a taxonomy where `path` contains every
    junction id that bans the whole map.

    Sorted and de-duplicated so the emitted tree is stable across runs: this
    value lands in a checked-in `mission.*.json`, and a set's iteration order
    would show up as a spurious diff.
    """
    modes = plan.forbid_modes or None
    if not modes:
        return None
    ids: set[int] = set()
    for mode in modes:
        ids.update(taxonomy.accept_clusters(mode))
    return sorted(ids)


# --------------------------------------------------------------------------- #
# Result container                                                             #
# --------------------------------------------------------------------------- #

@dataclass
class MaterializedSegment:
    """One linear chunk of a tree NavPlan, ready to ship to brain_controller."""

    #: Plan dict in brain_controller's JSON schema.
    brain_plan: dict[str, Any]
    #: The branch-bearing PlanStep that follows this chunk (None at terminus).
    decision_step: PlanStep | None
    #: Path of branch indices that produced this segment (for diagnostic logs).
    path: tuple[int, ...]
    #: True iff this is the final segment (no more decisions in the tree).
    is_terminal: bool


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def materialize_segment(
    plan: NavPlan,
    taxonomy: ClusterTaxonomy,
    *,
    path: Sequence[int] = (),
    plan_name_override: str | None = None,
) -> MaterializedSegment:
    """Build the brain-compatible linear plan for the segment selected by ``path``.

    ``path`` is the list of branch indices chosen at each decision step
    encountered so far. ``path=[]`` selects the root linear chunk.
    """
    sub_plan = _follow_path(plan, path)

    linear_steps: list[PlanStep] = []
    decision_step: PlanStep | None = None
    for step in sub_plan:
        if step.branches is None:
            linear_steps.append(step)
        else:
            decision_step = step
            break

    if not linear_steps and decision_step is None:
        # Empty sub_plan — should be impossible because schemas.Branch enforces
        # min_length=1 on sub_plan, but guard anyway.
        raise ValueError(f"segment at path={list(path)} is empty")

    brain_plan = _to_brain_plan(
        plan=plan,
        linear_steps=linear_steps,
        taxonomy=taxonomy,
        path=tuple(path),
        plan_name_override=plan_name_override,
    )

    return MaterializedSegment(
        brain_plan=brain_plan,
        decision_step=decision_step,
        path=tuple(path),
        is_terminal=(decision_step is None),
    )


def walk_all_segments(
    plan: NavPlan,
    taxonomy: ClusterTaxonomy,
) -> list[MaterializedSegment]:
    """Enumerate every reachable linear segment in the tree (DFS over branches).

    Useful for offline validation / unit testing — not used by the executor at
    runtime, which materializes only the path the VLM picks.
    """
    out: list[MaterializedSegment] = []
    _walk(plan, taxonomy, path=[], out=out)
    return out


def _walk(plan: NavPlan, taxonomy: ClusterTaxonomy, *, path: list[int], out: list[MaterializedSegment]) -> None:
    seg = materialize_segment(plan, taxonomy, path=tuple(path))
    out.append(seg)
    if seg.decision_step is None:
        return
    for i in range(len(seg.decision_step.branches or ())):
        _walk(plan, taxonomy, path=path + [i], out=out)


# --------------------------------------------------------------------------- #
# Tree walking                                                                 #
# --------------------------------------------------------------------------- #

def _follow_path(plan: NavPlan, path: Sequence[int]) -> list[PlanStep]:
    """Return the sub_plan reached by descending ``path`` from ``plan.steps``."""
    current: list[PlanStep] = list(plan.steps)
    for depth, branch_idx in enumerate(path):
        decision: PlanStep | None = None
        for step in current:
            if step.branches is not None:
                decision = step
                break
        if decision is None or decision.branches is None:
            raise ValueError(
                f"path index {depth} expected a decision step, found none in "
                f"sub_plan {[s.description for s in current]!r}"
            )
        if branch_idx < 0 or branch_idx >= len(decision.branches):
            raise ValueError(
                f"path index {depth} = {branch_idx} out of range; "
                f"decision step has {len(decision.branches)} branches"
            )
        current = list(decision.branches[branch_idx].sub_plan)
    return current


# --------------------------------------------------------------------------- #
# brain_controller plan-dict construction                                      #
# --------------------------------------------------------------------------- #

def _to_brain_plan(
    *,
    plan: NavPlan,
    linear_steps: list[PlanStep],
    taxonomy: ClusterTaxonomy,
    path: tuple[int, ...],
    plan_name_override: str | None,
) -> dict[str, Any]:
    """Render a brain_controller-shaped plan dict from a list of linear PlanSteps."""
    brain_steps: list[dict[str, Any]] = [
        _step_common(step, taxonomy, i) for i, step in enumerate(linear_steps)
    ]

    cluster_labels: dict[str, str] = {
        str(cid): label for cid, label in taxonomy.cluster_labels().items()
    }

    suffix = f" [path={list(path)}]" if path else ""
    name = plan_name_override or f"{plan.plan_name}{suffix}"

    forbid = _forbid_clusters(plan, taxonomy)
    extra: dict[str, Any] = {}
    if forbid is not None:
        # PLAN-level, so it is carried on EVERY segment of the tree. A negative
        # constraint that applied only to the segment it was declared in would
        # switch itself off at the first branch descent.
        extra["forbid_modes"] = list(plan.forbid_modes or ())
        extra["forbid_clusters"] = forbid

    return {
        "plan_name":      name,
        "description":    plan.description,
        "cluster_labels": cluster_labels,
        **extra,
        "nl_planner": {
            # Executor uses these to know "where in the tree am I" without
            # re-walking the tree itself.
            "path":           list(path),
            "ends_at_decision": False,  # filled in below
        },
        "steps": brain_steps,
    }


def attach_decision_metadata(
    segment: MaterializedSegment,
) -> None:
    """Fill in the ``ends_at_decision`` and per-branch metadata on the brain plan.

    Called by the executor after materialization so the brain JSON can be
    pretty-printed in logs / dashboards without the executor having to keep
    parallel state.
    """
    meta = segment.brain_plan.setdefault("nl_planner", {})
    meta["path"] = list(segment.path)
    meta["ends_at_decision"] = segment.decision_step is not None
    if segment.decision_step is not None:
        meta["decision"] = {
            "description":    segment.decision_step.description,
            "start_mode":     segment.decision_step.start_mode,
            "transition_cue": segment.decision_step.transition_cue,
            "branches": [
                {"index": i, "vlm_cue": b.vlm_cue}
                for i, b in enumerate(segment.decision_step.branches or ())
            ],
        }


# --------------------------------------------------------------------------- #
# Full-tree conversion for brain_controller v2 (tree-aware brain)              #
# --------------------------------------------------------------------------- #

def to_brain_tree(
    plan: NavPlan,
    taxonomy: ClusterTaxonomy,
    stl: str | None = None,
) -> dict[str, Any]:
    """Convert a NavPlan tree to a brain_controller-compatible JSON dict.

    Unlike :func:`materialize_segment` (which slices the tree into linear
    chunks for the old segment-by-segment executor), this preserves the
    FULL branching structure so the tree-aware brain_controller can walk
    it natively and make VLM branch decisions in-process.

    Output schema (each step)::

        {
            "step":           int,            # index in its containing sub_plan
            "description":    str,
            "start_cluster":  int,            # taxonomy.canonical_id(start_mode)
            "goal_cluster":   int,
            "start_mode":     str,            # kept for diagnostics
            "goal_mode":      str,
            "transition_cue": str | None,
            "notes":          str | None,
            "branches":       None | [
                {"vlm_cue": str, "sub_plan": [<more steps in the same shape>]},
                ...
            ],
        }

    A dwell step additionally carries ``hold_mode`` / ``until`` /
    ``hold_accept_clusters``; steps that hold nothing carry none of the three, so
    every tree emitted before ``Phi_X U cue`` existed is unchanged byte for byte.

    The top-level dict carries ``cluster_labels`` (every cluster id known
    to the taxonomy, not only the ones referenced), ``forbid_modes`` /
    ``forbid_clusters`` when the plan declares a negative constraint, and a small
    ``nl_planner`` metadata block so consumers can tell tree-shaped plans
    from legacy linear ones.
    """
    cluster_labels: dict[str, str] = {
        str(cid): label for cid, label in taxonomy.cluster_labels().items()
    }
    result: dict[str, Any] = {
        "plan_name":      plan.plan_name,
        "description":    plan.description,
        "cluster_labels": cluster_labels,
        "steps":          _convert_steps(plan.steps, taxonomy),
        "nl_planner": {
            "version":      2,
            "tree_shaped":  True,
            "environment":  taxonomy.environment,
        },
    }
    forbid = _forbid_clusters(plan, taxonomy)
    if forbid is not None:
        result["forbid_modes"] = list(plan.forbid_modes or ())
        result["forbid_clusters"] = forbid
    # INSTANCE-SCOPED PROHIBITION, passed through UNRESOLVED and on purpose.
    #
    # `brain_controller` builds its InstanceForbidTracker from
    # `plan_data.get("forbid_instances")`, so the field must be written into the tree brain
    # loads; otherwise the tracker is empty and "never turn at the second intersection"
    # never binds.
    #
    # NOT resolved to cluster ids here, unlike `forbid_modes` above. InstanceForbidTracker's
    # docstring is explicit that it cannot be: a plan names modes so that one plan is
    # portable between CARLA and the campus bags, and on a branching plan WHICH junction is
    # second depends on the branch taken, which is not known until it is taken. So the
    # counting happens at run time and this only has to carry the claim across.
    inst = getattr(plan, "forbid_instances", None)
    if inst:
        result["forbid_instances"] = [
            e.model_dump() if hasattr(e, "model_dump") else dict(e) for e in inst
        ]
    req = getattr(plan, "require_modes", None)
    if req:
        # STRICT sets, not degraded. An invariant is the one place permissiveness is
        # exactly wrong: the claim is "the robot never left this mode", and admitting
        # the coarser parent would make it vacuous.
        # UNION within an axis, INTERSECTION across axes.
        #
        # Two modes on the same axis are alternatives -- "stay on a path or a junction"
        # means either is fine. Two modes on DIFFERENT axes are simultaneous facts about
        # one place: "a path" and "surface: sidewalk" both have to hold, and unioning them
        # yields "path OR sidewalk", which a route driven entirely on the carriageway
        # satisfies -- the mission's opposite (e.g. a plan requiring `sidewalk` merged
        # with a formula requiring `path`).
        result["require_modes"] = list(req)
        result["require_clusters"] = _require_ids(req, taxonomy)
    # Attach CARLA spatial data if the taxonomy provides it.
    centroids = taxonomy.centroid_map()
    if centroids:
        result["centroids"] = centroids
    bearing_map = taxonomy.bearing_map_dict()
    if bearing_map:
        result["bearing_map"] = bearing_map
    if stl:
        _apply_stl_monitors(result, stl, taxonomy)
    return result


def _apply_stl_monitors(tree: dict[str, Any], stl: str,
                        taxonomy: ClusterTaxonomy) -> None:
    """Compile the FORMULA into monitors and merge them into the tree.

    This is the only place the STL formula reaches the executor. Everything else in this
    module derives from the JSON plan; most of a formula is a redundant re-encoding of
    the tree.

    Only two shapes carry information the tree CANNOT hold, and only those are merged:

        Phi_X U psi   the tree records a step's DESTINATION, never that a mode had to
                      HOLD on the way. `path -> junction` and "path -> junction without
                      leaving the path" are the same tree.
        G(~Phi_X)     a plan-level invariant with no field in the tree at all.

    Reach-goals (`F Phi_X`, `F Detect(x)`) are deliberately NOT merged. They restate
    steps that already exist, and writing them in a second time would double-count the
    same requirement and let the compiler appear to be doing work it is not.

    NOTHING IS SILENTLY DROPPED OR SILENTLY GUESSED. A hold is applied only when the
    plan does not already declare one and there is exactly one candidate step; anything
    else is recorded under `unapplied` so a reader can see what the formula asked for
    and did not get.
    """
    from .stl_compile import ParseError, compile_monitors, parse

    tree["stl_formula"] = stl
    try:
        ast, junk = parse(stl)
    except ParseError as exc:
        tree["stl_monitors"] = {"parsed": False, "error": str(exc)}
        return
    spec = compile_monitors(ast)
    report: dict[str, Any] = {"parsed": True, "ignored_prose": junk,
                              "n_reach_redundant": len(spec.reach),
                              "applied": [], "unapplied": []}

    # G(Phi_X): a mission-wide positive invariant. Union with the plan's own, since
    # both are the same claim about the same run.
    if spec.require_modes:
        have = set(tree.get("require_modes") or ())
        add = [m for m in spec.require_modes if m not in have and m in taxonomy]
        for m in spec.require_modes:
            if m not in taxonomy:
                report["unapplied"].append(
                    f"G(Phi) names {m!r}, which is not a mode in this environment")
        if add:
            modes = list(tree.get("require_modes") or ()) + add
            tree["require_modes"] = modes
            tree["require_clusters"] = _require_ids(modes, taxonomy)
            report["applied"].append({"require_modes": add})

    # G(~Phi_X): union with anything the plan declared, since both are the same claim.
    if spec.forbid_modes:
        have = set(tree.get("forbid_modes") or ())
        add = [m for m in spec.forbid_modes if m not in have and m in taxonomy]
        skipped = [m for m in spec.forbid_modes if m not in taxonomy]
        if add:
            modes = list(tree.get("forbid_modes") or ()) + add
            ids: set[int] = set()
            for m in modes:
                ids.update(taxonomy.accept_clusters(m, degraded=False))
            tree["forbid_modes"] = modes
            tree["forbid_clusters"] = sorted(ids)
            report["applied"].append({"forbid_modes": add})
        for m in skipped:
            report["unapplied"].append(
                f"G(~Phi) names {m!r}, which is not a mode in this environment")

    # Phi_X U psi: a hold over the run-up. Applied to step 0 only, and only when the
    # plan left it open — the formula says WHAT must hold, not WHERE in the tree, and
    # inventing a location would be a guess dressed as a compilation.
    steps = tree.get("steps") or []
    for h in spec.holds:
        mode = h["hold_mode"]
        if mode not in taxonomy:
            report["unapplied"].append(f"hold_mode {mode!r} is not in this environment")
            continue
        if not steps:
            report["unapplied"].append(f"hold {mode!r}: plan has no steps")
            continue
        if steps[0].get("hold_mode"):
            report["unapplied"].append(
                f"hold {mode!r}: step 0 already declares "
                f"hold_mode={steps[0]['hold_mode']!r}; the plan wins")
            continue
        steps[0]["hold_mode"] = mode
        steps[0]["until"] = h["until"]
        steps[0]["hold_accept_clusters"] = list(
            taxonomy.accept_clusters(mode, degraded=False))
        report["applied"].append({"hold_mode": mode, "on_step": 0})

    for u in spec.unhandled:
        report["unapplied"].append(u)
    tree["stl_monitors"] = report


def _convert_steps(
    steps: Sequence[PlanStep],
    taxonomy: ClusterTaxonomy,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, step in enumerate(steps):
        d: dict[str, Any] = _step_common(step, taxonomy, i)
        d["branches"] = None
        if step.branches:
            d["branches"] = [
                {
                    "vlm_cue":  b.vlm_cue,
                    "sub_plan": _convert_steps(b.sub_plan, taxonomy),
                }
                for b in step.branches
            ]
        out.append(d)
    return out


__all__ = [
    "MaterializedSegment",
    "materialize_segment",
    "walk_all_segments",
    "attach_decision_metadata",
    "to_brain_tree",
]

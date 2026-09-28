"""Export a classifier's label vocabulary to nl_planner's cluster_map schema.

The legacy pipeline only ever *consumed* hand-authored ``cluster_map.<env>.yaml``
taxonomies; nothing generated them. This writes one in the exact schema
``nl_planner.taxonomy.load_taxonomy`` expects::

    environment: <env>
    source: <provenance>
    modes:
      "Open Road":          [0]
      "Building: Approach":  [1]
      ...

Each closed-set label becomes one mode mapped to its integer id (the classifier
label index) — which is exactly the scalar published on ``/predicted_cluster``,
so the id nl_planner/brain read stays consistent end-to-end.
"""

from __future__ import annotations

from typing import List, Optional

import yaml


def prettify_label(label: str) -> str:
    """'along_building' -> 'Building: Along'; 'open_road' -> 'Open Road'."""
    if label == "open_road":
        return "Open Road"
    phase, _, ltype = label.partition("_")
    if not ltype:
        return label.replace("_", " ").title()
    return f"{ltype.title()}: {phase.title()}"


def load_planner_mode_map(nav_modes_yaml: str) -> dict:
    """Return {label_name: planner_mode} from nav_modes.yaml (None -> prettify)."""
    import yaml
    cfg = yaml.safe_load(open(nav_modes_yaml))
    return {m["name"]: m.get("planner_mode") for m in cfg.get("modes", [])}


def build_cluster_map(labels: List[str], environment: str,
                      source: Optional[str] = None,
                      planner_mode_map: Optional[dict] = None) -> dict:
    """Build the cluster_map dict (id = index into ``labels``).

    If ``planner_mode_map`` is given (from nav_modes.yaml), label names are
    mapped to legacy planner modes for drop-in nl_planner compatibility, and
    multiple labels sharing a planner mode are merged into one id list.
    """
    modes: dict = {}
    for idx, label in enumerate(labels):
        if planner_mode_map is not None:
            mode = planner_mode_map.get(label) or prettify_label(label)
        else:
            mode = prettify_label(label)
        modes.setdefault(mode, []).append(idx)
    doc = {"environment": environment, "modes": modes}
    if source:
        doc["source"] = source
    return doc


def write_cluster_map(labels: List[str], environment: str, out_path: str,
                      source: Optional[str] = None) -> dict:
    """Write ``cluster_map.<env>.yaml`` and return the doc dict."""
    doc = build_cluster_map(labels, environment, source)
    with open(out_path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False)
    return doc


# Subsumption hierarchy for the nav_modes vocab: each cluster's id is listed
# under its OWN mode AND every ancestor mode it "defaults down" to. This is what lets a "Road: On" plan step be satisfied
# while at a junction (defaulting), while an "Intersection: In" step matches it
# specifically. nl_planner.taxonomy explicitly allows overlapping id sets.
#
# NOTE this relation is UPWARD only: child id -> parent mode. It is semantically
# sound and always safe (a junction really IS on a road). It is NOT the same as
# the downward degradation in DEGRADE_TO below, and the two must not be merged.
# PEDESTRIAN domain (the real robot, and what the bags contain). The planner-facing
# name IS the cluster name: one vocabulary, no translation layer, nothing to keep in
# sync. A footpath fork is not an "Intersection" and a hedge is not a "Wall", so the
# road-network names below were actively misleading here.
HIERARCHY = {
    "open_space": ("open_space", []),
    "path":       ("path", []),
    "along_edge": ("along_edge", ["path"]),
    "passage":    ("passage", ["path"]),
    "junction":   ("junction", ["path"]),
    "other":      ("other", []),
}

# CAR-ON-ROAD domain (CARLA). Same lattice, road-network planner names, because the
# CARLA cluster maps and the driving macros already use them. Selected per
# environment via the `hierarchy=` argument — this is what the taxonomy seam is for.
ROAD_HIERARCHY = {
    "open_space": ("Open Space", []),
    "path":       ("Road: On", []),
    "along_edge": ("Along Wall", ["Road: On"]),
    "passage":    ("Passage", ["Road: On"]),
    "junction":   ("Intersection: In", ["Road: On"]),
    "covered":    ("Covered", ["Road: On"]),
    "other":      ("Other", []),
}

# DOWNWARD degradation: which COARSER geometric clusters may still satisfy a
# plan step whose goal_mode is the fine one, when the fine classifier does not
# fire. This is the opposite direction from HIERARCHY and is NOT logically
# implied — being on a `path` does not mean you are alongside a building.
#
# It is a deliberate permissive policy: along_edge is poorly recalled from LiDAR
# alone (it needs the camera) and does not transfer across bags, and junction is
# not reliably separable from the rest across bags. Requiring the fine cluster
# would strand the plan at a step the perception cannot confirm.
#
# SAFETY CONDITION: a degraded match is only accepted for steps whose trigger is
# `landmark` — i.e. where a Detect(...) predicate carries the real evidence and
# the cluster is merely a permissive guard. brain_controller enforces this; the
# taxonomy only declares what is permissible. A `traverse` step, where the
# cluster IS the evidence, never degrades.
# Degradation targets the coarse mode's OWN cluster id (e.g. `path` for
# "Road: On"), never the ids that merely reach it by subsumption — otherwise
# "Along Wall" would silently accept `junction` and `passage` too, which is a
# much weaker condition than "on the way past the building".
#
# `Open Space` is deliberately absent: it is not a specialization of `path`, so
# being on a path does not degrade-satisfy it, and no landmark idiom needs it.
DEGRADE_TO = {
    "along_edge": ["path"],
    "passage":    ["path"],
    "junction":   ["path"],
}

#: Car-on-road equivalent, paired with ROAD_HIERARCHY.
ROAD_DEGRADE_TO = {
    "Along Wall":       ["Road: On"],
    "Passage":          ["Road: On"],
    "Intersection: In": ["Road: On"],
}


def build_hierarchical_cluster_map(labels: List[str], environment: str,
                                   source: Optional[str] = None,
                                   hierarchy: Optional[dict] = None,
                                   degrade_to: Optional[dict] = None,
                                   perception_backed: Optional[dict] = None,
                                   id_labels: Optional[dict] = None) -> dict:
    """cluster_map with OVERLAPPING mode->id sets encoding subsumption.

    Each label id is added to its primary mode and to all ancestor modes, so a
    specialized cluster (e.g. junction) satisfies both its specific mode
    ("Intersection: In") and the coarse parent ("Road: On"). Labels absent from
    ``hierarchy`` fall back to a prettified standalone mode.

    Each mode's id list is ordered PREFERRED-FIRST: the mode's own cluster id
    comes first (so ``ClusterTaxonomy.canonical_id`` keeps returning the finest
    id), then any ids that reach it by subsumption.

    ``mode_meta`` is emitted alongside ``modes`` and carries, per mode:

    ``accept_degraded``
        Coarser modes that may satisfy this one on a ``landmark``-triggered step
        (see ``DEGRADE_TO``). Resolved here to concrete cluster ids so brain does
        not need the label vocabulary.
    ``perception_backed``
        Whether a classifier can actually produce this mode in this environment.
        Defaults True for any mode with at least one own id. Pass
        ``perception_backed={"Intersection: In": False}`` for environments where
        the mode is a plan-level construct (real campus) rather than a percept
        (CARLA) — same plan, different binding.

    ``mode_meta`` is additive; ``nl_planner.taxonomy.load_taxonomy`` ignores
    unknown top-level keys, so older consumers still load these files.
    """
    h = hierarchy or HIERARCHY
    deg = DEGRADE_TO if degrade_to is None else degrade_to
    pb = perception_backed or {}

    modes: dict = {}
    own_ids: dict = {}          # mode -> ids whose PRIMARY mode this is
    inherited: dict = {}        # mode -> ids that reach it by subsumption

    # ``id_labels`` is for SPARSE cluster ids — a CARLA mission corridor uses a
    # subset of a town's region ids, so the ids are not 0..n-1. Padding a
    # positional list with placeholder labels instead leaks a junk mode into the
    # taxonomy (it gets prettified into something like "_Unused__: "), which then
    # shows up in the LLM's legal-mode list. Prefer this parameter.
    pairs = (sorted(id_labels.items()) if id_labels is not None
             else list(enumerate(labels)))

    for idx, label in pairs:
        if label in h:
            primary, ancestors = h[label]
            own_ids.setdefault(primary, []).append(idx)
            for m in ancestors:
                inherited.setdefault(m, []).append(idx)
        else:
            own_ids.setdefault(prettify_label(label), []).append(idx)

    for mode in list(own_ids) + [m for m in inherited if m not in own_ids]:
        ordered = list(own_ids.get(mode, []))
        for idx in inherited.get(mode, []):
            if idx not in ordered:
                ordered.append(idx)
        modes[mode] = ordered

    mode_meta: dict = {}
    for mode, ids in modes.items():
        meta: dict = {
            "perception_backed": bool(pb.get(mode, bool(own_ids.get(mode)))),
        }
        degraded_ids: list = []
        for coarse in deg.get(mode, []):
            # OWN ids of the coarse mode only — not ids that reach it by
            # subsumption (see the DEGRADE_TO comment).
            for idx in own_ids.get(coarse, []):
                if idx not in ids and idx not in degraded_ids:
                    degraded_ids.append(idx)
        if degraded_ids:
            meta["accept_degraded"] = degraded_ids
        mode_meta[mode] = meta

    doc = {"environment": environment, "modes": modes, "mode_meta": mode_meta}
    if source:
        doc["source"] = source
    return doc


def write_hierarchical_cluster_map(labels: List[str], environment: str,
                                   out_path: str, source: Optional[str] = None,
                                   hierarchy: Optional[dict] = None,
                                   degrade_to: Optional[dict] = None,
                                   perception_backed: Optional[dict] = None,
                                   centroids: Optional[dict] = None,
                                   bearing_map: Optional[dict] = None,
                                   id_labels: Optional[dict] = None) -> dict:
    """Write a hierarchical (overlapping) ``cluster_map.<env>.yaml``.

    ``centroids`` / ``bearing_map`` are optional GEOMETRY, forwarded verbatim.
    They matter because ``nl_planner.branch_materializer.to_brain_tree`` only
    attaches geometry to the brain plan when the taxonomy carries it — without
    them a controller downstream has centroids for nothing to steer by. Units and
    frames are the caller's responsibility and must match what the controller
    expects: **raw world metres** for centroids, **degrees** for bearings.
    """
    doc = build_hierarchical_cluster_map(
        labels, environment, source, hierarchy, degrade_to, perception_backed,
        id_labels=id_labels,
    )
    if centroids:
        doc["centroids"] = centroids
    if bearing_map:
        doc["bearing_map"] = bearing_map
    with open(out_path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False)
    return doc

"""The contract for ``/predicted_cluster``: what the integer MEANS.

THE PROBLEM THIS EXISTS FOR. Five live nodes publish an Int16 on ``/predicted_cluster``
and about twelve subscribe, and the integer means three different things depending on who
sent it:

    gt_cluster_node              a REGION ID      (regions.<town>.npz / cluster_map)
    live_cluster_inference_node  an HDBSCAN CLUSTER ID (a trained weight set)
    bev_inference_node           a LABEL INDEX    (a classifier's vocabulary)

Each producer is self-consistent with its own ``cluster_map.<env>.yaml``, so nothing is
wrong until a producer from one pair meets a consumer holding the other's map. Then the
ids still parse, still fall in range, and still resolve to modes -- they just resolve to
the WRONG ones, and the robot drives somewhere reasonable-looking for the wrong reason.
That is a silent failure with a steering wheel attached.

THE FIX IS A NAME, ANNOUNCED AND CHECKED. Every producer states its id space once, on a
latched topic, and anything that resolves ids against a map asserts the space is the one
it loaded. Renaming ``/predicted_cluster`` would have touched robot-side nodes, and a
rename does not stop the next producer from guessing wrong -- an assertion does.

A SPACE NAME is ``<kind>:<env>``, where kind is one of the constants below. It is not a
free-form string: the point is that two nodes cannot agree by accident.
"""
from __future__ import annotations

import hashlib
from typing import Iterable

TOPIC = "/predicted_cluster"
SPACE_TOPIC = "/predicted_cluster/space"

#: What the integers are drawn from.
REGION = "region"     # ids index a region table (regions.<town>.npz)
HDBSCAN = "hdbscan"   # ids are cluster labels from a trained weight set
LABEL = "label"       # ids index a closed-set classifier's vocabulary
SPOOF = "spoof"       # a test fixture; never valid in a scored run
KINDS = (REGION, HDBSCAN, LABEL, SPOOF)


def space(kind: str, env: str, ids: Iterable[int] | None = None) -> str:
    """Build a space name, optionally fingerprinted by the id set.

    The fingerprint catches the subtler case: the same kind and env, but a cluster_map
    regenerated with a different id set (e.g. a region corpus and a cluster_map with
    different region counts that both call themselves `carla_town01`).
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}; got {kind!r}")
    base = f"{kind}:{env}"
    if ids is None:
        return base
    s = ",".join(str(int(i)) for i in sorted(set(ids)))
    return f"{base}#{hashlib.sha1(s.encode()).hexdigest()[:8]}"


def compatible(declared: str, expected: str) -> bool:
    """True when a producer's space satisfies a consumer's expectation.

    An expectation without a fingerprint accepts any fingerprint of the same kind and
    env, so a consumer can be strict or lenient without a second flag.
    """
    if not declared or not expected:
        return False
    if "#" in expected:
        return declared == expected
    return declared.split("#", 1)[0] == expected.split("#", 1)[0]

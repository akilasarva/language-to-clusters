#!/usr/bin/env python3
"""Read a CARLA town's region graph from its own geometry. No invented topology.

A proposed mission is only worth a CARLA run if the route it describes EXISTS. The region
tables carry waypoints and a region id per waypoint, so adjacency is recoverable rather
than assumed: two regions are adjacent when the road network actually carries you from one
to the other.

Adjacency is built from the WAYPOINT ORDER, not from centroid distance. Two junctions on
opposite sides of a block can be metres apart in a straight line and not connected at all;
consecutive waypoints, by contrast, are consecutive because the road goes that way. That
distinction is the whole reason this file exists instead of a `cdist` call.
"""
from __future__ import annotations

import collections
import os

import numpy as np

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(WS, "carla_gt_bridge", "config")


def town_npz(town: str) -> str:
    for p in (os.path.join(CFG, f"regions.{town}.npz"),
              os.path.join(CFG, "towns", f"regions.{town}.npz")):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no region table for {town}")


class TownGraph:
    def __init__(self, town: str):
        self.town = town
        d = np.load(town_npz(town), allow_pickle=True)
        self.centroids = d["centroids"]
        self.rids = [int(r) for r in d["rids"]]
        self.labels = {int(r): str(l) for r, l in zip(d["rids"], d["labels"])}
        self.waypoints = d["waypoints"]
        self.adj = self._adjacency()

    def _region_of_waypoints(self) -> list[int]:
        """Nearest centroid per waypoint. The table stores no per-waypoint region id, so
        this reconstructs it; centroid proximity is right HERE because a waypoint lies
        inside the region whose centroid it is nearest to, which is how the regions were
        cut in the first place."""
        w = self.waypoints[:, :2]
        c = self.centroids
        d2 = ((w[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        return [self.rids[i] for i in d2.argmin(1)]

    def _adjacency(self) -> dict[int, set[int]]:
        seq = self._region_of_waypoints()
        adj: dict[int, set[int]] = collections.defaultdict(set)
        for a, b in zip(seq, seq[1:]):
            if a != b:
                adj[a].add(b)
                adj[b].add(a)
        return dict(adj)

    def label(self, r: int) -> str:
        return self.labels.get(r, "?")

    def junctions(self) -> list[int]:
        return [r for r in self.rids if self.label(r) == "junction"]

    def paths(self) -> list[int]:
        return [r for r in self.rids if self.label(r) == "path"]

    def degree(self, r: int) -> int:
        return len(self.adj.get(r, ()))

    def route(self, start: int, goal: int, avoid: set[int] | None = None) -> list[int] | None:
        """Shortest region route, or None. `avoid` is what makes a prohibition mission
        checkable: if the route only exists THROUGH the forbidden region, the mission is
        impossible and should not be proposed."""
        avoid = avoid or set()
        if start == goal:
            return [start]
        seen, q = {start}, collections.deque([[start]])
        while q:
            p = q.popleft()
            for n in sorted(self.adj.get(p[-1], ())):
                if n in seen or n in avoid:
                    continue
                if n == goal:
                    return p + [n]
                seen.add(n)
                q.append(p + [n])
        return None

    def corridor(self, hops: int = 4) -> list[int] | None:
        """A route that crosses at least two junctions -- the shape every branching or
        ordinal mission needs. Picked by walking from the highest-degree junction so the
        result has somewhere to branch TO."""
        js = sorted(self.junctions(), key=lambda r: -self.degree(r))
        for j in js[:6]:
            for goal in sorted(self.paths(), key=lambda r: -self.degree(r))[:12]:
                r = self.route(j, goal)
                if r and len(r) >= hops and sum(
                        1 for x in r if self.label(x) == "junction") >= 2:
                    return r
        return None

    def summary(self) -> dict:
        return {
            "town": self.town,
            "regions": len(self.rids),
            "junctions": len(self.junctions()),
            "paths": len(self.paths()),
            "mean_degree": round(
                sum(self.degree(r) for r in self.rids) / max(len(self.rids), 1), 2),
            "max_degree": max((self.degree(r) for r in self.rids), default=0),
        }

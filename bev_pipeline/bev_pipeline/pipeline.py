"""Shared per-frame geometry pipeline: align -> ground-removal -> submap.

Used by both the offline dataset builder and the live inference node so they
produce identical inputs. Given a raw LiDAR cloud + pose, yields the
gravity-leveled, ground-removed, submap-accumulated cloud ready for
rasterization / voxelization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frame_geometry import gravity_align
from .ground_removal import (GroundRemover, GroundRemovalParams,
                             plane_level_and_remove)
from .submap import SubmapAccumulator


@dataclass
class PipelineParams:
    submap_window: int = 5
    ground: GroundRemovalParams = None
    prefer_patchwork: bool = True
    # Self-calibrating plane-fit leveling + ground removal. Replaces the
    # Patchwork path, which barely removed the (tilted) ground on this data.
    use_plane_fit: bool = True
    plane_dist_thresh: float = 0.20
    # dynamic-obstacle removal via submap voxel temporal persistence
    filter_dynamic: bool = False
    dyn_voxel: float = 0.5
    dyn_min_frames: int = 2


class FramePipeline:
    """Stateful (submap needs history) per-frame geometry processor."""

    def __init__(self, params: PipelineParams = None):
        self.params = params or PipelineParams()
        self._gr = GroundRemover(self.params.ground or GroundRemovalParams(),
                                 prefer_patchwork=self.params.prefer_patchwork)
        self._submap = SubmapAccumulator(
            self.params.submap_window, filter_dynamic=self.params.filter_dynamic,
            dyn_voxel=self.params.dyn_voxel, dyn_min_frames=self.params.dyn_min_frames)

    @property
    def ground_backend(self) -> str:
        return self._gr.backend

    def reset(self):
        self._submap.reset()

    def process(self, points: np.ndarray, position=None, quat=None) -> np.ndarray:
        """Return the accumulated leveled/ground-removed cloud for this frame.

        If a pose (position+quat) is present, points are gravity-leveled first;
        otherwise they are used as-is (still ground-removed + accumulated).
        """
        pts = points
        if quat is not None:
            pts = gravity_align(pts, quat)          # coarse (odom/GPS heading + roll/pitch)
        if self.params.use_plane_fit:
            # fine self-calibrating tilt correction + ground removal (fixes the
            # residual mount pitch Patchwork/odom-leveling leaves behind)
            nonground, _ground = plane_level_and_remove(
                pts, dist_thresh=self.params.plane_dist_thresh)
        else:
            nonground, _ground = self._gr.remove_ground(pts)
        return self._submap.push(nonground, position=position, quat=quat)

# spot_e2e — running the plan tree on Spot

**Status: scaffold, never run on hardware.**

## What exists today

`dgppo_ros_node_pkg/sampling_mpc_spot_ros_node.py` runs SamplingMPC on Spot, but it walks
a **flat list of cluster pairs** (`plan_sequence`, advanced by `_advance_plan` when the
observed cluster equals `expected_next`). It contains zero occurrences of `branch` and
zero of `brain`. That is the prior system's linear representation: Spot cannot currently
execute a plan tree.

## What has to change, and why it is small

`brain_controller` is already portable. Its whole interface is four topics —
`/predicted_cluster` (Int16), a cue topic (String), an image, and odometry — and it
publishes `/brain/state` carrying `start_cluster`, `goal_cluster`, `trigger` and
`branch_path`. Nothing in it is CARLA-specific.

So the Spot node does not need to learn about branches. It needs to stop deciding *when*
to advance and start asking the brain:

| today | with the brain |
|---|---|
| `plan_sequence[i]` gives `{start, next}` | `/brain/state` gives `start_cluster`, `goal_cluster` |
| `_advance_plan()` when cluster == next | brain advances; it owns cues, ordinals, branches, bearings |
| `_forbidden` = everything but the pair | `forbid_clusters` from the plan, if declared |
| targets a cluster id directly | `StepTargeter.target_for(goal_label)` over livox1 adjacency |

`control_loop` is the only method that needs replacing; `_run_sampling_mpc_step`,
`_apply_action`, `_update_spot_state` and the Spot SDK wiring are untouched.

## Wiring checklist

1. **Perception.** livox1 encoder + HDBSCAN exist
   (trained weights go in `clustering/clustering/encoder_weights/livox1/`; none are
   included in this repo). Publish `/predicted_cluster`.
2. **Cues.** `brain_controller` already defaults to `cue_source=vlm` (gpt-4o). Point
   `image_topic` at the ZED. No new code.
3. **Taxonomy.** `nl_planner/config/cluster_map.livox1.yaml` — Road: On,
   Intersection: Approach/Enter, Intersection: In, Intersection: Exit, Open Space,
   Along Wall. Note this taxonomy DOES split the junction into approach/in/exit, which
   CARLA's two-mode Town05 map does not.
4. **Odometry.** `odom_topic` — heading only, for `Bearing(...)` steps.
5. **Targeting.** Build `StepTargeter` from the livox1 adjacency and centroids, exactly
   as `carla_mpc_ros_node` does.

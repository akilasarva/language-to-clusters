# CARLA and Spot navigation stack

English mission → STL formula + plan tree → plan executor → sampling / terrain MPC, run in
CARLA (ground-truth perception bridge) or on Spot (LiDAR clustering + camera cues).

## Packages

| Package | What it is |
|---|---|
| `nl_planner`, `nl_planner_msgs` | English → validated plan tree + STL monitors (LLM generator, prompts in `nl_planner/nl_planner/prompts/`) |
| `brain` | Plan executor: walks the tree, owns cues, ordinals, branches, bearings; publishes `/brain/state` |
| `carla_gt_bridge` | CARLA ground-truth cluster/cue/obstacle nodes, town configs, mission launch, run/score scripts |
| `dgppo_ros_node_pkg` | Controllers: `sampling_mpc`, `terrain_mpc`, the CARLA and Spot MPC nodes, `sim2d` 2-D rig (the package name is historical; no DGPPO policy code remains) |
| `clustering` | LiDAR encoder + HDBSCAN clustering code and live inference node (no trained weights are included) |
| `bev_pipeline` | BEV perception pipeline and live node |
| `terrain_analysis` | Camera terrain segmentation nodes |
| `spot_mpc_pkg`, `spot_e2e` | Spot MPC node, and the adapter that drives Spot from `/brain/state` (scaffold, not yet run on hardware) |
| `baselines` | Waypoint-list LLM baseline |

## Setup

```bash
mkdir -p ~/ros2_ws && cd ~/ros2_ws
git clone <this repo> src
pip install 'pydantic-ai-slim[openai]' 'pydantic>=2.5' 'pyyaml>=6.0' 'openai>=1.40'
colcon build && source install/setup.bash
export OPENAI_API_KEY=...                          # plan generation and VLM cues; never commit it
```

The scripts assume the workspace is at `~/ros2_ws` (override with `WS=...`).

## CARLA

External pieces, not in this repo:

- `carlasim/carla:0.9.14` — public Docker image (server).
- `carla-ros-bridge-dev:latest` — the bridge container (ROS Humble + carla 0.9.14). It was
  built by hand and has no Dockerfile; get a saved copy of the image
  (`carla_gt_bridge/scripts/remote/export_bundle.sh` produces one) and `docker load` it.
- The CARLA PythonAPI at `~/carla` (override with `CARLA_PYTHONAPI=...`).
- An NVIDIA container runtime. `carla_gt_bridge/scripts/remote/bootstrap.sh` checks all of
  the above on a new machine.

Entry points (from `~/ros2_ws/src/carla_gt_bridge`):

```bash
python3 scripts/preflight.py                                   # check known failure modes first

# the nine missions (config/missions.yaml), two arms, ground-truth perception
python3 scripts/run_missions.py --arm ours --list              # show the runs
python3 scripts/run_missions.py --arm ours --only M3 --reps 1  # English -> plan tree -> drive
python3 scripts/run_missions.py --arm llm  --only M3 --reps 1  # LLM drives, decision by decision

# one sentence, by hand
python3 scripts/drive_english.py --english "..." --start 0 --toward 63 --dry   # plan only
python3 scripts/missions.py --simulate config/english.town05.full.json        # offline, ~1 s
python3 scripts/drive_english.py --english "..." --start 0 --toward 63        # drive in CARLA
bash scripts/run_phase_a.sh --stop                                              # tear down
```

`config/missions.yaml` holds the nine missions (English verbatim) and the CARLA world each
is driven in. Runs write raw outcomes (result, regions visited, path length) to
`reports/missions/<stamp>/<arm>/results.jsonl`; nothing is scored automatically.

Prompts: the `ours` arm generates plans with `nl_planner/nl_planner/prompts/<name>.md`
(`--prompt`, default `generator_deep`); the `llm` arm's prompt is in
`carla_gt_bridge/carla_gt_bridge/nodes/llm_brain_node.py`.

## Spot

`spot_mpc_pkg` (Spot SDK + CasADi MPC) and `dgppo_ros_node_pkg/sampling_mpc_spot_ros_node.py`.
See `spot_e2e/README.md` for wiring the plan tree onto Spot. The 2-D sampling MPC and region
associator the Spot sampling-MPC node uses are in `dgppo_ros_node_pkg/vendor/`.

## Data

Datasets, bags and run outputs are not in the repo. Training and collection scripts read
PCDs from `~/carla_data/<corpus>/<corpus>_pcds`.

## Tests

```bash
cd ~/ros2_ws/src/<package>
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=.:../nl_planner:../brain:../carla_gt_bridge:../dgppo_ros_node_pkg \
    python3 -m pytest test --ignore-glob='*flake8*' --ignore-glob='*pep257*' --ignore-glob='*copyright*' --ignore-glob='*xmllint*'
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is needed with ROS Jazzy and pytest 9: ROS's
`launch_testing` plugin fails to load otherwise. `test_lane_samples` currently has one
known failure.

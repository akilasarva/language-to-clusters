"""Every tuning knob the node declares must be REACHABLE from the launch file.

A parameter declared on a node and never passed by the launch file is invisible: it
looks configurable, and the only way to change it is to edit the node.

`mpc_K` is the MPC's rollout count and a major per-tick cost. Under
synchronous_mode + wait_for_vehicle_control_command the CARLA world advances only when
the MPC emits a command, so simulated time is gated on tick duration.

`regions_npz` defaults to Town05 inside the node, so on any other town the MPC would
score its rollouts against a different town's road network (a one-sided steering bias).
"""
import os
import re

import pytest
import pathlib

WS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LAUNCH = os.path.join(WS, "carla_gt_bridge", "launch", "mission.launch.py")
NODE = os.path.join(WS, "dgppo_ros_node_pkg", "dgppo_ros_node_pkg",
                    "carla_mpc_ros_node.py")
RUNNER = os.path.join(WS, "carla_gt_bridge", "scripts", "run_phase_a.sh")


@pytest.fixture(scope="module")
def launch_src():
    return open(LAUNCH).read()


@pytest.mark.parametrize("param", ["mpc_K", "mpc_N", "control_every_n_odom",
                                   "regions_npz", "v_max"])
def test_the_mpc_knobs_are_reachable_from_the_launch_file(launch_src, param):
    """Declared on the node AND passed by the launch file -- both halves."""
    assert f'"{param}"' in open(NODE).read(), f"{param} not declared on the node"
    assert f'"{param}"' in launch_src, f"{param} declared but never passed by launch"


@pytest.mark.parametrize("arg", ["mpc_K", "mpc_N", "control_every_n_odom"])
def test_the_new_knobs_have_launch_arguments_with_defaults(launch_src, arg):
    """Settable as `mission.launch.py mpc_K:=150`, not just hardcoded in the file."""
    assert re.search(rf'DeclareLaunchArgument\(\s*"{arg}"', launch_src), arg


def test_integer_params_are_cast_not_passed_as_strings(launch_src):
    """A launch substitution arrives as a string; rclpy rejects that for an int param.

    Without the ParameterValue cast this fails at declare time, at run time, in CARLA --
    the most expensive place to find it.
    """
    for arg in ("mpc_K", "mpc_N", "control_every_n_odom"):
        m = re.search(rf'"{arg}":\s*ParameterValue\(\s*LaunchConfiguration\(\s*"{arg}"\s*\),'
                      rf'\s*value_type=int', launch_src, re.S)
        assert m, f"{arg} must be cast with ParameterValue(..., value_type=int)"


def test_the_runner_forwards_the_mpc_knobs():
    """run_phase_a.sh is how these actually get set in practice."""
    src = open(RUNNER).read()
    for var, arg in (("MPC_K", "mpc_K"), ("MPC_N", "mpc_N"),
                     ("CONTROL_EVERY_N", "control_every_n_odom")):
        assert f"{arg}:=${var}" in src, f"{arg} not forwarded by the runner"
        assert re.search(rf'{var}="\$\{{{var}:-\d+\}}"', src), f"{var} needs a default"


def test_control_every_n_defaults_to_one_because_that_was_the_whole_problem():
    """Under wait-for-control the server holds each world tick until it receives a
    control command, and the node emits one only every Nth odometry message, so
    unserviced ticks stall on a ~1 s server timeout.

    The default must stay 1 while wait-for-control is on; n>1 silently reintroduces a
    large slowdown that looks like the simulator merely being slow.
    """
    src = open(RUNNER).read()
    assert "CONTROL_EVERY_N" in src
    assert 'CONTROL_EVERY_N="${CONTROL_EVERY_N:-1}"' in src, "default must match the node"


def test_the_cue_node_and_the_offline_oracle_share_one_implementation():
    """A cue the twin answered and the node did not, or answered differently, would mean
    the offline harness certifies plans the robot then executes differently.

    The agreement is structural rather than checked: both delegate to
    `carla_gt_bridge.cue_answers`. This asserts the delegation, and
    test_cue_answers.py asserts they agree on every cue and world.
    """
    node = open(os.path.join(WS, "carla_gt_bridge", "carla_gt_bridge", "nodes",
                             "gt_cue_node.py")).read()
    twin = open(os.path.join(WS, "carla_gt_bridge", "scripts", "missions.py")).read()
    assert "cue_answers.answers_for_world" in node, "the node rebuilt its own vocabulary"
    assert "cue_answers.resolve" in twin, "the offline twin rebuilt its own matching"


def test_only_one_carla_run_can_hold_the_entry_point():
    """A lock, because a convention is not enough.

    Gating a batch of runs on `pgrep drive_english` fails because the process clears
    during the settle between runs, so a second batch can start inside the first one's
    gap; from then on every run's `docker rm -f` tears down a LIVE container.

    The lock belongs on run_phase_a.sh rather than in a sweep script because this is the
    single entry point to CARLA -- a sweep, drive_english.py, or a hand-typed command all
    serialise on it without having to know the others exist.
    """
    whole = open(RUNNER).read()
    assert "flock" in whole, "run_phase_a.sh must take an exclusive lock"
    assert "LOCK_FILE" in whole and 'exec 9>"$LOCK_FILE"' in whole
    # Compare CODE lines only. The rationale comment above the lock quotes
    # `docker rm -f`, and indexing the raw text would find that instead of the
    # teardown.
    code = "\n".join(l for l in whole.splitlines() if not l.lstrip().startswith("#"))
    assert code.index("flock") < code.index("docker rm -f"), (
        "the lock must be acquired before the container teardown it exists to prevent")


def test_log_dir_is_passed_absolute_to_run_phase_a():
    """A relative LOG_DIR lands in a shadow tree and NOTHING warns.

    `run_phase_a.sh` runs with its own cwd (the package dir), so a relative log dir
    resolves one level too deep: mission/bridge/twist/spawn.log would land in
        src/carla_gt_bridge/carla_gt_bridge/reports/runs/<run>/
    instead of
        src/carla_gt_bridge/reports/runs/<run>/
    `save_logs` still succeeds, so no warning is printed -- the logs are simply somewhere
    nobody looks. Same class as a stale default: silent, and wrong only when the cwds
    differ.
    """
    src = (pathlib.Path(__file__).parents[1] / "scripts" / "drive_english.py").read_text()
    assert "LOG_DIR=os.path.abspath(log_dir)" in src, \
        "drive_english passes a relative LOG_DIR; run_phase_a resolves it against a " \
        "different cwd and the logs vanish into a parallel reports/ tree"


def test_the_bridge_log_is_snapshotted_mid_run_not_only_at_teardown():
    """save_logs runs in stop_all; a container that dies early takes its logs with it.

    The camera stall kills the bridge's synchronous tick thread, which is exactly the
    case where the most informative log is the one least likely to survive teardown.
    """
    rp = (pathlib.Path(__file__).parents[1] / "scripts" / "run_phase_a.sh").read_text()
    assert "snapshot_logs" in rp, "no mid-run log snapshot"
    assert "bridge" in rp.split("snapshot_logs()")[1][:400], \
        "the mid-run snapshot does not include bridge.log, which is where the " \
        "synchronous-tick-thread death is reported"

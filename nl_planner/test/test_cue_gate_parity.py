"""The cue-vocabulary gate reaches the ROBOT path, and both paths resolve it identically.

WHY THIS FILE EXISTS. Both `stl_ablation.run_one` (the harness) and
`pipeline.generate_plan` (what `nodes/planner_node.py` runs, i.e. the path a real robot
takes) must gate cues, or the harness rejects an unanswerable cue that the robot accepts.
The rule lives once, in `cue_vocab.vocab_for_deployment`, and this file asserts both
callers agree.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from nl_planner.cue_vocab import CARLA_GT_VOCAB, vocab_for_deployment
from nl_planner.pipeline import _VOCAB_FROM_DEPLOYMENT, generate_plan
from nl_planner.schemas import FilteredCommand, GeneratorOutput, NavPlan, PlanStep
from nl_planner.taxonomy import load_taxonomy, repair_medium_detect_cue

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
TAX = os.path.join(WS, "config", "cluster_map.livox1.yaml")


# --------------------------------------------------------------------------- #
# the shared resolver                                                          #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("src,closed", [("topic", True), ("vlm", False),
                                        ("TOPIC", True), (" vlm ", False)])
def test_vocabulary_follows_the_sensor(src, closed):
    got = vocab_for_deployment(src)
    assert (got is CARLA_GT_VOCAB) is closed


def test_default_is_topic_and_reads_the_env(monkeypatch):
    """`topic` is the default because a cue nothing publishes TIMES OUT rather than fails."""
    monkeypatch.delenv("CUE_SOURCE", raising=False)
    assert vocab_for_deployment() is CARLA_GT_VOCAB
    monkeypatch.setenv("CUE_SOURCE", "vlm")
    assert vocab_for_deployment() is None


def test_both_paths_resolve_the_same_vocabulary(monkeypatch):
    """THE DIVERGENCE TEST. `stl_ablation` resolves inline; `generate_plan` calls the
    helper. Same environment must mean the same vocabulary, or the harness certifies a
    plan the robot hangs on.

    This compares the helper against the inline rule TRANSCRIBED below, so on its own it
    would pass if the ablation's expression changed. `test_cue_vocab.py:142` is what pins
    the ablation source verbatim; the two tests are only meaningful together, and the
    strict xfail further down is what retires both once the call-site is swapped."""
    src = open(os.path.join(WS, "scripts", "stl_ablation.py")).read()
    assert 'os.environ.get("CUE_SOURCE", "topic")' in src, \
        "stl_ablation no longer reads CUE_SOURCE with a 'topic' default"
    for value in ("topic", "vlm"):
        monkeypatch.setenv("CUE_SOURCE", value)
        ablation = None if value == "vlm" else CARLA_GT_VOCAB   # the inline rule, verbatim
        assert vocab_for_deployment() is ablation


def test_launch_file_uses_the_same_variable_and_default():
    """`mission.launch.py` sets brain's own `cue_source` from CUE_SOURCE/'topic'."""
    p = os.path.join(os.path.dirname(WS), "carla_gt_bridge", "launch", "mission.launch.py")
    if not os.path.exists(p):                       # package not checked out beside us
        pytest.skip("carla_gt_bridge not present")
    assert 'os.environ.get("CUE_SOURCE", "topic")' in open(p).read()


def test_planner_node_mirrors_brains_default_not_the_env_default():
    """THE DEFAULTS DISAGREE, and taking the wrong one re-creates the bug.

    `brain_controller.py` declares `cue_source` defaulting to **vlm**. `CUE_SOURCE`
    falls back to **topic** in `stl_ablation` and `mission.launch.py`. Under
    mission.launch.py both end up at the same value, so the disagreement is invisible
    there. Under any OTHER launch -- a campus robot -- brain answers free text through a
    VLM while a `topic` default would make the generator reject every cue outside CARLA's
    prop families. So `planner_node` declares its own parameter with BRAIN's default
    and passes it explicitly; one launch file then sets both nodes.
    """
    node = open(os.path.join(WS, "nl_planner", "nodes", "planner_node.py")).read()
    assert 'self.declare_parameter("cue_source", "vlm")' in node,         "planner_node no longer mirrors brain's cue_source default"
    assert "cue_vocab=vocab_for_deployment(self._cue_source)" in node,         "planner_node resolves the vocabulary from somewhere other than its parameter"

    brain = os.path.join(os.path.dirname(WS), "brain", "brain", "brain_controller.py")
    if not os.path.exists(brain):
        pytest.skip("brain not present")
    assert 'self.declare_parameter("cue_source",              "vlm")' in open(brain).read(),         "brain's cue_source default moved; planner_node must follow it"


@pytest.mark.xfail(strict=True, reason=(
    "Known gap: `stl_ablation.run_one` resolves the vocabulary inline and does not "
    "call `repair_medium_detect_cue`. This xfail is strict, so it fails the suite once "
    "that is wired in -- the signal to delete it."))
def test_ablation_path_shares_the_helper_and_the_repair():
    src = open(os.path.join(WS, "scripts", "stl_ablation.py")).read()
    assert "vocab_for_deployment()" in src
    assert "repair_medium_detect_cue" in src.split("from nl_planner.taxonomy import")[1][:400]


# --------------------------------------------------------------------------- #
# the gate, in generate_plan                                                   #
# --------------------------------------------------------------------------- #

def _out(cue: str) -> GeneratorOutput:
    return GeneratorOutput(
        filtered_command=FilteredCommand(original="go", filtered="go"),
        stl_formula="",
        json_plan=NavPlan(
            plan_name="t", description="one legal step",
            steps=[PlanStep(step=0, description="walk", start_mode="path",
                            goal_mode="path", transition_cue=cue, trigger="landmark")]),
    )


class _Run:
    def __init__(self, out): self.output = out


class _Gen:
    def __init__(self, cues): self.cues, self.seen = list(cues), 0
    def run_sync(self, _msg):
        cue = self.cues[min(self.seen, len(self.cues) - 1)]
        self.seen += 1
        return _Run(_out(cue))


class _Bundle:
    def __init__(self, gen): self.generator = gen


def _gen_kwargs(**kw):
    return dict(taxonomy=load_taxonomy(TAX), verify_syntax=False,
                verify_tripartite=False, **kw)


def test_unanswerable_cue_is_rejected_on_the_robot_path(monkeypatch):
    """`Detect(Scaffolding)` publishes nothing, so the step WAITS FOREVER -- it does not
    fail. Without this gate in `generate_plan`, planner_node would return that plan."""
    monkeypatch.setenv("CUE_SOURCE", "topic")
    gen = _Gen(["Detect(Scaffolding)"])
    from nl_planner.schemas import PlanGenerationError
    with pytest.raises(PlanGenerationError) as exc:
        generate_plan("go", agents=_Bundle(gen), **_gen_kwargs())
    # ASSERT THE REASON, not just the exception: otherwise an unrelated schema error
    # would make this pass with no gate at all.
    assert "cue vocab" in str(exc.value), f"failed for another reason: {exc.value}"
    assert gen.seen > 1, "the gate did not retry; it is not wired to the feedback loop"


def test_the_retry_feedback_names_the_answerable_set(monkeypatch):
    """The prompt tells the model 'any concrete visual landmark works', which is true of a
    VLM and false of this publisher. The retry is where it finds out, so the message has
    to carry the set -- and a repaired second attempt must then be ACCEPTED."""
    monkeypatch.setenv("CUE_SOURCE", "topic")
    gen = _Gen(["Detect(Scaffolding)", "Detect(Bench)"])
    res = generate_plan("go", agents=_Bundle(gen), **_gen_kwargs())
    assert res.accepted and gen.seen == 2
    first = res.attempts[0]
    assert first.bad_vocab, "the failed attempt did not record bad_vocab"
    assert "bench" in res.attempts[1].generator_input.lower()


def test_vlm_deployment_does_not_apply_the_closed_set(monkeypatch):
    """SCOPE. Under a VLM the open set is answerable, and enforcing the CARLA families
    would reject a perfectly good campus cue. This is the mode a real robot runs in."""
    monkeypatch.setenv("CUE_SOURCE", "vlm")
    res = generate_plan("go", agents=_Bundle(_Gen(["Detect(Scaffolding)"])),
                        **_gen_kwargs())
    assert res.accepted


def test_explicit_none_beats_the_deployment_default(monkeypatch):
    """A translation-only caller passes None deliberately: against a non-CARLA corpus
    the CARLA vocabulary would reject most plans for reasons unrelated to translation."""
    monkeypatch.setenv("CUE_SOURCE", "topic")
    res = generate_plan("go", agents=_Bundle(_Gen(["Detect(Scaffolding)"])),
                        cue_vocab=None, **_gen_kwargs())
    assert res.accepted


def test_none_is_distinguishable_from_unset():
    """A plain `None` default could not tell 'resolve from the deployment' apart from
    'there is no closed set'. They are different requests."""
    assert _VOCAB_FROM_DEPLOYMENT is not None

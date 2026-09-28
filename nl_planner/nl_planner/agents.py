"""pydantic-ai Agent factories.

Two agents, each with a typed output schema. Built lazily so importing this
module does NOT require ``pydantic_ai`` to be installed (handy for unit
tests of pure-python pieces like schemas.py / taxonomy.py).

There used to be a third agent, ``syntax_verifier``, backed by
``prompts/verify_syntax.md``. It was replaced by the deterministic
``stl_syntax.quick_syntax_check`` (the LLM check was inconsistent across runs
and rejected valid formulas like ``Detect(Wall)``), and ``pipeline.py`` does
not call it.
``verify_syntax.md`` itself is KEPT: it is the human-readable spec whose rule
numbers ``stl_syntax.py`` and the retry feedback both cite.

Provider/model is passed in as a string in pydantic-ai's
``provider:model_id`` syntax, e.g. ``"openai:gpt-4.1"``. The default is
read from the ROS param / env var by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .prompts import load_prompt
from .schemas import GeneratorOutput, TripartiteVerdict

# Default: gpt-4.1. It produces schema-valid branching plans with balanced STL
# more reliably than gpt-4o and at lower cost. The -mini variants are not used as
# the primary model: they have been unreliable at producing schema-valid branching
# plans with balanced STL syntax, even with retries and explicit feedback
# (pydantic-ai's `Exceeded maximum output retries`).
DEFAULT_MODEL_ID = "openai:gpt-4.1"


@dataclass
class AgentBundle:
    """The agents the pipeline needs, all bound to the same model."""

    generator: Any                # pydantic_ai.Agent[Any, GeneratorOutput]
    tripartite_verifier: Any      # pydantic_ai.Agent[Any, TripartiteVerdict]
    model_id: str


def build_agents(model_id: str = DEFAULT_MODEL_ID) -> AgentBundle:
    """Construct the Agents wired to their .md system prompts and schemas.

    Importing ``pydantic_ai`` is deferred until this is called, so the rest of
    the package can be imported (and unit-tested) without the optional dep.
    """
    try:
        from pydantic_ai import Agent  # type: ignore
    except ImportError as exc:  # pragma: no cover - import-time guidance
        raise ImportError(
            "pydantic-ai is required for the LLM pipeline. Install with:\n"
            "    pip install 'pydantic-ai-slim[openai]'\n"
            f"(original error: {exc})"
        ) from exc

    generator = Agent(
        model_id,
        output_type=GeneratorOutput,
        system_prompt=load_prompt("generator"),
    )
    tripartite_verifier = Agent(
        model_id,
        output_type=TripartiteVerdict,
        system_prompt=load_prompt("verify_tripartite"),
    )

    return AgentBundle(
        generator=generator,
        tripartite_verifier=tripartite_verifier,
        model_id=model_id,
    )


__all__ = ["AgentBundle", "build_agents", "DEFAULT_MODEL_ID"]

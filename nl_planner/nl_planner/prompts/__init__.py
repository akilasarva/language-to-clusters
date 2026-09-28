"""Prompt loader.

System prompts live next to this file as ``.md`` so they can be edited without
touching Python. ``load_prompt(name)`` returns the file contents verbatim
(trailing newline trimmed). Names map 1-1 to filenames:

    load_prompt("generator")          -> generator.md
    load_prompt("verify_syntax")      -> verify_syntax.md
    load_prompt("verify_tripartite")  -> verify_tripartite.md
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

_HERE = Path(__file__).parent

#: Canonical set of prompt names. Used by the smoke test.
KNOWN_PROMPTS: tuple[str, ...] = (
    "generator",            # production: JSON plan + STL formula
    "generator_v3",         # generator.md with its self-contradictions resolved
    "generator_v4",         # restructured: concepts, compressed syntax, rebuilt examples
    "generator_v5",         # v4 + named specification patterns (after Lang2LTL Table 4)
    "generator_stl_only",   # two-stage, stage 1: the formula alone
    "generator_stl_only_v2",  # stage 1 + the empty-formula rule, flat ordinal example
    "generator_nostl",      # ablation arm: JSON plan only, STL never taught
    "generator_flat",       # ablation arm: linear plan, no branches -- the control
    "generator_untaxed",    # ablation arm: tree, open vocabulary
    "generator_free",       # ablation arm: linear, open vocabulary
    "verify_syntax",
    "verify_tripartite",
)
# Every prompt any arm loads belongs here, so `test_prompts_smoke.py` covers it (an
# absent prompt could be empty, malformed or missing and nothing would say so). `generator_deep`, `generator_v2` and
# `generator_noforbid` are deliberately absent: they are selected only by setting
# GENERATOR_PROMPT by hand and no script, test or document references them.


def load_prompt(name: str) -> str:
    """Read ``<name>.md`` from this directory and return its contents.

    Raises ``FileNotFoundError`` if the prompt does not exist. Caller may
    catch and surface a clearer message.
    """
    # An env override lets a variant be A/B'd without editing call sites:
    #   GENERATOR_PROMPT=generator_deep  ->  load_prompt("generator") reads
    #   generator_deep.md. Only the exact prompt named is redirected, so an
    #   override cannot silently change the verifier as well.
    override = os.environ.get(f"{name.upper()}_PROMPT")
    path = _HERE / f"{(override or name)}.md"
    if not path.exists():
        raise FileNotFoundError(
            f"prompt '{name}' not found at {path}. Known prompts: {KNOWN_PROMPTS}"
        )
    return _strip_doc_header(path.read_text(encoding="utf-8")).rstrip("\n")


_DOC_HEADER = re.compile(r"\A\s*<!--.*?-->\s*", re.DOTALL)


def _strip_doc_header(text: str) -> str:
    r"""Drop a leading ``<!-- ... -->`` block before the prompt reaches the model.

    These headers are notes for maintainers — provenance, why a variant exists, what was
    removed — and must not reach the model. E.g. a header on `generator_nostl.md` naming
    the `\Phi_{...}` macro vocabulary would tell the arm meant to have never heard of STL
    about it in its first sentence, making the ablation arms asymmetric.

    Only a header is stripped, and only one: a comment anywhere else is left alone,
    because prompt bodies use them deliberately.
    """
    return _DOC_HEADER.sub("", text, count=1)


def list_prompts() -> Iterable[str]:
    """Iterate over every ``.md`` prompt file actually present on disk."""
    for p in sorted(_HERE.glob("*.md")):
        yield p.stem


__all__ = ["load_prompt", "list_prompts", "KNOWN_PROMPTS"]

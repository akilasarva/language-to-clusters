#!/usr/bin/env python3
"""Derive the no-STL generator prompt MECHANICALLY from the current generator prompt.

WHY THIS EXISTS. The hand-forked `generator_nostl.md` does not track `generator.md`: it lacks
`## forbid_instances` -- a plan-tree JSON field, nothing to do with STL -- and
`generator_deep.md`'s branch-correctness sections, so a "with STL vs without STL" comparison
against it would really be "current prompt vs a stale fork". Its in-context examples are also
INVALID JSON: removing the `stl_formula` line left the previous field's trailing comma, so
every example ends `},\n}`.

An ablation arm that is maintained by hand drifts from the arm it is compared against. This
derives it instead, so re-deriving after any prompt change is one command.

WHAT COUNTS AS STL. Only material that is exclusively about the FORMULA is removed. Predicate
SPELLING rules are kept, because `Detect(...)` and `Bearing(...)` are also the `transition_cue`
strings in the plan tree -- dropping them would remove plan-tree teaching and reintroduce the
very confound this script exists to remove. Mistake items are NOT renumbered, so the
cross-reference "which mistake 5 explains" stays valid.
"""
import argparse, json, os, re, sys

# Sections removed whole. Each is formula-only; `## Worked Examples` and `## Predicate Args
# vs Mode Names` are subsections of the cheatsheet and go with it.
DROP_SECTIONS = (
    "# Logic Macros",
    "# What the STL Formula Is FOR",
    "# STL Syntax Cheatsheet",
)
# Numbered items under "Common Generator Mistakes" that are formula-only.
DROP_MISTAKES = ("**Inventing macros.**", "**Using ASCII boolean/temporal operators.**")

SUBS = [
    (r"\*\*branch-aware Signal Temporal Logic \(STL\) plan\*\*", "**branch-aware navigation plan**"),
    (r"all three fields \(`filtered_command`,\s*\n?\s*`json_plan`, `stl_formula`\)",
     "both fields (`filtered_command`, `json_plan`)"),
    (r"`json_plan`, `stl_formula`\)", "`json_plan`)"),
    (r"all three fields", "both fields"),
]
STL_TOKENS = re.compile(r"stl|formula|\\mathbf|\\Phi|\\land|\\lor|\\lnot|LaTeX", re.I)

#: Sentences that are formula-only but sit INSIDE a paragraph that teaches the plan tree.
#: Dropping the whole paragraph would remove plan-tree teaching and so reintroduce exactly
#: the handicap this script exists to remove (this is how `forbid_instances` goes missing).
#: These are excised sentence-wise and the paragraph is kept.
#: Individual LINES that are formula-only but sit in a block with no blank line separating
#: them from plan-tree rules. The ordinal-unroll example sits directly above "USE MACRO
#: PHASES / FILTER TRANSIENT FEATURES / MODE VOCABULARY IS CLOSED", so a paragraph-level
#: drop would delete three tree rules -- the TREE_TOKENS guard below catches that.
LINE_DROPS = [
    r"^\s*In STL, an ordinal unrolls as detect / lose sight of / detect again:\s*$",
    r"^\s*`\\text\{Detect\}\(\\text\{Bench\}\).*$",
]

SENTENCE_DROPS = [
    r'\s*The nested `\\lnot\\text\{Detect\}` in the STL is what "second" means:'
    r' see it, lose sight of it, see it again\.',
]

#: Plan-tree vocabulary. A paragraph carrying any of these is teaching the TREE, so dropping
#: it wholesale is a bug in this script, not a clean ablation. Reported, never silent.
TREE_TOKENS = re.compile(
    r"cue_ordinal|transition_cue|start_mode|goal_mode|forbid_modes|require_modes|"
    r"forbid_instances|branches|sub_plan|accept_clusters|trigger", re.I)


def heading_level(line):
    m = re.match(r"^(#+)\s", line)
    return len(m.group(1)) if m else None


def derive(text):
    lines, out, removed = text.split("\n"), [], []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        lvl = heading_level(line)
        if lvl and any(line.startswith(h) for h in DROP_SECTIONS):
            j = i + 1
            while j < n:
                l2 = heading_level(lines[j])
                if l2 and l2 <= lvl:
                    break
                j += 1
            removed.append((line.strip(), j - i))
            i = j
            continue
        # a formula-only numbered mistake: drop the item through to the next item/heading
        if any(k in line for k in DROP_MISTAKES) and re.match(r"^\d+\.\s", line.strip()):
            j = i + 1
            while j < n and not re.match(r"^\d+\.\s", lines[j].strip()) \
                    and heading_level(lines[j]) is None:
                j += 1
            removed.append((line.strip()[:48], j - i))
            i = j
            continue
        # the `stl_formula` output-field bullet, and its continuation lines
        if line.lstrip().startswith("- `stl_formula`"):
            j = i + 1
            while j < n and lines[j].startswith("  ") and not lines[j].lstrip().startswith("- "):
                j += 1
            removed.append(("- `stl_formula` output field", j - i))
            i = j
            continue
        if re.match(r'^\s*"stl_formula"\s*:', line):
            # the example JSON's formula line. The comma on the PREVIOUS kept line now
            # dangles; strip it, or the example teaches malformed JSON (which is exactly
            # what the hand-forked generator_nostl.md does).
            for k in range(len(out) - 1, -1, -1):
                if out[k].strip():
                    out[k] = re.sub(r",\s*$", "", out[k])
                    break
            removed.append(("example stl_formula line", 1))
            i += 1
            continue
        if any(re.match(pat, line) for pat in LINE_DROPS):
            removed.append((f"line: {line.strip()[:52]}", 1))
            i += 1
            continue
        out.append(line)
        i += 1

    txt = "\n".join(out)
    for pat, rep in SUBS:
        txt = re.sub(pat, rep, txt)
    for pat in SENTENCE_DROPS:
        txt, k = re.subn(pat, "", txt)
        if k:
            removed.append((f"sentence x{k} (paragraph kept)", 0))
    # drop any remaining paragraph that is purely about the formula
    paras, keep = txt.split("\n\n"), []
    for p in paras:
        if STL_TOKENS.search(p) and "```" not in p:
            tag = "" if not TREE_TOKENS.search(p) else "  <-- ALSO TEACHES THE TREE"
            removed.append((p.strip().replace("\n", " ")[:60] + tag, p.count("\n") + 1))
            continue
        keep.append(p)
    return "\n\n".join(keep), removed


def json_blocks(text):
    """Every fenced json block, for a validity check."""
    return re.findall(r"```json\s*\n(.*?)```", text, re.S)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="nl_planner/prompts/generator_deep.md")
    ap.add_argument("--out", default="nl_planner/prompts/generator_deep_nostl.md")
    ap.add_argument("--check", default="", help="also JSON-check this existing prompt")
    a = ap.parse_args()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(here, a.src)
    text = open(src).read()
    derived, removed = derive(text)

    print(f"{os.path.basename(src)}: {text.count(chr(10))+1} lines -> "
          f"{derived.count(chr(10))+1} lines")
    for what, k in removed:
        print(f"   -{k:3} lines  {what}")

    leftover = [l for l in derived.split("\n") if STL_TOKENS.search(l)]
    if leftover:
        print("\nSTL RESIDUE REMAINS -- not written:")
        for l in leftover[:10]:
            print("   ", l.strip()[:100])
        return 2

    for label, blob in (("derived", derived),) + ((("existing", open(
            os.path.join(here, a.check)).read()),) if a.check else ()):
        bad = 0
        for b in json_blocks(blob):
            try:
                json.loads(b)
            except Exception as e:                                   # noqa: BLE001
                bad += 1
                print(f"  [{label}] INVALID example JSON: {e}")
        print(f"  [{label}] {len(json_blocks(blob))} example JSON blocks, {bad} invalid")
        if label == "derived" and bad:
            return 3

    out = os.path.join(here, a.out)
    with open(out, "w") as f:
        f.write(derived)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

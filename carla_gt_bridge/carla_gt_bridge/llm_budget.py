"""Spend-capped OpenAI chat client.

Every call goes through `call()`, which (1) refuses the call if the worst-case cost would
cross the cap, (2) records the actual cost from `usage` in a JSONL ledger. File-locked so
parallel processes share one total.

Env:
  LLM_CAP_USD      spend cap in USD (default 30)
  LLM_LEDGER_PATH  ledger file (default ~/.cache/carla_gt_bridge/spend_ledger.jsonl)
"""
import fcntl, json, os, time, pathlib
LEDGER = pathlib.Path(os.environ.get("LLM_LEDGER_PATH", "")
                      or os.path.expanduser("~/.cache/carla_gt_bridge/spend_ledger.jsonl"))
CAP_USD = float(os.environ.get("LLM_CAP_USD", "30.0"))
PRICES = {"gpt-4o": (2.50, 10.00), "gpt-4.1": (2.00, 8.00), "gpt-4.1-mini": (0.40, 1.60),
          "gpt-4o-mini": (0.15, 0.60), "gpt-5.6-terra": (2.00, 12.00), "gpt-5.6-luna": (0.20, 1.20)}

class BudgetExceeded(RuntimeError): pass

def spent() -> float:
    if not LEDGER.exists(): return 0.0
    return sum(json.loads(l)["usd"] for l in LEDGER.read_text().splitlines() if l.strip())

def _cost(model, i, o):
    pi, po = PRICES[model]; return i / 1e6 * pi + o / 1e6 * po

import openai as _openai
_RealOpenAI = _openai.OpenAI   # captured at import so a later monkeypatch of openai.OpenAI cannot recurse into us
_client = None
def call(model, messages, tag, est_in=4000, max_out=4000, **kw):
    """chat.completions with a hard cap. Worst case = est_in input + max_out output."""
    global _client
    if _client is None: _client = _RealOpenAI(timeout=120, max_retries=3)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    worst = _cost(model, est_in, max_out)
    with open(str(LEDGER) + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if spent() + worst > CAP_USD:
            raise BudgetExceeded(f"cap ${CAP_USD} would be exceeded (spent ${spent():.3f})")
    budget = {"max_tokens": max_out} if model.startswith("gpt-4") else {"max_completion_tokens": max_out}
    r = _client.chat.completions.create(model=model, messages=messages, **budget, **kw)
    u = r.usage; usd = _cost(model, u.prompt_tokens, u.completion_tokens)
    with open(str(LEDGER) + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        with open(LEDGER, "a") as f:
            f.write(json.dumps({"t": time.time(), "tag": tag, "model": model, "in": u.prompt_tokens,
                                "out": u.completion_tokens, "usd": usd}) + "\n")
    return (r.choices[0].message.content or "").strip(), u

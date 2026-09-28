"""VLM-based auto-labeler (a stand-in for human bbox annotation).

This is *not* part of the live inference path and is *not* the long-term
labeling strategy — the GUI bbox editor + geometric trajectory labeler remain
the precise path. It turns raw camera frames into ``(phase, landmark_type)``
labels with no human in the loop.

Design:

  * The VLM tags BOTH the landmark *type* and the *phase* (approach / on /
    along / exit / open_road), choosing only from the closed vocabulary built
    from ``landmark_types.yaml`` — it cannot invent a state nl_planner has
    never seen.
  * It is given a rolling window of the most recent frames it has already
    labeled (thumbnail + assigned label), so it can reason about phase from the
    visual *trend* (a building growing in view => approaching; shrinking =>
    exiting) rather than guessing from a single still.
  * It degrades safely: on any API/parse failure after retries it returns
    ``open_road`` at zero confidence and records the error, so an unattended
    run never stalls.

Cost: one chat-completion call per frame with 1-2 low-detail images. At ~1
frame / 3 s over the target bags this is a few hundred calls, well under a
dollar total.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .label_vocab import LandmarkVocab


@dataclass
class FrameLabel:
    """One frame's VLM label result."""

    label: str
    landmark_type: Optional[str]
    phase: Optional[str]
    confidence: float
    reason: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "landmark_type": self.landmark_type,
            "phase": self.phase,
            "confidence": self.confidence,
            "reason": self.reason,
            "error": self.error,
        }


@dataclass
class _HistoryEntry:
    jpeg_b64: str
    label: str


def _rgb_to_jpeg_b64(rgb: np.ndarray, max_side: int = 512, quality: int = 70) -> str:
    """Encode an (H,W,3) uint8 RGB array to a base64 JPEG data payload."""
    from PIL import Image

    img = Image.fromarray(np.ascontiguousarray(rgb.astype(np.uint8)))
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((max(1, int(img.size[0] * scale)),
                          max(1, int(img.size[1] * scale))))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class VLMLabeler:
    """Stateful auto-labeler that keeps a rolling frame/label history."""

    def __init__(
        self,
        vocab: LandmarkVocab,
        *,
        model: str = "gpt-4o",
        history_len: int = 4,
        max_retries: int = 4,
        api_key: Optional[str] = None,
        detail: str = "low",
    ):
        self.vocab = vocab
        self.model = model
        self.history_len = history_len
        self.max_retries = max_retries
        self.detail = detail
        self._history: List[_HistoryEntry] = []
        self._client = None
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")

    # ------------------------------------------------------------------ #
    def _client_lazy(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI(api_key=self._api_key)
        return self._client

    def reset(self) -> None:
        """Clear rolling history (call between bags)."""
        self._history = []

    # ------------------------------------------------------------------ #
    def _system_prompt(self) -> str:
        pass_through = [t for t, k in self.vocab.kinds.items() if k == "pass_through"]
        perimeter = [t for t, k in self.vocab.kinds.items() if k == "perimeter"]
        return (
            "You are labeling the behavioral navigation state of a ground robot "
            "from its forward camera, for a mobile-robot planner.\n"
            "You are shown the CURRENT frame and the few most recent frames you "
            "already labeled (with their labels), oldest to newest. Use the "
            "visual TREND across frames to judge phase:\n"
            "  - approach_X: X is ahead and growing in view (getting closer)\n"
            "  - on_X:       the robot is passing THROUGH X right now "
            "(pass-through landmarks only: {pt})\n"
            "  - along_X:    the robot is skirting ALONGSIDE X, not through it "
            "(perimeter landmarks only: {pm})\n"
            "  - exit_X:     X was just passed and is now shrinking/receding\n"
            "  - open_road:  no salient landmark; ordinary road / open space\n"
            "Choose EXACTLY ONE label from the allowed set. Prefer open_road "
            "unless a landmark is clearly salient. Do not invent labels.\n"
            "Respond with STRICT JSON only: "
            '{{"label": "<one of allowed>", "landmark_type": "<type or null>", '
            '"phase": "<approach|on|along|exit|null>", '
            '"confidence": <0..1>, "reason": "<short>"}}'
        ).format(pt=", ".join(pass_through) or "(none)",
                 pm=", ".join(perimeter) or "(none)")

    def _user_content(self, jpeg_b64: str) -> list:
        content = []
        # rolling history (oldest -> newest), each with its label
        for h in self._history[-self.history_len:]:
            content.append({"type": "text", "text": f"[recent frame — labeled: {h.label}]"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{h.jpeg_b64}",
                              "detail": self.detail},
            })
        content.append({"type": "text",
                        "text": ("CURRENT frame to label. Allowed labels: "
                                 + ", ".join(self.vocab.labels))})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64}",
                          "detail": self.detail},
        })
        return content

    # ------------------------------------------------------------------ #
    def _parse(self, text: str) -> FrameLabel:
        """Parse the model's JSON, clamping to the closed vocabulary."""
        raw = text.strip()
        # tolerate ```json fences
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            # last-ditch: find first {...}
            i, j = raw.find("{"), raw.rfind("}")
            if 0 <= i < j:
                try:
                    obj = json.loads(raw[i:j + 1])
                except json.JSONDecodeError:
                    return FrameLabel(self.vocab.open_label, None, None, 0.0,
                                      error="unparseable")
            else:
                return FrameLabel(self.vocab.open_label, None, None, 0.0,
                                  error="unparseable")

        label = str(obj.get("label", "")).strip()
        if not self.vocab.is_valid(label):
            # try to rebuild from phase+type
            phase = obj.get("phase")
            lt = obj.get("landmark_type")
            rebuilt = f"{phase}_{lt}" if phase and lt else None
            label = rebuilt if (rebuilt and self.vocab.is_valid(rebuilt)) \
                else self.vocab.open_label
        # derive type/phase from the (valid) label for consistency
        if label == self.vocab.open_label:
            lt, phase = None, None
        else:
            phase, _, lt = label.partition("_")
        conf = obj.get("confidence", 0.0)
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.0
        return FrameLabel(label, lt, phase, max(0.0, min(1.0, conf)),
                          reason=str(obj.get("reason", ""))[:200])

    # ------------------------------------------------------------------ #
    def label_frame(self, rgb: np.ndarray) -> FrameLabel:
        """Label one RGB frame, updating rolling history."""
        jpeg_b64 = _rgb_to_jpeg_b64(rgb)
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": self._user_content(jpeg_b64)},
        ]
        result: Optional[FrameLabel] = None
        last_err = None
        for attempt in range(self.max_retries):
            try:
                client = self._client_lazy()
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=120,
                    temperature=0.0,
                )
                text = resp.choices[0].message.content or ""
                result = self._parse(text)
                break
            except Exception as e:                       # noqa: BLE001 - want resilience
                last_err = str(e)
                time.sleep(min(2 ** attempt, 30))
        if result is None:
            result = FrameLabel(self.vocab.open_label, None, None, 0.0,
                                error=f"api_failed: {last_err}")
        # update history with the current frame + assigned label
        self._history.append(_HistoryEntry(jpeg_b64=jpeg_b64, label=result.label))
        if len(self._history) > self.history_len:
            self._history = self._history[-self.history_len:]
        return result

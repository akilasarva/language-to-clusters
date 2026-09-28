#!/usr/bin/env python3
"""Offline VLM-assisted audit of labels (QA only — not the live path).

For a stratified sample of frames, ask the VLM whether the camera image is
consistent with the assigned label, and report per-label agreement. A sanity
check that the (geometry or VLM) labels are semantically meaningful, without a
human eyeballing every frame. Needs OPENAI_API_KEY.

Reuses the show-image convention from the legacy cluster_training workflow;
follows nl_planner/vlm_client's gpt-4o pattern.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)


def _describe(label):
    if label == "open_road" or label == "open_space":
        return "open space / no salient nearby structure"
    if "_" in label:
        phase, ltype = label.split("_", 1)
        return f"{phase} a {ltype}"
    return label


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env-dir", required=True)
    ap.add_argument("--label-set", choices=["vlm", "geom"], default="geom")
    ap.add_argument("--per-label", type=int, default=8)
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from PIL import Image
    import base64
    import io
    import openai

    with open(os.path.join(args.env_dir, "dataset_meta.json")) as f:
        dm = json.load(f)
    image_paths = dm["image_paths"]
    if args.label_set == "geom":
        y = np.load(os.path.join(args.env_dir, "labels_geom.npy"))
        names = dm["geom_label_names"]
    else:
        y = np.load(os.path.join(args.env_dir, "labels.npy"))
        names = dm["label_names"]

    client = openai.OpenAI()
    by_label = defaultdict(list)
    for i, lab in enumerate(y):
        by_label[int(lab)].append(i)

    results = {}
    for lab_id, idxs in by_label.items():
        lab = names[lab_id]
        step = max(1, len(idxs) // args.per_label)
        sample = [i for i in idxs[::step][:args.per_label]
                  if image_paths[i] and os.path.exists(os.path.join(args.env_dir, image_paths[i]))]
        agree = 0
        for i in sample:
            img = Image.open(os.path.join(args.env_dir, image_paths[i])).convert("RGB")
            img.thumbnail((512, 512))
            buf = io.BytesIO(); img.save(buf, format="JPEG", quality=70)
            b64 = base64.b64encode(buf.getvalue()).decode()
            prompt = (f"A robot's navigation state here is labeled '{lab}' "
                      f"(meaning: {_describe(lab)}). Looking at this forward "
                      f"camera image, is that label plausible? Answer only YES or NO.")
            try:
                r = client.chat.completions.create(
                    model=args.model, max_tokens=3, temperature=0.0,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}}]}])
                if "YES" in (r.choices[0].message.content or "").upper():
                    agree += 1
            except Exception as e:                    # noqa: BLE001
                print(f"  [warn] VLM call failed on frame {i}: {e}")
        rate = agree / len(sample) if sample else 0.0
        results[lab] = {"n": len(sample), "agree": agree, "agreement_rate": round(rate, 3)}
        print(f"  {lab:16s} agreement {agree}/{len(sample)} = {rate:.2f}")

    overall = (sum(v["agree"] for v in results.values())
               / max(1, sum(v["n"] for v in results.values())))
    summary = {"env": dm["env"], "label_set": args.label_set,
               "overall_agreement": round(overall, 3), "per_label": results}
    out = args.out or os.path.join(PKG, "reports", f"qa_review_{dm['env']}_{args.label_set}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\noverall VLM-label agreement: {overall:.2f}  -> {out}")


if __name__ == "__main__":
    main()

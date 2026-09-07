"""Run a single ablation by name from the registry.

Usage:
    python scripts/ablate.py \\
        --name V8a_transition_validity \\
        --base-config configs/default.yaml \\
        --out artifacts/ablations/V8a.json
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from src.config import load_config
from src.eval.ablations import ABLATION_REGISTRY


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True, choices=list(ABLATION_REGISTRY.keys()))
    ap.add_argument("--base-config", default="configs/default.yaml")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = load_config(args.base_config)
    spec = ABLATION_REGISTRY[args.name]
    overridden = copy.deepcopy(base)
    for k, v in spec["overrides"].items():
        if isinstance(v, dict) and k in overridden and isinstance(overridden[k], dict):
            overridden[k].update(v)
        else:
            overridden[k] = v
    out = {
        "name": args.name,
        "kind": spec["kind"],
        "hypothesis": spec["hypothesis"],
        "base_config": base,
        "overridden_config": overridden,
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[ablate] wrote {p}")


if __name__ == "__main__":
    main()

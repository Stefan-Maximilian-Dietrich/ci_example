from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .core import fit_logreg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="example-repo",
        description="Train a logistic regression on synthetic data and print metrics.",
    )
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--n-features", type=int, default=10)
    p.add_argument("--noise", type=float, default=0.5)
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument("--json", action="store_true", help="Print output as JSON.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    res = fit_logreg(
        n_samples=args.n_samples,
        n_features=args.n_features,
        noise=args.noise,
        test_size=args.test_size,
        C=args.C,
        seed=args.seed,
        max_iter=args.max_iter,
    )

    payload = {
        "train_accuracy": res.train_accuracy,
        "test_accuracy": res.test_accuracy,
        "n_train": res.n_train,
        "n_test": res.n_test,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"Train acc: {payload['train_accuracy']:.3f} | "
            f"Test acc: {payload['test_accuracy']:.3f} | "
            f"n_train={payload['n_train']} n_test={payload['n_test']}"
        )

    return 0
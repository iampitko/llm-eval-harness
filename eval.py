#!/usr/bin/env python3
"""Run both matchers over the labelled dataset and report ranking metrics.

Usage:  python eval.py [--data data/pairs.jsonl] [--k 3]

Writes results.json and RESULTS.md next to this file and prints the table.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from matchers import KeywordMatcher, SemanticMatcher
from metrics import evaluate

HERE = Path(__file__).parent


def load_pairs(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rank_for_query(matcher, profile: str, jobs: list[dict]) -> list[int]:
    scores = matcher.score(profile, [j["job"] for j in jobs])
    order = sorted(range(len(jobs)), key=lambda i: scores[i], reverse=True)
    return [jobs[i]["label"] for i in order]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "data" / "pairs.jsonl"))
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    pairs = load_pairs(Path(args.data))
    queries: dict[str, list[dict]] = defaultdict(list)
    profiles: dict[str, str] = {}
    for p in pairs:
        queries[p["query_id"]].append(p)
        profiles[p["query_id"]] = p["profile"]

    corpus = [p["job"] for p in pairs] + [p["profile"] for p in pairs]
    matchers = [KeywordMatcher(), SemanticMatcher()]
    for m in matchers:
        m.fit(corpus)

    results: dict[str, dict[str, float]] = {}
    for m in matchers:
        rankings = [rank_for_query(m, profiles[qid], jobs) for qid, jobs in queries.items()]
        results[m.name] = evaluate(rankings, k=args.k)

    n_rel = sum(p["label"] for p in pairs)
    payload = {
        "dataset": {"pairs": len(pairs), "queries": len(queries), "relevant": n_rel},
        "k": args.k,
        "results": results,
    }
    (HERE / "results.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Evaluation results",
        "",
        f"Dataset: {len(pairs)} labelled pairs, {len(queries)} profiles, "
        f"{n_rel} relevant pairs. Metrics are macro-averaged per profile.",
        "",
        f"| Matcher | P@{args.k} | R@{args.k} | MRR |",
        "|---|---|---|---|",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r[f'precision@{args.k}']:.3f} | "
            f"{r[f'recall@{args.k}']:.3f} | {r['mrr']:.3f} |"
        )
    lines.append("")
    (HERE / "RESULTS.md").write_text("\n".join(lines))

    print("\n".join(lines))


if __name__ == "__main__":
    main()

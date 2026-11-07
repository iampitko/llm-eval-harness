# llm-eval-harness

A small, runnable evaluation harness that measures how well two job↔candidate
matching strategies actually rank — instead of trusting that one "feels"
smarter than the other.

It scores a hand-labelled dataset of job posts and candidate profiles with:

1. **Keyword baseline** — TF-IDF word n-gram cosine similarity
   (`scikit-learn`). This is how a lot of production matching starts out.
2. **Semantic matcher** — dense embeddings from `BAAI/bge-small-en-v1.5`,
   run locally via [fastembed](https://github.com/qdrant/fastembed) (ONNX,
   no API key, no GPU). If fastembed or the model cannot load, the harness
   degrades to a char-ngram + word-overlap matcher **and labels it as a
   fallback in the results** — it never reports embedding metrics for a run
   that did not use embeddings.

## Why evals

Swapping a keyword matcher for embeddings is a common "AI upgrade" — and an
easy one to ship blind. A labelled set with ranking metrics (precision@k,
recall@k, MRR) turns that decision into a measurement: you can see whether
the new matcher actually puts more relevant items in the top-k, which queries
it still fails, and whether a later change is a regression. This repo is a
minimal reference implementation of that loop: labelled data in, metrics
table out.

## Dataset

`data/pairs.jsonl` — 40 short, original, hand-written pairs: 8 candidate
profiles × 5 job posts each, 16 labelled relevant. It deliberately includes
traps for keyword matching (a nurse who must "react quickly", a zoo hiring
for "pythons", a "chief" executive vs a chef) and paraphrased positives with
little word overlap, which is where semantic matching is supposed to earn
its keep.

## Results (measured, this dataset)

| Matcher | P@3 | R@3 | MRR |
|---|---|---|---|
| keyword-tfidf | 0.583 | 0.875 | 1.000 |
| semantic-embeddings (BAAI/bge-small-en-v1.5 via fastembed) | 0.667 | 1.000 | 1.000 |

Embeddings recover every relevant job into the top-3 (R@3 1.000 vs 0.875)
and put more of them there (P@3 0.667 vs 0.583). Both rank a relevant job
first for every profile (MRR 1.000) — on a set this small, MRR saturates
and stops discriminating; P@k/R@k carry the signal.

## Run it

```bash
pip install -r requirements.txt
python eval.py            # prints the table, writes results.json + RESULTS.md
pytest                    # metric unit tests
python eval.py --k 5      # different cut-off
```

## Layout

- `eval.py` — CLI: loads pairs, groups by profile, ranks, reports.
- `matchers.py` — the two matchers (plus the honestly-named fallback).
- `metrics.py` — precision@k, recall@k, reciprocal rank, macro-averaging.
- `data/pairs.jsonl` — the labelled set (one JSON object per line).
- `tests/test_metrics.py` — unit tests for the metric functions.
- `results.json` / `RESULTS.md` — output of the latest run.

## Limitations

- 40 pairs is a smoke-test set, not a benchmark: numbers show the harness
  works and illustrate the keyword-vs-semantic gap; they would not survive
  as claims about either method in general.
- Labels are one annotator's judgement; a few pairs are genuinely arguable
  (e.g. an Angular role for a React profile).
- The embedding model is small and English-only; multilingual or
  domain-specific text needs a different model and a bigger labelled set.
- Metrics are ranking-only — no latency, cost or threshold calibration,
  which a production rollout would also need.

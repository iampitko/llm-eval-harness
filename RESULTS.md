# Evaluation results

Dataset: 40 labelled pairs, 8 profiles, 16 relevant pairs. Metrics are macro-averaged per profile.

| Matcher | P@3 | R@3 | MRR |
|---|---|---|---|
| keyword-tfidf | 0.583 | 0.875 | 1.000 |
| semantic-embeddings (BAAI/bge-small-en-v1.5 via fastembed) | 0.667 | 1.000 | 1.000 |

# Reference-aligned experiment: 5,000 studies

## Execution

- Run: [GitHub Actions - 5,000 record reproduction](https://github.com/shubharth2006/Team19-Clinical-Trial-Keyword-Recommender/actions/runs/38131877004)
- Dataset: shuffled streaming sample of 5,000 studies with usable pre-assigned MeSH condition/intervention terms
- Vocabulary: 134 frequent terms
- Split: 4,000 train / 500 validation / 500 test
- Parameters: tau=-0.001, k=1000, alpha=0.1, lambda=0.01, 5 epochs, seed=42
- Masking: 80% of positive terms in each held-out study

## Test results

| Metric | Collaborative filtering | Popularity baseline | Random baseline |
|---|---:|---:|---:|
| AP@3 / Precision@3 | 0.02539 | 0.00993 | 0.00110 |
| MSE | 0.00463 | - | - |
| MSER | 0.00463 | - | - |

For this single seeded run, collaborative filtering's test AP@3 is about 2.56 times the popularity baseline and about 23 times the random baseline. These ratios are descriptive for this run only, not estimates of expected performance.

## Interpretation and limitations

The experiment pipeline completed successfully: streaming download, matrix construction, training, held-out evaluation, model saving and vocabulary saving all completed. The training-loss sequence decreased from 0.19662 before updates to 0.19525 after five epochs.

This result uses pre-assigned MeSH condition/intervention annotations from the Hugging Face dataset. It does not reproduce the original XML pipeline's MeSH-trie matching across six text fields. Therefore its scores are not directly comparable with the upstream project's reported AP@K. The test AP@3 is still low in absolute terms; this run supports pipeline functionality and a preliminary comparison, not a claim of high recommendation quality. More seeds and manual relevance inspection would be needed for a stronger conclusion.

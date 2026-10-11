# Reference-aligned reproduction notes

This mini-project intentionally reproduces the task and core method of the public Clinical-Trial-Keyword-Recommendation implementation, rather than proposing a new algorithm.

## Upstream code used

- https://github.com/2005-Aneeshdutt/Clinical-Trial-Keyword-Recommendation
- Original license notice: LICENSES/Xiao-Zhou-MIT.txt (MIT; copyright notice retained)
- Original methodological paper: https://cs229.stanford.edu/proj2020spr/report/Zhou.pdf

The copied model/evaluation classes preserve the upstream formulation:
- term-document log-TF-IDF utility matrix
- cosine-similarity-derived term-neighbor matrix
- ReLU prediction with L2 regularization
- mini-batch gradient descent
- masked-keyword evaluation with MSE/MSER and AP@K

## Practical workaround

The original pipeline ingests a historical ClinicalTrials.gov XML archive and uses a MeSH trie to recognize terms in selected text sections. For easier reproducibility, the current adapter consumes the nested MeSH condition/intervention annotations in the public Hugging Face Clinical Trials dataset. This avoids requiring the historical XML archive and annual MeSH descriptor files.

The workaround is not identical in data semantics: the vocabulary is built from pre-assigned condition/intervention MeSH annotations, not from every MeSH synonym matched against all six original text attributes. The metrics are therefore project results and must not be compared numerically as if they were an exact replication on the original corpus.

## Reproducibility changes

- fixed random seed controls the document split and evaluation masking
- saved model and vocabulary artifacts make the demo repeatable
- configurable record and vocabulary caps fit a student laptop
- added a popularity baseline beside the reference implementation's random baseline

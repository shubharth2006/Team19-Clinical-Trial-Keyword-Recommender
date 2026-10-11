# Collaborative Filtering on Keywords Recommendation for Clinical Trial Records

**UE24CS352A - Machine Learning Mini-Project | Problem 14 | Team 19**  
Shubharth Upadhyay (PES2UG24CS499) and Suraj Acharya (PES2UG24CS539)

This project reproduces the assigned keyword-recommendation task using the methodology in Zhou's CS229 report and the public implementation linked under THIRD_PARTY_NOTICES.md. It is not presented as a new algorithm.

> Academic information-retrieval prototype only; not a clinical decision-support system.

## Primary implementation

The main experiment in scripts/run_reference_model.py uses the reference model/evaluation classes in ml/. Default hyperparameters follow the reference repository's final-training script; they can be changed through command-line arguments.

- MeSH term-document log-TF-IDF utility matrix
- cosine-similarity-derived term-neighbor matrix
- ReLU prediction layer with L2 regularization
- mini-batch gradient descent
- masked-keyword evaluation using MSE, MSER and AP@3
- random and additional popularity baselines
- saved model/vocabulary artifacts for the Streamlit interface

Core reference model/evaluation files are reused under the MIT license. Upstream credit and the license are preserved in THIRD_PARTY_NOTICES.md and LICENSES/Xiao-Zhou-MIT.txt.

## Quick start (offline smoke test)

Create an environment using:

    python -m venv .venv

Windows activation:

    .venv\Scripts\activate

macOS/Linux activation:

    source .venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

First run the synthetic offline smoke test (no network download needed):

    python scripts/smoke_test_reference.py

Then run the included 15-record pilot to verify data parsing and end-to-end evaluation:

    python scripts/run_reference_model.py --input data/sample_trials.csv --min-doc-freq 1 --max-terms 134 --epochs 5

This small pilot is only a smoke test; it cannot support meaningful generalization or model-superiority claims.

## Recommended scaled experiment

The closest public reproduction reports 55,634 clinical-trial documents and 134 MeSH terms. To keep the scale comparable without downloading the historical XML archive, this repository provides an adapter for a current ClinicalTrials.gov-derived dataset containing pre-assigned MeSH condition/intervention annotations.

Download a shuffled 5,000-record pilot first:

    python scripts/download_hf_sample.py --rows 5000 --out data/clinical_trials_5000.csv --seed 42
    python scripts/run_reference_model.py --input data/clinical_trials_5000.csv --min-doc-freq 5 --max-terms 134 --epochs 5 --batch-size 10000

For a larger run near the reference project's reported corpus size:

    python scripts/download_hf_sample.py --rows 55634 --out data/clinical_trials_55634.csv --seed 42
    python scripts/run_reference_model.py --input data/clinical_trials_55634.csv --min-doc-freq 5 --max-terms 134 --epochs 5 --batch-size 10000 --seed 42

Then launch the demo using the trained artifacts:

    streamlit run app/reference_app.py

Training outputs are saved locally in output/; evaluation metrics are saved in results/reference_metrics.csv and results/reference_experiment.json. Large downloaded data and trained model artifacts are intentionally not committed to GitHub.

## Automated validation

The reference-model CI workflow compiles the project, runs a controlled model smoke test, and exercises the complete training/evaluation CLI on the bundled 15-record sample. The data reproduction workflow downloads either 5,000, 20,000, or 55,634 MeSH-labelled records, fits the model, evaluates hidden-term recovery, checks recommendation inference, verifies Streamlit startup, and uploads the run artifacts.

A reference-scale run with **55,634 studies** completed successfully. Its split was 44,507 train / 5,563 validation / 5,564 test with 134 terms. Test Top-3 precision was **0.03033**, compared with **0.01165** for popularity and **0.00532** for random selection; test MSE/MSER was **0.00473**. Training, CLI inference and Streamlit startup passed. See [the measured report](results/REFERENCE_RUN_55634.md), [CSV metrics](results/reference_metrics_55634.csv), and [GitHub Actions run](https://github.com/shubharth2006/Team19-Clinical-Trial-Keyword-Recommender/actions/runs/38132544175).

These are results from one seed and the current dataset adapter, not an exact reproduction of the original XML-to-MeSH-text matching pipeline. The adapter's MeSH annotation workaround and absolute Top-3 precision limitations are documented in the report.

## Dataset and reproduction caveat

Dataset source: https://huggingface.co/datasets/louisbrulenaudet/clinical-trials

The data source provides MeSH condition/intervention annotations. The original paper's pipeline uses a historical XML archive plus a MeSH trie to match MeSH terms in six clinical-trial text fields. The adapter here uses the pre-assigned annotations as a practical workaround. The algorithm and evaluation are reference-aligned, but the data representation is not identical, so results are **not directly comparable** with the upstream project's reported scores.

Use --rows 0 only if you intend to stream every eligible record. Downloading all rows may require significant time and disk space, and is not necessary for the recommended student-scale reproduction.

## Dataset adapter

The module src/data_utils.py parses the exported MeSH annotation field and keeps memory usage manageable by reading only the trial ID and MeSH columns. The original text fields can be included in the download with --include-text if qualitative examples are needed.

## References

1. Zhou, X. *Collaborative Filtering on Keywords Recommendation for Clinical Trial Records*, Stanford CS229 Project, 2020: https://cs229.stanford.edu/proj2020spr/report/Zhou.pdf
2. Reference implementation: https://github.com/2005-Aneeshdutt/Clinical-Trial-Keyword-Recommendation
3. Dataset: https://huggingface.co/datasets/louisbrulenaudet/clinical-trials
4. ClinicalTrials.gov: https://clinicaltrials.gov/

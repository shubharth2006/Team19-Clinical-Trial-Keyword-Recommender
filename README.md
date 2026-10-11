# Collaborative Filtering on Keywords Recommendation for Clinical Trial Records

**UE24CS352A - Machine Learning Mini-Project | Problem 14 | Team 19**  
Shubharth Upadhyay (PES2UG24CS499) and Suraj Acharya (PES2UG24CS539)

This project reproduces the assigned keyword-recommendation task using the methodology in Zhou's CS229 report and the public implementation linked under THIRD_PARTY_NOTICES.md. It is not presented as a new algorithm.

> Academic information-retrieval prototype only; not a clinical decision-support system.

## Primary implementation

The main experiment in scripts/run_reference_model.py uses the reference model/evaluation classes in ml/:

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

Run the included 15-record pilot to verify the pipeline:

    python scripts/run_reference_model.py --input data/sample_trials.csv --min-doc-freq 1 --max-terms 134 --epochs 5

This small pilot is only a smoke test; it cannot support meaningful generalization or model-superiority claims.

## Recommended scaled experiment

The closest public reproduction reports 55,634 clinical-trial documents and 134 MeSH terms. To keep the scale comparable without downloading the historical XML archive, this repository provides an adapter for a current ClinicalTrials.gov-derived dataset containing pre-assigned MeSH condition/intervention annotations.

Download a shuffled 5,000-record pilot first:

    python scripts/download_hf_sample.py --rows 5000 --out data/clinical_trials_5000.csv --seed 42
    python scripts/run_reference_model.py --input data/clinical_trials_5000.csv --min-doc-freq 5 --max-terms 134 --epochs 5 --batch-size 10000

For a larger run near the reference project's reported corpus size:

    python scripts/download_hf_sample.py --rows 55634 --out data/clinical_trials_55634.csv --seed 42
    python scripts/run_reference_model.py --input data/clinical_trials_55634.csv --min-doc-freq 10 --max-terms 134 --epochs 5 --batch-size 10000 --seed 42

Then launch the demo using the trained artifacts:

    streamlit run app/reference_app.py

Training outputs are saved locally in output/; evaluation metrics are saved in results/reference_metrics.csv and results/reference_experiment.json. Large downloaded data and trained model artifacts are intentionally not committed to GitHub.

## Dataset and reproduction caveat

Dataset source: https://huggingface.co/datasets/louisbrulenaudet/clinical-trials

The data source provides MeSH condition/intervention annotations. The original paper's pipeline uses a historical XML archive plus a MeSH trie to match MeSH terms in six clinical-trial text fields. The adapter here uses the pre-assigned annotations as a practical workaround. The algorithm and evaluation are reference-aligned, but the data representation is not identical, so results are **not directly comparable** with the upstream project's reported scores.

Use --rows 0 only if you intend to stream every eligible record. Downloading all rows may require significant time and disk space, and is not necessary for the recommended student-scale reproduction.

## Other prototype scripts

The earlier compact implementation remains under src/, scripts/run_experiment.py, and app/app.py. The reference-aligned path above is the recommended one for the assignment because it explicitly uses the upstream model and test classes.

## References

1. Zhou, X. *Collaborative Filtering on Keywords Recommendation for Clinical Trial Records*, Stanford CS229 Project, 2020: https://cs229.stanford.edu/proj2020spr/report/Zhou.pdf
2. Reference implementation: https://github.com/2005-Aneeshdutt/Clinical-Trial-Keyword-Recommendation
3. Dataset: https://huggingface.co/datasets/louisbrulenaudet/clinical-trials
4. ClinicalTrials.gov: https://clinicaltrials.gov/

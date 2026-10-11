# Collaborative Filtering on Keywords Recommendation for Clinical Trial Records

**UE24CS352A — Machine Learning Mini-Project | Problem 14 | Team 19**  
Shubharth Upadhyay (PES2UG24CS499) · Suraj Acharya (PES2UG24CS539)

Recommend additional medical keywords for a clinical-trial query when the original query is incomplete. This project implements a MeSH-based TF-IDF utility matrix, similarity-derived term neighbors, and a ReLU-regularized collaborative-filtering model inspired by Zhou's CS229 project.

> Academic information-retrieval prototype only; not a clinical decision-support system.

## Repository contents

- src/: MeSH data preparation, collaborative filtering model and evaluation metrics
- scripts/: experiment runner and streaming dataset downloader
- app/: Streamlit user interface
- data/sample_trials.csv: small offline pilot corpus for smoke tests
- results/: pilot-run metrics and query examples
- docs/DATASET.md: dataset download and compute guidance

## Dataset

The project supports the public ClinicalTrials.gov-derived dataset at https://huggingface.co/datasets/louisbrulenaudet/clinical-trials. The source dataset contains hundreds of thousands of studies and includes MeSH term metadata. Start with a reproducible sample and increase scale after verifying runtime and memory. The current implementation builds dense matrices and is **not optimized to train for many epochs across the entire 542k-study corpus**.

## Setup

    python -m venv .venv

Windows:

    .venv\Scripts\activate

macOS/Linux:

    source .venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

## Run the included pilot experiment

    python scripts/run_experiment.py

## Download a larger sample

    python scripts/download_hf_sample.py --rows 5000 --out data/clinical_trials_5000.csv
    python scripts/run_experiment.py --input data/clinical_trials_5000.csv --max-documents 5000 --min-doc-freq 5 --max-terms 300 --epochs 20 --batch-size 64

To stream every source record with non-empty MeSH metadata into a local CSV, use --rows 0. This may take a long time and consume substantial disk space. Downloading the full source corpus does not mean the current dense training code can practically train on all of it in one run. Keep the vocabulary capped and report only experiments that actually completed.

## Launch the demo

    streamlit run app/app.py

The demo defaults to the small bundled pilot dataset, so it can run without downloading the large dataset first.

## Method summary

1. Parse MeSH terms from each study record and filter rare terms.
2. Construct the log-scaled TF-IDF utility matrix.
3. Make an 80:10:10 train/dev/test split by document.
4. Derive a term-neighborhood matrix using cosine similarity.
5. Fit a ReLU prediction layer with L2 regularization and mini-batch gradient descent.
6. Mask positive test entries and compare top-3 recovery against popularity and random baselines.

The 15-record bundled dataset is only a smoke-test corpus. Its pilot ranking values are not a stable estimate of generalization. Do not present them as evidence that the method beats the baselines.

## References

1. Zhou, X. *Collaborative Filtering on Keywords Recommendation for Clinical Trial Records*, Stanford CS229 Project, 2020: https://cs229.stanford.edu/proj2020spr/report/Zhou.pdf
2. ClinicalTrials.gov: https://clinicaltrials.gov/
3. Hugging Face dataset: https://huggingface.co/datasets/louisbrulenaudet/clinical-trials

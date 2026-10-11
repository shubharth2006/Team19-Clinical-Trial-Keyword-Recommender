# Dataset acquisition and scale guidance

## Source and schema

Source: [louisbrulenaudet/clinical-trials on Hugging Face](https://huggingface.co/datasets/louisbrulenaudet/clinical-trials), derived from ClinicalTrials.gov. The published metadata reports 541,897 studies, with nested MeSH annotations under mesh_terms.conditions and mesh_terms.interventions. The Parquet dataset is large; the project downloads a shuffled stream of records with usable MeSH labels rather than fetching all shards into local storage.

**Licensing note:** the current Hugging Face dataset metadata declares Apache-2.0, while the dataset-card text says MIT. Keep source attribution and check the current upstream terms before redistributing any downloaded dataset. This project does not commit the downloaded corpus.

## Recommended workflow

The reference model caps the vocabulary and uses a dense document matrix during training. Therefore, start with a 5,000-record run, then try 20,000 or 55,634 records if the first run fits the available time and compute budget. The reference implementation's reported scale was 55,634 trial documents and 134 terms, so that is a useful target for an apples-to-apples scale check, but not an apples-to-apples data comparison because our adapter uses pre-assigned MeSH condition/intervention terms.

## Option A: run it through GitHub Actions

1. Open the private repository.
2. Select the **Actions** tab.
3. Choose **Run clinical-trial reproduction**.
4. Click **Run workflow** and choose 5,000 records first.
5. Wait for the job to finish, then download its artifact named clinical-trial-reproduction-5000.

The artifact contains reference_metrics.csv, reference_experiment.json, the saved model and vocabulary. If GitHub Actions is disabled for the repository, enable Actions under Settings > Actions > General first.

## Option B: run on Windows PowerShell

From the cloned repository:

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    python scripts/smoke_test_reference.py
    python scripts/download_hf_sample.py --rows 5000 --out data/clinical_trials_5000.csv --seed 42
    python scripts/run_reference_model.py --input data/clinical_trials_5000.csv --min-doc-freq 5 --max-terms 134 --epochs 5 --batch-size 10000 --seed 42

Launch the trained recommendation demo after training:

    streamlit run app/reference_app.py

To download every eligible record, use --rows 0. This is not recommended as the first run: streaming the entire split may take significant time and storage, and full-corpus model training is substantially more expensive than the 55,634-record target.

## Reproducibility and reporting

The experiment uses a fixed seed and writes results/reference_metrics.csv and results/reference_experiment.json. Only use metrics from a completed run. The included 15-record CSV is for parser/model smoke tests, not final performance claims.

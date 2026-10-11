# Dataset acquisition and scale guidance

## Source

The project references the public Hugging Face dataset
[louisbrulenaudet/clinical-trials](https://huggingface.co/datasets/louisbrulenaudet/clinical-trials),
derived from ClinicalTrials.gov study records and including mesh_terms.

## Recommended practical workflow

The whole source corpus can be used as the source of truth, but the current reference-model implementation builds dense NumPy arrays and trains the model in mini-batches. It is **not designed to train across all ~542k records for 100+ epochs in one run**. A full-corpus run can consume substantial RAM and time.

For a reproducible mini-project, start with 5,000 records; increase to 10,000 or 20,000 after the pipeline has been tested and runtime measured. Keep the vocabulary capped (for example, 200–500 frequent MeSH terms). Do not report a full-corpus experiment unless the run actually completed.

## Download a streaming sample

PowerShell:

    python -m pip install -U datasets pandas pyarrow
    python scripts/download_hf_sample.py --rows 5000 --out data/clinical_trials_5000.csv

The download script streams source rows and writes batches to CSV, rather than collecting all records in RAM.

To write every study with usable mesh_terms, use --rows 0. This can take a long time and produce a large local file. It only downloads the records; it does not imply that the dense model will fit or finish on the full corpus.

## Evaluate reproducibly

    python scripts/run_experiment.py --input data/clinical_trials_5000.csv --max-documents 5000 --min-doc-freq 5 --max-terms 300

The script records input size, effective filtered document count, vocabulary size, random seeds, ranking metrics and baselines under results/. Before final submission, use the actual generated results to update the write-up and presentation.

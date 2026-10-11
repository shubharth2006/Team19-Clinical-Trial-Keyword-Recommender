# Team 19 - Problem 14 Submission Checklist

## Project structure
- [x] Private GitHub repository
- [x] README with setup and run instructions
- [x] Reference paper and upstream implementation cited
- [x] MIT license notice retained for reused model/evaluation classes
- [x] Reference-aligned ReLU collaborative-filtering model and masked MSE/AP@K evaluators
- [x] Dataset adapter, experiment runner, inference CLI and Streamlit demo
- [x] Offline smoke-test script and GitHub Actions workflows
- [x] Two-page PDF and nine-slide presentation prepared with adaptation disclosure

## Required before final submission
- [ ] Run .github/workflows/run-reproduction.yml with 5,000 records first (or run locally)
- [ ] Confirm the workflow completes and download its generated artifact
- [ ] Inspect recommendations and compare AP@3/MSE/MSER with random and popularity baselines
- [ ] Update the report and presentation with measured metrics from the completed reference-aligned run
- [ ] Ensure both team members can explain the model, adapter, metrics and limitations
- [ ] Add faculty and assigned TAs to the private repository as instructed
- [ ] Submit the two-page PDF and slide deck through the course's submission channel

## Main commands

    pip install -r requirements.txt
    python scripts/smoke_test_reference.py
    python scripts/download_hf_sample.py --rows 5000 --out data/clinical_trials_5000.csv --seed 42
    python scripts/run_reference_model.py --input data/clinical_trials_5000.csv --min-doc-freq 5 --max-terms 134 --epochs 5 --batch-size 10000 --seed 42
    streamlit run app/reference_app.py

## Honest reporting note

The larger real-data run has not yet been executed in this prepared deliverable. Do not state final performance figures until results/reference_metrics.csv and results/reference_experiment.json are generated from a completed run. The MeSH annotation workaround must also be disclosed: it uses pre-assigned condition/intervention labels rather than reproducing the original text-to-MeSH-trie matching stage exactly.

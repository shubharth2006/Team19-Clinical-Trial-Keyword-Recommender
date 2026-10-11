# Team 19 - Problem 14 Submission Checklist

## Project structure
- [x] Private GitHub repository
- [x] README with setup and run instructions
- [x] Reference paper and upstream implementation cited
- [x] MIT license notice retained for reused model/evaluation classes
- [x] Reference-aligned ReLU collaborative-filtering model and masked MSE/AP@K evaluators
- [x] Dataset adapter, experiment runner, inference CLI and Streamlit demo
- [x] Offline smoke-test and end-to-end CLI test in GitHub Actions
- [x] 5,000-record real-data experiment completed; measured metrics recorded
- [x] Two-page PDF and nine-slide presentation updated with measured results and adaptation disclosure

## Remaining before/at hand-in
- [ ] Run the updated data workflow again to confirm the recommendation CLI and Streamlit health-check steps complete
- [ ] Inspect the saved top-three recommendations for topical relevance
- [ ] Consider an additional seed or the 55,634-record run if time permits; do not imply this is already complete
- [ ] Ensure both team members can explain the model equation, data adapter, metrics and limitations
- [ ] Add faculty and assigned TAs to the private repository
- [ ] Submit the PDF and presentation through the course submission channel

## Main commands

    pip install -r requirements.txt
    python scripts/smoke_test_reference.py
    python scripts/download_hf_sample.py --rows 5000 --out data/clinical_trials_5000.csv --seed 42
    python scripts/run_reference_model.py --input data/clinical_trials_5000.csv --min-doc-freq 5 --max-terms 134 --epochs 5 --batch-size 10000 --seed 42
    python scripts/reference_recommend.py --query "Neoplasms" --top-k 3
    streamlit run app/reference_app.py

## Measured experiment

See results/REFERENCE_RUN_5000.md. The completed 5,000-record run used 4,000/500/500 train/validation/test records and a 134-term MeSH vocabulary. Test AP@3 was 0.02539 for collaborative filtering, 0.00993 for popularity, and 0.00110 for random selection. This is one seeded result with an adapted MeSH-label schema, not a directly comparable reproduction of the upstream paper's dataset.

## Honest reporting note

The current adapter uses pre-assigned MeSH condition/intervention labels rather than reproducing the original XML-to-MeSH-trie matching across all six text attributes. Keep that limitation visible in the write-up and presentation.

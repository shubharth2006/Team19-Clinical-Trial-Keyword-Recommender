# Collaborative Filtering on Keywords Recommendation for Clinical Trial Records

**UE24CS352A - Machine Learning Mini-Project | Problem 14 | Team 19**  
Shubharth Upadhyay (PES2UG24CS499) and Suraj Acharya (PES2UG24CS539)

This is a reference-aligned reproduction/adaptation of the assigned task from Zhou's Stanford CS229 project report. It is not presented as a new algorithm.

> Educational information-retrieval prototype only; not a clinical decision-support system.

## Primary implementation

The main experiment is scripts/run_reference_model.py. It uses the attributed reference model/evaluation modules under ml/.

- MeSH term-by-trial log-TF-IDF utility matrix
- Cosine-similarity-derived term-neighbor weights
- ReLU prediction with a trainable parameter matrix and bias
- Mini-batch gradient descent and L2 regularization
- Masked-term evaluation using Top-3 precision (named AP@K by the reused class), MSE and MSER
- Random baseline from the reference evaluator, plus an added popularity baseline
- Saved model/vocabulary artifacts for CLI inference and the Streamlit demo

Core model/evaluation classes are reused under the MIT license. The original license and attribution are kept in LICENSES/Xiao-Zhou-MIT.txt and THIRD_PARTY_NOTICES.md. The project-specific integration includes the streaming data adapter, experiment runner, popularity baseline, saved outputs, CLI, Streamlit UI and automation. See docs/REFERENCE_REPRODUCTION.md.

## Quick start: use the saved 55,634-record model

1. Create and activate a virtual environment.
2. Install dependencies using pip install -r requirements.txt.
3. Download the run artifact from [the successful reference-scale GitHub Actions run](https://github.com/shubharth2006/Team19-Clinical-Trial-Keyword-Recommender/actions/runs/38135025261).
4. Extract Team19_Clinical_Trial_CF_55634_Run_Artifacts.zip into the repository root, preserving paths such that output/ and results/ are created.
5. Run CLI inference and launch the app:

       python scripts/reference_recommend.py --query "Neoplasms" --top-k 3
       streamlit run app/reference_app.py

Open the local Streamlit URL, typically http://localhost:8501.

## Reproduce the complete experiment

Recommended Windows PowerShell sequence:

    py -3.11 -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    python scripts/smoke_test_reference.py
    python scripts/download_hf_sample.py --rows 55634 --out data/clinical_trials_55634.csv --seed 42
    python scripts/run_reference_model.py --input data/clinical_trials_55634.csv --min-doc-freq 5 --max-terms 134 --epochs 5 --batch-size 10000 --seed 42
    python scripts/reference_recommend.py --query "Neoplasms" --top-k 3
    streamlit run app/reference_app.py

If PowerShell blocks activation, run Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass in the current terminal, then activate again. The dataset download requires internet access.

## Measured reference-scale run

A 55,634-record run completed successfully in GitHub Actions, including download, training, evaluation, CLI inference, Streamlit startup, and artifact upload.

- Split: 44,507 train / 5,563 validation / 5,564 test
- Vocabulary: 134 terms; seed: 42; epochs: 5; masked terms: 80%
- Test Top-3 precision: collaborative filtering 0.03033, popularity 0.01165, random 0.00532
- Test MSE/MSER: 0.00473
- Validation Top-3 precision: 0.02936; random 0.00466

Full measured run notes: results/REFERENCE_RUN_55634.md  
Metrics CSV: results/reference_metrics_55634.csv  
Experiment JSON: results/reference_experiment_55634.json  
Successful workflow: https://github.com/shubharth2006/Team19-Clinical-Trial-Keyword-Recommender/actions/runs/38135025261

These results are from one seed and the current adapted dataset; they are not directly comparable with the original project's reported scores. Absolute Top-3 precision remains low.

## Dataset adaptation and known limitations

Source dataset: https://huggingface.co/datasets/louisbrulenaudet/clinical-trials

The current adapter uses pre-assigned MeSH annotations rather than recreating the original historical XML plus MeSH-trie matching across six free-text fields. The utility matrix vocabulary/IDF is computed before the train/validation/test split; a stricter future evaluation should fit vocabulary/IDF on training records only. Query matching relies on exact controlled-vocabulary phrases, so the interface is not a general semantic search model.

The app demonstrates the pipeline, but qualitative relevance should be inspected across several queries before claiming strong recommendation quality.

## Team handoff and TA preparation

- Start with docs/PARTNER_HANDOFF_AND_TA_DEMO_GUIDE.md for Windows run instructions, file map, equations, demonstration plan, and viva questions.
- Read docs/VIVA_PREP.md and docs/REFERENCE_REPRODUCTION.md.
- Keep the private repository as the source of truth; add faculty and assigned TAs under GitHub repository access.
- Be explicit about reuse and each teammate's real contribution.

## References

1. Zhou, X. *Collaborative Filtering on Keywords Recommendation for Clinical Trial Records*, Stanford CS229 Project, 2020: https://cs229.stanford.edu/proj2020spr/report/Zhou.pdf
2. MIT-licensed reference implementation: https://github.com/2005-Aneeshdutt/Clinical-Trial-Keyword-Recommendation
3. Dataset: https://huggingface.co/datasets/louisbrulenaudet/clinical-trials
4. ClinicalTrials.gov: https://clinicaltrials.gov/

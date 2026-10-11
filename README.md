# Collaborative Filtering on Keywords Recommendation for Clinical Trial Records

**UE24CS352A — Machine Learning Mini-Project | Problem 14 | Team 19**  
Shubharth Upadhyay (PES2UG24CS499) · Suraj Acharya (PES2UG24CS539)

Recommend additional medical keywords for a clinical-trial query when the original query is incomplete. This project implements a MeSH-based TF-IDF utility matrix, similarity-derived term neighbors, and a ReLU-regularized collaborative-filtering model inspired by Zhou's CS229 project.

> Academic information-retrieval prototype only; not a clinical decision-support system.

## Repository status
The initial project upload is in progress. See the submission ZIP for the full local package until all files are committed here.

## Dataset
The project supports the public ClinicalTrials.gov-derived dataset on Hugging Face: https://huggingface.co/datasets/louisbrulenaudet/clinical-trials. The dataset contains hundreds of thousands of study records. Recommended practice is to keep the full source corpus but run a reproducible subset first, with streaming/chunked processing and a capped keyword vocabulary to control RAM.

## Method summary
1. Parse MeSH terms and build a filtered vocabulary.
2. Construct a log-scaled TF-IDF utility matrix.
3. Split records into 80% train, 10% validation, 10% test.
4. Derive a term-neighbor matrix from cosine similarities.
5. Predict masked terms with a ReLU collaborative-filtering model and L2 regularization.
6. Evaluate masked keyword recovery using APTK@3 and compare with popularity and random baselines.

## Local setup
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_experiment.py
streamlit run app/app.py
```

See the report and slides for full methodology and the current pilot results. The bundled 15-record pilot is a smoke test, not sufficient for stable generalization claims. Run a larger data experiment before final submission and update report numbers only from actual measured outputs.

## References
- Zhou, X. *Collaborative Filtering on Keywords Recommendation for Clinical Trial Records*, Stanford CS229 Project, 2020: https://cs229.stanford.edu/proj2020spr/report/Zhou.pdf
- ClinicalTrials.gov: https://clinicaltrials.gov/
- Hugging Face dataset: https://huggingface.co/datasets/louisbrulenaudet/clinical-trials

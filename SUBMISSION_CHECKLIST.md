# Team 19 — Problem 14 Submission Checklist

## Project source repository

- [x] Private GitHub repository created
- [x] README, requirements and run instructions committed
- [x] Collaborative-filtering model source committed
- [x] Dataset parsing and evaluation source committed
- [x] Streamlit demo source committed
- [x] Streaming Hugging Face dataset downloader committed
- [x] Small offline pilot dataset and pilot metrics committed

## Before final submission

- [ ] Download at least 5,000 MeSH-labelled studies on a machine with internet access
- [ ] Run the experiment and preserve the generated metrics, plots and console output
- [ ] Inspect recommendations manually for relevance and explain limitations
- [ ] Update the PDF report and presentation using only measured results from the final code/data run
- [ ] Add faculty and TAs to this private repository as instructed
- [ ] Confirm both team members can explain the code, equations, data flow and evaluation
- [ ] Submit the two-page PDF, slide deck and any LMS-required files separately

## Reproducible commands

    pip install -r requirements.txt
    python scripts/run_experiment.py
    python scripts/download_hf_sample.py --rows 5000 --out data/clinical_trials_5000.csv
    python scripts/run_experiment.py --input data/clinical_trials_5000.csv --max-documents 5000 --min-doc-freq 5 --max-terms 300 --epochs 20 --batch-size 64
    streamlit run app/app.py

## Important caveat

The bundled 15-record corpus is an offline smoke-test corpus. It is too small to support claims about generalization or model superiority. Report performance from the larger evaluation only after that run actually completes.

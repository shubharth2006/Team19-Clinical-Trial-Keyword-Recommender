# Third-party code and data notices

## Collaborative-filtering reference implementation

This project directly reuses core model and evaluation modules from:

- Repository: https://github.com/2005-Aneeshdutt/Clinical-Trial-Keyword-Recommendation
- Original author/copyright notice in the upstream license: Copyright (c) 2020 Xiao Zhou
- License: MIT; a copy is provided at LICENSES/Xiao-Zhou-MIT.txt
- Research reference: Xiao Zhou, “Collaborative Filtering on Keywords Recommendation for Clinical Trial Records,” Stanford CS229, 2020: https://cs229.stanford.edu/proj2020spr/report/Zhou.pdf

The copied core modules include the latent-factor model and its masked-term MSE/AP@K evaluation. New adapter code was added for the Hugging Face dataset's pre-assigned MeSH annotation fields, deterministic experiment setup, outputs, and usability. The adapter is a workaround for obtaining a reproducible dataset without manually downloading the original XML archive.

## Dataset

The optional dataset downloader uses https://huggingface.co/datasets/louisbrulenaudet/clinical-trials, based on ClinicalTrials.gov. Dataset metadata currently labels the repository Apache-2.0, while its README text contains a different license statement; consult the source repository metadata and terms before redistributing any downloaded dataset. This project does not commit downloaded source records.

# Viva preparation: Problem 14

## One-sentence explanation

The project recommends additional controlled medical subject headings (MeSH terms) for a clinical-trial search by learning co-occurrence structure between medical terms across trial records.

## Pipeline

1. Read clinical-trial records and parse their MeSH terms.
2. Count term occurrences by study and filter rare terms to bound vocabulary size.
3. Build a log-scaled TF-IDF utility matrix with terms as rows and trials as columns.
4. Split the data into train, validation and test records.
5. Calculate cosine similarity between term rows and form a term-neighbor matrix W.
6. Fit a ReLU prediction layer with L2 regularization using mini-batch gradient descent.
7. Hide a portion of known terms in held-out studies and rank candidate terms.
8. Compare Top-3 recovery against popularity and random baselines.

## Likely questions

### Why MeSH?

MeSH is a controlled biomedical vocabulary. It reduces variation caused by using different free-text phrases for the same concept.

### Why TF-IDF?

The term-document representation gives more weight to terms that distinguish a study and less weight to terms that appear in many studies.

### Why collaborative filtering?

The model learns co-occurrence relationships across studies, so a term can receive a high score because it is related to the terms already present in the query.

### Why mask terms?

For evaluation, masking simulates an incomplete keyword set. The model is tested on whether it ranks the hidden known terms highly.

### Why use baselines?

A popularity baseline tests whether the model does better than simply recommending frequent terms. A random baseline gives a simple reference point.

### What is the main limitation?

The current implementation builds dense NumPy matrices and uses a small controlled vocabulary. The complete source corpus has hundreds of thousands of trials, but training on all of them would require a more scalable sparse/mini-batch design and careful runtime measurement. The 15-record pilot is a smoke test, not proof of strong predictive performance.

### What would improve it next?

Run a larger real dataset experiment; tune vocabulary size and regularization on validation data; compare TF-IDF nearest-neighbor and collaborative-filtering baselines; and preserve train/test separation when computing vocabulary statistics to avoid information leakage.

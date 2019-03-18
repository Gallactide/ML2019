# Machine Learning Project - 2019
## Required Packages: 
python3, sklearn, pandas, spacy (optional, allows for stopword removal)

## Setup
  Run `sh ./setup.sh` to download/parse dataset
  
## Actual Code
Run `python -i code/main.py dataset/parsed.csv` after setup. This runs the currently set vectorizer/classifier.

The repl then has `mails` (dataframe of csv, slightly processed further), `features` (result of vectorizer), `classifier` set once completed.

# ASA-CD linguistic biomarker analysis

Code for the empirical study in *Applied Sociolinguistic AI for Community Development (ASA-CD): a framework for linguistically grounded community intervention, with evidence from two public corpora* (Alam & Ferzana).

## Data
Download both public datasets into `data/`:
- Corpus A – YouTube comments dataset: https://www.kaggle.com/datasets/atifaliak/youtube-comments-dataset (file `YoutubeCommentsDataSet.csv`, columns `Comment`, `Sentiment`)
- Corpus B – Twitter US Airline Sentiment: https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment (file `Tweets.csv`; keep columns `text`, `airline_sentiment` and rename the latter to `sentiment`, or edit the column names in the scripts)

## Run
```
pip install -r requirements.txt
python biomarkers.py data/YoutubeCommentsDataSet.csv data/Tweets.csv results/
python robustness.py data/YoutubeCommentsDataSet.csv data/Tweets.csv results/
```
`biomarkers.py` produces prevalence tables, chi-square / Kruskal–Wallis tests, the main logistic regression (`results/results.json`) and Figure 1. `robustness.py` runs checks R1–R6 (`results/robustness.json`) and writes `referent_coding_sheet.csv`.

## Referent coding
`referent_coding_sheet.csv` contains 200 randomly sampled texts (100 per corpus) that contain *they/them/their*. The column `referent_coder1` gives the referent of the first exclusive pronoun, coded with AI assistance and reviewed by the first author, using these categories:

| Code | Meaning |
|---|---|
| `other_people_group` | people or social groups (e.g. kids, teachers, parents) |
| `institution_company_staff` | organisations, companies, governments, or their staff |
| `specific_individuals` | named or identifiable individuals (e.g. two presenters, a politician) |
| `object_non_human` | products, animals, games, bags, documents |
| `generic_impersonal` | generic or impersonal uses ("they say", "someone ... their") |
| `unclear` | referent cannot be determined |

`referent_coder2` is left blank for an independent second coder; agreement can then be reported as Cohen's kappa.

## Word lists
- Exclusive pronouns: they, them, their, theirs, themselves
- Inclusive pronouns: we, us, our, ours, ourselves
- Absolutist words (Al-Mosaiwi & Johnstone, 2018): absolutely, all, always, complete, completely, constant, constantly, definitely, entire, ever, every, everyone, everything, full, must, never, nothing, totally, whole

## Licence
MIT. The datasets are subject to their own licences on Kaggle.

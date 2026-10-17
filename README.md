# CineMatch – Hybrid Content-Based Movie Recommender (TMDB 5000)

Master's project in Data Science. Extends a TF-IDF/cosine baseline into a hybrid recommender with offline evaluation and a web demo.

## Structure
- `src/pipeline.py` – cleaning, features, 5 models, evaluation, export
- `web/template.html` → `web/index.html` – self-contained demo (data embedded)
- `results/metrics.json` – evaluation output
- `data/` – put `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` here (Kaggle: TMDB 5000)

## Run
    pip install -r requirements.txt
    python src/pipeline.py data/      # regenerates results/ and web/index.html
    open web/index.html               # no server needed

## Improvements over the original notebook
| Original | This project |
|---|---|
| Merge on `title` (duplicate rows) | Merge on movie ID |
| Names split into single words | Each person/genre/keyword = one token |
| Whole crew list as text | Director + top-5 cast only (less noise) |
| One TF-IDF over everything | Separate text and metadata spaces, weighted hybrid (α) |
| No ranking prior | Bayesian weighted-rating re-rank |
| No evaluation | 4 proxy metrics, 5 models, α sweep |

## Evaluation note
TMDB has no user ratings, so metrics are proxies (genre Jaccard, same-director rate, coverage, mean rating). Genre and director are also model inputs, so these favour metadata models; discuss this limitation in the thesis and, as future work, validate with MovieLens ratings or a user study.

## Future work
Sentence-embedding text features (SBERT), collaborative filtering on MovieLens, MMR diversity re-ranking, user study.

"""Movie recommender pipeline (TMDB 5000). Usage: python src/pipeline.py data/
Outputs: results/metrics.json, web/index.html (self-contained demo)."""
import ast, json, re, sys, os
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

D = sys.argv[1] if len(sys.argv) > 1 else 'data'
K = 10

# ---------- 1. Load & clean (merge on ID, not title: avoids duplicate rows) ----------
m = pd.read_csv(f'{D}/tmdb_5000_movies.csv')
c = pd.read_csv(f'{D}/tmdb_5000_credits.csv')[['movie_id', 'cast', 'crew']]
df = m.merge(c, left_on='id', right_on='movie_id').dropna(subset=['title']).reset_index(drop=True)
for col in ['overview', 'tagline']:
    df[col] = df[col].fillna('')
df['year'] = pd.to_datetime(df.release_date, errors='coerce').dt.year

L = lambda x: ast.literal_eval(x) if isinstance(x, str) else []
df['genre_l'] = df.genres.map(lambda x: [i['name'] for i in L(x)])
df['kw_l'] = df.keywords.map(lambda x: [i['name'] for i in L(x)])
df['cast_l'] = df.cast.map(lambda x: [i['name'] for i in L(x)][:5])
df['dir'] = df.crew.map(lambda x: next((i['name'] for i in L(x) if i['job'] == 'Director'), ''))
df['crew_all'] = df.crew.map(lambda x: ' '.join(i['name'] for i in L(x)))

# Bayesian weighted rating (IMDB formula): damps movies with few votes
C0, vm = df.vote_average[df.vote_count > 0].mean(), df.vote_count.quantile(0.80)
df['wr'] = df.vote_count / (df.vote_count + vm) * df.vote_average + vm / (df.vote_count + vm) * C0
wr_pct = df.wr.rank(pct=True).values

# ---------- 2. Features ----------
tok = lambda xs: [re.sub(r'\W', '', x.lower()) for x in xs]  # "Sam Worthington" -> one token
meta = (df.genre_l.map(lambda g: tok(g) * 2) + df.kw_l.map(tok) + df.cast_l.map(tok)
        + df['dir'].map(lambda d: tok([d]) * 2 if d else [])).map(' '.join)
text = df.overview + ' ' + df.tagline

baseline_txt = (df.overview + ' ' + df.genre_l.map(' '.join) + ' ' + df.tagline + ' ' + df.kw_l.map(' '.join)
                + ' ' + df.cast_l.map(' '.join) + ' ' + df.crew_all).str.lower().str.replace(r'[^a-z\s]', '', regex=True)
Xb = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2)).fit_transform(baseline_txt)
Xt = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(text)
Xm = TfidfVectorizer(token_pattern=r'\S+', min_df=2, sublinear_tf=True).fit_transform(meta)
St, Sm = linear_kernel(Xt).astype('f4'), linear_kernel(Xm).astype('f4')
Sb = linear_kernel(Xb).astype('f4')

# ---------- 3. Models ----------
def topk(S, k=K, rerank=0.0):
    S = S.copy(); np.fill_diagonal(S, -1)
    if rerank:  # blend similarity with quality prior
        S = S * (1 - rerank + rerank * wr_pct[None, :])
    i = np.argpartition(-S, k, axis=1)[:, :k]
    o = np.take_along_axis(S, i, 1).argsort(1)[:, ::-1]
    i = np.take_along_axis(i, o, 1)
    return i, np.take_along_axis(S, i, 1)

A = 0.4  # text weight in hybrid (chosen from the alpha sweep below)
Sh = A * St + (1 - A) * Sm
models = {'Baseline (TF-IDF on all text)': topk(Sb), 'Text only (overview+tagline)': topk(St),
          'Metadata only': topk(Sm), 'Hybrid': topk(Sh), 'Hybrid + quality re-rank': topk(Sh, rerank=0.30)}

# ---------- 4. Offline evaluation (no ratings in TMDB -> proxy metrics) ----------
gs = df.genre_l.map(set).values
def evaluate(nb):
    jac = np.mean([np.mean([len(gs[q] & gs[j]) / max(len(gs[q] | gs[j]), 1) for j in r]) for q, r in enumerate(nb)])
    has = (df['dir'] != '').values
    sd = np.mean([np.mean([df['dir'][j] == df['dir'][q] for j in r]) for q, r in enumerate(nb) if has[q]])
    return dict(genre_jaccard=round(jac, 4), same_director=round(sd, 4),
                coverage=round(len(set(nb.ravel())) / len(df), 4),
                mean_rating=round(float(df.vote_average.values[nb].mean()), 3))
metrics = {k: evaluate(v[0]) for k, v in models.items()}
sweep = [dict(alpha=a, **evaluate(topk(a * St + (1 - a) * Sm)[0])) for a in [0, .2, .4, .6, .8, 1]]

# ---------- 5. EDA summary + web export ----------
g = df.genre_l.explode().value_counts().head(12)
dec = (df.year // 10 * 10).dropna().astype(int).value_counts().sort_index()
dec = dec[dec.index >= 1950]
eda = dict(genres=[[k, int(v)] for k, v in g.items()], decades=[[int(k), int(v)] for k, v in dec.items()],
           n=len(df), n_kw=int(len(set(sum(df.kw_l, [])))), n_people=int(len(set(sum(df.cast_l, [])))))
ni, ns = models['Hybrid + quality re-rank']
movies = []
for q, r in df.iterrows():
    movies.append([r.title, None if pd.isna(r.year) else int(r.year), r.genre_l, round(r.vote_average, 1), int(r.vote_count),
                   r.overview[:230], r['dir'], r.cast_l[:3], r.kw_l[:8], [[int(j), round(float(s), 3)] for j, s in zip(ni[q], ns[q])]])
os.makedirs('results', exist_ok=True); os.makedirs('web', exist_ok=True)
out = dict(metrics=metrics, sweep=sweep, eda=eda, alpha=A)
json.dump(out, open('results/metrics.json', 'w'), indent=1)
payload = json.dumps(dict(movies=movies, **out), separators=(',', ':'), ensure_ascii=False)
tpl = open('web/template.html', encoding='utf8').read()
open('web/index.html', 'w', encoding='utf8').write(tpl.replace('/*DATA*/null', payload))
print(json.dumps(metrics, indent=1)); print(sweep); print(f'web/index.html {os.path.getsize("web/index.html")/1e6:.1f} MB')

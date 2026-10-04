# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A Flask web app that clusters songs (scraped from chosic.com) by audio features using K-Means, then recommends similar tracks via K-nearest-neighbors. Users can either enter feature sliders manually or paste a Spotify track ID to auto-scrape its features, then get a cluster prediction plus 3D PCA and Camelot-wheel visualizations.

This is a course final project, published on GitHub as `EricC430/music-cluster-recommender` (public, portfolio). `music.db` and `models/*.pkl` are **gitignored** — a fresh clone has no data or model and must scrape + train first (see `README.md`). `_scratch/` is a gitignored local folder holding legacy prototypes (`scraper_test*.py`, unused `db.py`) and third-party HTML snapshots; don't commit it or build on it.

## Running the app

```bash
python app.py
```

Starts the Flask dev server on port 5000 with `debug=True`. Requires the pipeline model at `models/music_clustering_pipeline.pkl` to already exist (see "Model training pipeline" below) — `app.py` raises `FileNotFoundError` on startup if it's missing, via `MusicRecommender.load_models_and_data()`.

## Dependencies

`requirements.txt` lists unpinned package names (no lockfile):

```bash
pip install -r requirements.txt
```

Chrome must be installed locally — `webdriver_manager.chrome.ChromeDriverManager` downloads a matching ChromeDriver automatically for every Selenium-based script.

## Architecture

### Data flow: scrape → store → train → serve

1. **Scraping** (`scraper.py`) — Selenium + BeautifulSoup scrapers targeting `chosic.com`:
   - `Chosic_Topics_Scraper` — collects topic list URLs from chosic's "lists of songs" page.
   - `Chosic_Songlist_Scraper` — collects Spotify track IDs from a topic page, filtering out IDs already in `music.db` (`get_existing_ids()`).
   - `Chosic_Song_Analysis_Scraper` — for one Spotify ID, loads chosic's music-genre-finder page and returns the raw `#result-p` + `#result-analyzer` HTML.
   - `parse_html_data(html, song_id)` — parses that HTML into a flat dict of song fields (title, artist, album, release_date, bpm, key/camelot, 0–100 scaled audio features, genres, YouTube/Spotify URLs).
   - `run_scraper()` (the `__main__` entry point) chains all three scrapers and calls `db_manager.save_song()` per track; `START_TOPIC_INDEX` (top of file, default 0) skips the first N topics when resuming.
   - Scrape logs are written to `logs/log_<ClassName>_<timestamp>.json` only when `LOG_MODE = 1` at the top of the file (currently `0`, off by default).

2. **Storage** (`db_manager.py`, schema owner) — SQLite database `music.db`, single table `songs`. Key points:
   - `id` is an autoincrement PK; `id_spotify` is a separate `UNIQUE` text column used for de-duplication.
   - `save_song()` uses `INSERT OR IGNORE` keyed on `id_spotify`, so re-running the scraper is idempotent.
   - `cluster` starts `NULL` and is only populated by the training pipeline (`train_kmeans.py`).
   - `app.py` and `recommender.py` open raw `sqlite3.connect()` calls; there is no shared connection helper (the old unused Flask-`g` `db.py` was moved to `_scratch/`).

3. **Data repair** (`fix_data.py`) — `DataFixer` runs a list of `fix_strategies`, each an (SQL `condition`, `handler`) pair, to patch bad rows: normalize `release_date`, backfill empty `genres_wikipedia`, strip `"from ... album:"` prefixes from `album`, or fully re-scrape rows with `duration = 0`/missing fields. Local string fixes run first; re-scraping (via `Chosic_Song_Analysis_Scraper`) is the fallback and only spins up a Selenium driver lazily (`init_driver()`) if actually needed.

4. **Clustering** (`tune_kmeans.py`, `train_kmeans.py`):
   - `tune_kmeans.py` is exploratory only — compares `StandardScaler` vs `PowerTransformer` across K=2..20 and plots elbow/silhouette curves (`plt.show()`, not saved to disk). Use it to pick `n_clusters` before training.
   - `train_kmeans.py`'s `MusicClusteringModel` builds a scikit-learn `Pipeline`: `ColumnTransformer` (`StandardScaler` on the 9 numeric features, `OneHotEncoder(handle_unknown='ignore')` on `key`/`camelot`) → `KMeans`. It filters `bpm <= 0` rows before fitting. Running the file's `__main__` (`n_clusters=16`, overriding the class default of 5) does the full cycle: load → build pipeline → train → save `models/music_clustering_pipeline.pkl` → write `cluster` back to every trained row in `music.db` → print per-cluster stats.
   - **The saved pipeline is the single source of truth for feature scaling/encoding** — `recommender.py` always transforms new input through `pipeline.named_steps['preprocessor']` rather than re-deriving scalers, so training and inference stay consistent.

5. **Cluster interpretation** (`interpret_model.py`, optional) — fits a shallow `DecisionTreeClassifier(criterion='entropy')` as a surrogate model to explain the KMeans clusters in human-readable rules, prints feature importances (~Information Gain), and writes `models/cluster_tree_plot.png` + `models/cluster_tree.dot`. `app.py` copies the PNG into `static/` at startup if present.

6. **Serving** (`app.py` + `recommender.py`):
   - `MusicRecommender` (in `recommender.py`) loads the trained pipeline and all `cluster IS NOT NULL` rows from `music.db` once at Flask startup, and builds a global `NearestNeighbors` index (`_build_global_knn`) over every song's transformed feature vector — recommendations always search the *whole* database, not just the predicted cluster.
   - **Feature scale gotcha**: `music.db` stores the seven percentage-style features (`energy`, `danceability`, `happiness`, `acousticness`, `instrumentalness`, `liveness`, `speechiness`) on a 0–100 scale, but the web form sliders submit 0.0–1.0. `predict_new_song()` and `get_pca_3d_plot()` both multiply those fields by 100 before running them through the pipeline — keep this conversion in sync if you touch either method or the frontend sliders.
   - Route `GET /` renders `templates/index.html` (the input form).
   - Route `POST /api/import_spotify` — given `{"spotify_id": ...}`, launches a headless Chrome, runs `Chosic_Song_Analysis_Scraper` + `parse_html_data` live, and returns the parsed features as JSON for the frontend to pre-fill the form (does **not** write to the database).
   - Route `POST /recommend` — reads form fields (including `camelot_code`, mapped server-side to a display `key` string via the `CAMELOT_TO_KEY` dict), calls `recommender.predict_new_song()`, then `get_pca_3d_plot()` and `get_polar_plot()` for the two Plotly visualizations, and renders `templates/result.html`.
   - Both `/api/import_spotify` and `/recommend` spin up their own Selenium/Chrome instances per request — there's no shared driver pool.

### Frontend

`templates/index.html` and `templates/result.html` are server-rendered Jinja templates using Bootstrap 5 + Font Awesome from CDN. Plotly figures are serialized server-side with `plotly.utils.PlotlyJSONEncoder` and rendered client-side from the passed-in JSON (`plot_json`, `polar_json`).

### Other files

- `logs/` (gitignored) holds JSON scrape logs from `Scraper.log_data()` when `LOG_MODE = 1`.
- `models/cluster_tree_plot.png` is the only model artifact committed; `app.py` copies it into `static/` (gitignored) at startup.

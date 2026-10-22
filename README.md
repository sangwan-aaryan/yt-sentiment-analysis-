yt-sentiment-analysis
==============================

a chrome plugin which help to know about the youtube comments

## Run the prediction API

Install the project dependencies and ensure the configured MLflow server has a
`staging` alias for `yt_chrome_plugin_model`. The model's MLflow run must also
contain the TF-IDF vectorizer artifact logged by model evaluation.

```powershell
python -m pip install -r requirements.txt
$env:MLFLOW_TRACKING_URI = "http://16.178.121.158:5000"
$env:MLFLOW_MODEL_ALIAS = "staging"
python backend/main.py
```

On first extension use, enter your YouTube Data API key in the popup. It is
stored in Chrome's local extension storage rather than embedded in source.
Comment retrieval follows YouTube's pagination tokens until no more top-level
comment threads are available; analysis is sent to the API in batches. The
YouTube API quota and any comments hidden or unavailable to that API still
limit what can be fetched.

Send comments as a JSON list:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/predict `
  -ContentType "application/json" `
  -Body '{"comments":["This video is great","I did not like it"]}'
```

Sentiment values are the model's class labels (`-1`, `0`, or `1`). The API
resolves the alias to one model version and loads that version's matching
vectorizer from the same MLflow run.

The `/generate_chart`, `/generate_wordcloud`, and `/generate_trend_graph`
endpoints return PNG images for the Chrome extension to display directly.
The extension displays the number actually analyzed and does not stop at 500.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

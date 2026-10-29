import logging
import math
import os
import re
from io import BytesIO
from collections import Counter
from functools import lru_cache

os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "false")

import joblib
import mlflow
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import pandas as pd
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from mlflow.tracking import MlflowClient
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud


app = Flask(__name__)
CORS(app)
logger = logging.getLogger(__name__)

MODEL_NAME = "yt_chrome_plugin_model"
MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "staging")
TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", "http://16.178.121.158:5000"
)
SENTIMENTS = {
    -1: {"label": "Negative", "color": "#fb7185"},
    0: {"label": "Neutral", "color": "#fbbf24"},
    1: {"label": "Positive", "color": "#34d399"},
}

model = None
vectorizer = None


@app.route("/")
def home():
    return "Welcome to the flask api"


@lru_cache(maxsize=1)
def get_stop_words():
    return set(stopwords.words("english")) - {
        "not",
        "but",
        "however",
        "no",
        "yet",
    }


def preprocess_comment(comment):
    """Apply the same text cleaning used when preparing the training data."""
    comment = comment.lower().strip()
    comment = re.sub(r"\n", " ", comment)
    comment = re.sub(r"[^A-Za-z0-9\s!?.,]", "", comment)
    lemmatizer = WordNetLemmatizer()
    return " ".join(
        lemmatizer.lemmatize(word)
        for word in comment.split()
        if word not in get_stop_words()
    )


def load_model_and_vectorizer(model_name=MODEL_NAME, model_alias=MODEL_ALIAS):
    """Load one registry version and the vectorizer logged by its source run."""
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    registered_version = client.get_model_version_by_alias(
        model_name, model_alias
    )
    if not registered_version.run_id:
        raise RuntimeError(
            f"Registry alias {model_name}@{model_alias} has no source run"
        )

    model_uri = (
        f"models:/{model_name}/{registered_version.version}"
    )
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    vectorizer_uri = (
        f"runs:/{registered_version.run_id}/"
        "preprocessing/tfidf_vectorizer.pkl"
    )
    vectorizer_path = mlflow.artifacts.download_artifacts(vectorizer_uri)
    loaded_vectorizer = joblib.load(vectorizer_path)
    return loaded_model, loaded_vectorizer


def get_model_and_vectorizer():
    global model, vectorizer
    if model is None or vectorizer is None:
        model, vectorizer = load_model_and_vectorizer()
    return model, vectorizer


def get_json_object():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return None, (jsonify({"error": "Request body must be a JSON object"}), 400)
    return data, None


def validate_comments(comments):
    if not isinstance(comments, list) or not comments:
        return "'comments' must be a non-empty list"
    if any(not isinstance(comment, str) for comment in comments):
        return "Every comment must be a string"
    return None


def predict_comments(comments):
    loaded_model, loaded_vectorizer = get_model_and_vectorizer()
    cleaned = [preprocess_comment(comment) for comment in comments]
    features = loaded_vectorizer.transform(cleaned)
    inputs = pd.DataFrame(
        features.toarray(),
        columns=loaded_vectorizer.get_feature_names_out(),
    )
    predictions = loaded_model.predict(inputs)
    if hasattr(predictions, "tolist"):
        predictions = predictions.tolist()
    return [str(prediction) for prediction in predictions]


def png_response(figure):
    image = BytesIO()
    FigureCanvasAgg(figure)
    figure.savefig(
        image,
        format="png",
        dpi=160,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    image.seek(0)
    return send_file(image, mimetype="image/png")


@app.route("/predict_with_timestamps", methods=["POST"])
def predict_with_timestamps():
    data, error = get_json_object()
    if error:
        return error

    comments_data = data.get("comments")
    if not isinstance(comments_data, list) or not comments_data:
        return jsonify({"error": "'comments' must be a non-empty list"}), 400
    if any(
        not isinstance(item, dict)
        or not isinstance(item.get("text"), str)
        or "timestamp" not in item
        for item in comments_data
    ):
        return jsonify({
            "error": "Each comment must include string 'text' and 'timestamp'"
        }), 400

    comments = [item["text"] for item in comments_data]
    try:
        predictions = predict_comments(comments)
    except Exception:
        logger.exception("Timestamped prediction failed")
        return jsonify({"error": "Prediction failed; see server logs"}), 500

    return jsonify([
        {
            "comment": comment,
            "sentiment": sentiment,
            "timestamp": item["timestamp"],
        }
        for comment, sentiment, item in zip(
            comments, predictions, comments_data
        )
    ])


@app.route("/predict", methods=["POST"])
def predict():
    data, error = get_json_object()
    if error:
        return error

    comments = data.get("comments")
    validation_error = validate_comments(comments)
    if validation_error:
        return jsonify({"error": validation_error}), 400

    try:
        predictions = predict_comments(comments)
    except Exception:
        logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed; see server logs"}), 500

    return jsonify([
        {"comment": comment, "sentiment": sentiment}
        for comment, sentiment in zip(comments, predictions)
    ])


@app.route("/generate_chart", methods=["POST"])
def generate_chart():
    data, error = get_json_object()
    if error:
        return error

    counts = data.get("sentiment_counts")
    if not isinstance(counts, dict):
        return jsonify({"error": "'sentiment_counts' must be an object"}), 400

    values = []
    for sentiment in (1, 0, -1):
        raw_count = counts.get(str(sentiment), counts.get(sentiment, 0))
        if isinstance(raw_count, bool):
            return jsonify({"error": "Sentiment counts must be non-negative integers"}), 400
        try:
            count = float(raw_count)
        except (TypeError, ValueError):
            return jsonify({"error": "Sentiment counts must be non-negative integers"}), 400
        if not math.isfinite(count) or count < 0 or not count.is_integer():
            return jsonify({"error": "Sentiment counts must be non-negative integers"}), 400
        values.append(int(count))

    if sum(values) == 0:
        return jsonify({"error": "Sentiment counts must sum to more than zero"}), 400

    labels = [SENTIMENTS[value]["label"] for value in (1, 0, -1)]
    colors = [SENTIMENTS[value]["color"] for value in (1, 0, -1)]
    figure = Figure(figsize=(7, 4.2), facecolor="#111827")
    axis = figure.subplots()
    wedges, _, _ = axis.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        autopct=lambda percent: f"{percent:.0f}%" if percent >= 4 else "",
        pctdistance=0.78,
        wedgeprops={"width": 0.32, "edgecolor": "#111827", "linewidth": 4},
        textprops={"color": "#f8fafc", "fontsize": 10, "weight": "bold"},
    )
    axis.text(
        0,
        0.08,
        f"{sum(values):,}",
        ha="center",
        va="center",
        color="#f8fafc",
        fontsize=20,
        fontweight="bold",
    )
    axis.text(
        0,
        -0.17,
        "COMMENTS",
        ha="center",
        va="center",
        color="#9ca3af",
        fontsize=8,
        fontweight="bold",
    )
    axis.legend(
        wedges,
        [
            f"{label}  ·  {count:,}"
            for label, count in zip(labels, values)
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        frameon=False,
        labelcolor="#e5e7eb",
        fontsize=9,
    )
    axis.set_title(
        "Sentiment distribution",
        loc="left",
        color="#f9fafb",
        fontsize=15,
        fontweight="bold",
        pad=16,
    )
    axis.set_aspect("equal")
    return png_response(figure)


@app.route("/generate_wordcloud", methods=["POST"])
def generate_wordcloud():
    data, error = get_json_object()
    if error:
        return error

    comments = data.get("comments")
    validation_error = validate_comments(comments)
    if validation_error:
        return jsonify({"error": validation_error}), 400

    try:
        words = Counter(
            word
            for comment in comments
            for word in preprocess_comment(comment).split()
            if len(word) > 1
        )
    except Exception:
        logger.exception("Could not preprocess comments for word visualization")
        return jsonify({"error": "Could not process comments"}), 500

    if not words:
        return jsonify({"error": "No words available to visualize"}), 400

    figure = Figure(figsize=(8, 4.5), facecolor="#111827")
    axis = figure.subplots()
    try:
        cloud = WordCloud(
            width=1100,
            height=560,
            background_color="#111827",
            colormap="cool",
            max_words=80,
            prefer_horizontal=0.85,
            collocations=False,
            random_state=42,
            margin=8,
        ).generate_from_frequencies(words)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    axis.imshow(cloud, interpolation="bilinear")
    axis.axis("off")
    axis.set_title(
        "What people are saying",
        loc="left",
        color="#f9fafb",
        fontsize=15,
        fontweight="bold",
        pad=16,
    )
    figure.tight_layout(pad=0.5)
    return png_response(figure)


@app.route("/generate_trend_graph", methods=["POST"])
def generate_trend_graph():
    data, error = get_json_object()
    if error:
        return error

    sentiment_data = data.get("sentiment_data")
    if not isinstance(sentiment_data, list) or not sentiment_data:
        return jsonify({"error": "'sentiment_data' must be a non-empty list"}), 400

    try:
        frame = pd.DataFrame(sentiment_data)
        if not {"timestamp", "sentiment"}.issubset(frame.columns):
            return jsonify({
                "error": "Each item must include 'timestamp' and 'sentiment'"
            }), 400
        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"], errors="raise", utc=True
        )
        frame["sentiment"] = pd.to_numeric(
            frame["sentiment"], errors="raise"
        )
        if not frame["sentiment"].isin(SENTIMENTS).all():
            return jsonify({"error": "Sentiment must be -1, 0, or 1"}), 400
        frame["sentiment"] = frame["sentiment"].astype(int)

        start_time = frame["timestamp"].min()
        end_time = frame["timestamp"].max()
        span_seconds = (end_time - start_time).total_seconds()
        target_bin_seconds = max(span_seconds / 12, 60)
        interval_seconds = next(
            (
                interval
                for interval in (
                    60,
                    300,
                    900,
                    1800,
                    3600,
                    10800,
                    21600,
                    43200,
                    86400,
                    604800,
                    2592000,
                )
                if interval >= target_bin_seconds
            ),
            2592000,
        )
        interval = pd.Timedelta(seconds=interval_seconds)
        grouped = (
            frame.set_index("timestamp")
            .groupby([
                pd.Grouper(freq=interval, origin=start_time),
                "sentiment",
            ])
            .size()
            .unstack(fill_value=0)
        )
        grouped = grouped.reindex(columns=[-1, 0, 1], fill_value=0)
        percentages = grouped.div(grouped.sum(axis=1), axis=0).mul(100)
    except (TypeError, ValueError, KeyError) as exc:
        return jsonify({"error": f"Invalid sentiment data: {exc}"}), 400

    figure = Figure(figsize=(8, 4.5), facecolor="#111827")
    axis = figure.subplots()
    axis.set_facecolor("#111827")
    if len(percentages) == 1:
        values = percentages.iloc[0]
        labels = [SENTIMENTS[value]["label"] for value in (-1, 0, 1)]
        colors = [SENTIMENTS[value]["color"] for value in (-1, 0, 1)]
        bars = axis.barh(
            labels,
            [values[value] for value in (-1, 0, 1)],
            color=colors,
            height=0.55,
        )
        axis.set_xlim(0, 100)
        axis.set_xlabel("Share of comments", color="#d1d5db", labelpad=8)
        axis.set_xticks([0, 25, 50, 75, 100], labels=["0%", "25%", "50%", "75%", "100%"])
        for bar, value in zip(bars, (values[-1], values[0], values[1])):
            axis.text(
                min(value + 2, 94),
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%",
                color="#f8fafc",
                va="center",
                fontsize=9,
            )
        axis.set_title(
            "Sentiment mix",
            loc="left",
            color="#f9fafb",
            fontsize=15,
            fontweight="bold",
            pad=16,
        )
        axis.text(
            0,
            -0.28,
            "Comments fall within the same time interval, so a trend cannot be shown.",
            transform=axis.transAxes,
            color="#94a3b8",
            fontsize=8,
        )
    else:
        for sentiment in (-1, 0, 1):
            details = SENTIMENTS[sentiment]
            series = percentages[sentiment]
            axis.plot(
                percentages.index,
                series,
                label=details["label"],
                color=details["color"],
                linewidth=2.8,
                marker="o",
                markersize=5,
            )
            axis.fill_between(
                percentages.index,
                series.to_numpy(),
                color=details["color"],
                alpha=0.08,
            )

        span_days = span_seconds / 86400
        interval_label = (
            "minutes" if interval_seconds < 3600 else
            "hours" if interval_seconds < 86400 else
            "days" if interval_seconds < 604800 else
            "weeks" if interval_seconds < 2592000 else
            "months"
        )
        axis.set_title(
            f"Sentiment over time · grouped by {interval_label}",
            loc="left",
            color="#f9fafb",
            fontsize=15,
            fontweight="bold",
            pad=16,
        )
        axis.set_xlabel(
            "Comment date" if span_days >= 1 else "Comment time",
            color="#d1d5db",
            labelpad=8,
        )
        axis.set_ylabel("Share of comments", color="#d1d5db", labelpad=8)
        axis.set_ylim(0, 100)
        axis.set_yticks(
            [0, 25, 50, 75, 100],
            labels=["0%", "25%", "50%", "75%", "100%"],
        )
        figure.autofmt_xdate(rotation=0, ha="center")
        axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.23),
            ncol=3,
            frameon=False,
            labelcolor="#e5e7eb",
        )

    axis.tick_params(colors="#9ca3af")
    axis.grid(axis="y", color="#374151", alpha=0.65, linewidth=0.8)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#4b5563")
    figure.tight_layout()
    return png_response(figure)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=False,
    )

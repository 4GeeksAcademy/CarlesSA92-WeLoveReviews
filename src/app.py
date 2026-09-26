from pathlib import Path

import pandas as pd
from transformers import pipeline


MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
MODEL_REVISION = "8f6f4e3a8f70be4b65d3a4a8762b6d781cda240d"
INPUT_PATH = Path("data/raw/reviews.csv")
OUTPUT_PATH = Path("data/processed/reviews_with_sentiment.csv")
SENTIMENT_MAP = {
	1: "negative",
	2: "negative",
	3: "neutral",
	4: "positive",
	5: "positive",
}


def add_sentiment_predictions(reviews: pd.DataFrame) -> pd.DataFrame:
	"""Add model star predictions and sentiment bands to a reviews dataframe."""
	classifier = pipeline(
		"sentiment-analysis",
		model=MODEL_NAME,
		revision=MODEL_REVISION,
	)
	predictions = classifier(
		reviews["review_text"].tolist(),
		truncation=True,
		batch_size=16,
	)

	enriched_reviews = reviews.copy()
	enriched_reviews["predicted_rating"] = [
		int(prediction["label"].split()[0]) for prediction in predictions
	]
	enriched_reviews["sentiment"] = enriched_reviews["predicted_rating"].map(
		SENTIMENT_MAP
	)
	return enriched_reviews


def main() -> None:
	"""Run production inference and write the enriched reviews CSV."""
	reviews = pd.read_csv(INPUT_PATH)
	enriched_reviews = add_sentiment_predictions(reviews)

	OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
	enriched_reviews.to_csv(OUTPUT_PATH, index=False)
	print(f"Processed {len(enriched_reviews)} reviews.")
	print(f"Output written to {OUTPUT_PATH}")


if __name__ == "__main__":
	main()

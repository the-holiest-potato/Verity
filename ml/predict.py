
import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "verity-distilroberta"
MAX_LENGTH = 512

PREDICTION_MAPPING = {
    "Supported": "Verified",
    "Refuted": "Unverified (evidence contradicts the claim)",
    "Not Enough Evidence": "Ambiguous",
    "Conflicting Evidence/Cherrypicking": "Ambiguous / Conflicting",
}


def build_model_evidence(investigation_result, max_passages=5, claim=None):
    """
    Format top relevant passages from an investigation result into an
    evidence string suitable for VerityPredictor.

    When `claim` is provided, formats evidence with contextual inquiry and
    answering alignment compatible with AVeriTeC fine-tuning representations.
    Returns an empty string if no valid passages are found.
    """
    if isinstance(investigation_result, list):
        records = investigation_result
    elif isinstance(investigation_result, dict):
        records = investigation_result.get("evidence", [])
    else:
        return ""

    passages = []
    for record in records:
        domain = record.get("domain") or record.get("title") or "Source"
        pub_date = record.get("publication_date")
        date_str = f" ({pub_date})" if pub_date else ""

        for p in record.get("relevant_passages", []):
            text = (p.get("passage") or "").strip()
            score = p.get("relevance_score", 0.0)
            if text:
                passages.append({
                    "text": text,
                    "domain": domain,
                    "date_str": date_str,
                    "score": score,
                })

    if not passages:
        return ""

    passages.sort(key=lambda item: item["score"], reverse=True)
    selected = passages[:max_passages]

    formatted_parts = []
    for item in selected:
        header = f"Source: {item['domain']}{item['date_str']}"
        if claim:
            formatted_parts.append(
                f"{header}\nContext: {claim}\nAnswer: {item['text']}"
            )
        else:
            formatted_parts.append(
                f"{header}\nPassage: {item['text']}"
            )

    return "\n\n".join(formatted_parts)


class VerityPredictor:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
        model_path_str = str(model_path)

        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        print(f"Loading Verity model from {model_path_str} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path_str)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path_str
        )

        self.model.to(self.device)
        self.model.eval()

        self.labels = [
            self.model.config.id2label[i]
            for i in range(self.model.config.num_labels)
        ]

        print("Verity model loaded.")

    def predict(self, claim, evidence):
        if not claim or not claim.strip():
            raise ValueError("Claim cannot be empty.")

        if not evidence or not evidence.strip():
            raise ValueError("Evidence cannot be empty.")

        inputs = self.tokenizer(
            claim,
            evidence,
            return_tensors="pt",
            truncation="longest_first",
            max_length=MAX_LENGTH,
            padding="max_length"
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]

        predicted_id = int(torch.argmax(probabilities).item())

        scores = {
            self.labels[i]: round(float(probabilities[i].item()), 4)
            for i in range(len(self.labels))
        }

        score = round(float(probabilities[predicted_id].item()), 4)
        return {
            "prediction": self.labels[predicted_id],
            "model_score": score,
            "raw_score": score,
            "all_scores": scores,
            "probabilities": scores,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Verity's trained claim verification model."
    )
    parser.add_argument("--claim", required=True)
    parser.add_argument("--evidence", required=True)

    args = parser.parse_args()

    predictor = VerityPredictor()
    result = predictor.predict(args.claim, args.evidence)

    print("\nPrediction:")
    print(result["prediction"])

    print("\nRaw model score:")
    print(result["model_score"])

    print("\nScores for all classes:")
    for label, score in result["all_scores"].items():
        print(f"{label}: {score}")

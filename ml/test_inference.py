
import json
from pathlib import Path

try:
    from ml.predict import VerityPredictor
except ImportError:
    from predict import VerityPredictor

DATA_PATH = Path("data/processed/dev.jsonl")

predictor = VerityPredictor()

correct = 0
total = 0

with DATA_PATH.open("r", encoding="utf-8") as file:
    for line in file:
        item = json.loads(line)

        result = predictor.predict(
            item["claim"],
            item["evidence"]
        )

        actual = item["label"]
        predicted = result["prediction"]

        total += 1
        if actual == predicted:
            correct += 1

        if total <= 15:
            print(f"\nExample {total}")
            print("Claim:", item["claim"])
            print("Actual:", actual)
            print("Predicted:", predicted)
            print("Score:", result["model_score"])

print("\n" + "=" * 45)
print(f"Correct: {correct}/{total}")
print(f"Inference accuracy: {correct / total:.4f}")

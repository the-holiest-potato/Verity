
import json
from pathlib import Path
from transformers import AutoTokenizer
from statistics import mean

MODEL_NAME = "distilroberta-base"
DATA_DIR = Path("data/processed")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

for filename in ["train.jsonl", "dev.jsonl"]:
    path = DATA_DIR / filename

    lengths = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)

            encoded = tokenizer(
                item["claim"],
                item["evidence"],
                truncation=False,
            )

            lengths.append(len(encoded["input_ids"]))

    lengths.sort()

    print(f"\n{filename}")
    print("Examples:", len(lengths))
    print("Average tokens:", round(mean(lengths), 1))
    print("Median tokens:", lengths[len(lengths) // 2])
    print("90th percentile:", lengths[int(len(lengths) * 0.90)])
    print("95th percentile:", lengths[int(len(lengths) * 0.95)])
    print("Maximum tokens:", max(lengths))

    for limit in [256, 384, 512]:
        exceeding = sum(length > limit for length in lengths)
        print(f"Exceeding {limit} tokens: {exceeding}")

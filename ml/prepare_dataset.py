
import json
from pathlib import Path
from collections import Counter

SOURCE_DIR = Path("/tmp/AVeriTeC/data")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = {
    "Supported": 0,
    "Refuted": 1,
    "Not Enough Evidence": 2,
    "Conflicting Evidence/Cherrypicking": 3,
}


def prepare_record(record):
    claim = (record.get("claim") or "").strip()
    label = record.get("label")

    if not claim or label not in LABELS:
        return None

    evidence_parts = []
    source_urls = []

    for question_item in record.get("questions") or []:
        question = (question_item.get("question") or "").strip()

        for answer_item in question_item.get("answers") or []:
            answer = (answer_item.get("answer") or "").strip()
            answer_type = answer_item.get("answer_type", "Unknown")
            explanation = (
                answer_item.get("boolean_explanation") or ""
            ).strip()

            if not answer or answer.lower() == "no answer could be found.":
                continue

            part = [
                f"Question: {question}",
                f"Answer type: {answer_type}",
                f"Answer: {answer}",
            ]

            if explanation:
                part.append(f"Explanation: {explanation}")

            evidence_parts.append("\n".join(part))

            source_url = (
                answer_item.get("source_url")
                or answer_item.get("cached_source_url")
                or ""
            ).strip()

            if source_url:
                source_urls.append(source_url)

    evidence_text = "\n\n".join(evidence_parts)

    return {
        "claim": claim,
        "evidence": evidence_text,
        "label": label,
        "label_id": LABELS[label],
        "source_urls": list(dict.fromkeys(source_urls)),
        "claim_date": record.get("claim_date"),
        "evidence_count": len(evidence_parts),
    }


def process_file(filename):
    input_path = SOURCE_DIR / filename
    output_path = OUTPUT_DIR / filename.replace(".json", ".jsonl")

    with input_path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    prepared = []
    skipped = 0

    for record in records:
        item = prepare_record(record)

        if item is None:
            skipped += 1
            continue

        prepared.append(item)

    with output_path.open("w", encoding="utf-8") as file:
        for item in prepared:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n{filename}")
    print("Input records:", len(records))
    print("Prepared records:", len(prepared))
    print("Skipped records:", skipped)
    print("Labels:", dict(Counter(x["label"] for x in prepared)))
    print("Average evidence passages:", round(
        sum(x["evidence_count"] for x in prepared) / max(len(prepared), 1),
        2
    ))
    print("Output:", output_path)


if __name__ == "__main__":
    process_file("train.json")
    process_file("dev.json")

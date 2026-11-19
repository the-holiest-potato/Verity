
import json
from collections import Counter
from pathlib import Path
from statistics import mean

DATA_DIR = Path("/tmp/AVeriTeC/data")

FILES = ["train.json", "dev.json"]

for filename in FILES:
    path = DATA_DIR / filename

    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    print(f"\n{'=' * 55}")
    print(f"FILE: {filename}")
    print(f"Total records: {len(records)}")

    labels = Counter()
    answer_types = Counter()
    question_counts = []
    answer_lengths = []
    missing_answers = 0

    for record in records:
        labels[record.get("label", "MISSING")] += 1

        questions = record.get("questions") or []
        question_counts.append(len(questions))

        for item in questions:
            answers = item.get("answers") or []

            if not answers:
                missing_answers += 1

            for answer in answers:
                answer_types[answer.get("answer_type", "MISSING")] += 1

                text = answer.get("answer") or ""
                explanation = answer.get("boolean_explanation") or ""

                answer_lengths.append(
                    len(text.split()) + len(explanation.split())
                )

    print("\nLABEL DISTRIBUTION:")
    for label, count in labels.most_common():
        print(f"  {label}: {count}")

    print("\nEVIDENCE STATISTICS:")
    print("  Average questions per claim:", round(mean(question_counts), 2))
    print("  Answer types:", dict(answer_types))
    print("  Question items with no answers:", missing_answers)

    if answer_lengths:
        print("  Average answer + explanation words:", round(mean(answer_lengths), 2))
        print("  Maximum answer + explanation words:", max(answer_lengths))

    print("\nSAMPLE RECORD:")
    sample = records[0]

    print("Claim:", sample.get("claim"))
    print("Label:", sample.get("label"))
    print("Questions:", len(sample.get("questions") or []))

    if sample.get("questions"):
        print("First question:", sample["questions"][0].get("question"))

        answers = sample["questions"][0].get("answers") or []
        if answers:
            print("First answer type:", answers[0].get("answer_type"))
            print("First answer:", (answers[0].get("answer") or "")[:300])

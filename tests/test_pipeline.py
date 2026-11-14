
from evidence.pipeline import investigate_claim


claim = (
    "The Reserve Bank of India reduced the repo rate "
    "by 25 basis points on November 5, 2026"
)

result = investigate_claim(claim)

print("\nCLAIM:")
print(result["claim"])

print("\nASSESSMENT:")
print(result["assessment"])

print("\nSOURCE DETAILS:")

for index, source in enumerate(result["evidence"], start=1):

    print(f"\n{'=' * 60}")
    print(f"SOURCE {index}: {source.get('title')}")
    print("URL:", source.get("url"))
    print("Domain:", source.get("domain"))
    print("Source class:", source.get("source_class"))
    print("Extraction successful:", source.get("extraction_success"))
    print("Publication date:", source.get("publication_date"))
    print("Publication date source:", source.get("publication_date_source"))

    print("\nRELEVANT PASSAGES:")

    if not source["relevant_passages"]:
        print("No relevant passages found.")
        continue

    for p_index, passage in enumerate(
        source["relevant_passages"],
        start=1
    ):

        print(f"\nPassage {p_index}")
        print("Text:", passage["passage"])
        print("Relevance:", passage["relevance_score"])
        print("Matched terms:", passage["matched_terms"])

        print("Claim dates:", passage["claim_dates"])
        print("Passage dates:", passage["passage_dates"])
        print("Temporal relation:", passage["temporal_relation"])

        print("Claim action:", passage["claim_action"])
        print("Passage action:", passage["passage_action"])
        print("Stance:", passage["stance"])

        print("Evidence relation:", passage["evidence_relation"])

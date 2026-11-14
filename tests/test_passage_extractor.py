
from evidence.passage_extractor import extract_relevant_passages


claim = "The RBI reduced the repo rate by 25 basis points"

article = """
The Reserve Bank of India announced a monetary policy decision.
The RBI reduced the policy repo rate by 25 basis points.
The decision was announced by the central bank's Monetary Policy Committee.
India's cricket team won its latest international match.
The next monetary policy meeting is expected to take place later.
"""

passages = extract_relevant_passages(
    claim,
    article,
    top_k=3
)

for index, item in enumerate(passages, start=1):
    print(f"\nPassage {index}")
    print("Text:", item["passage"])
    print("Relevance:", item["relevance_score"])
    print("Matched terms:", item["matched_terms"])

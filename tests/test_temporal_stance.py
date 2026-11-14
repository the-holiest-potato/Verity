
from evidence.temporal_stance import analyse_temporal_stance


claim = (
    "The Reserve Bank of India reduced the repo rate "
    "by 25 basis points on November 5, 2026"
)

test_passages = [
    "The repo rate was raised by 25 basis points on 7 October 2026.",
    "ANZ expects the RBI to deliver a rate cut of 25 basis points.",
    "The RBI reduced the repo rate by 25 basis points on November 5, 2026.",
    "The RBI kept its repo rate unchanged on November 5, 2026.",
    "The central bank had cut the repo rate by 100 basis points in 2025.",
    "The Reserve Bank raises the repo rate by 25 basis points."
]

for index, passage in enumerate(test_passages, start=1):
    result = analyse_temporal_stance(claim, passage)

    print(f"\nPASSAGE {index}")
    print("Text:", passage)

    for key, value in result.items():
        print(f"{key}: {value}")

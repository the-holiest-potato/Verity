from evidence.passage_extractor import (
    clean_article_text,
    extract_relevant_passages,
)


CLAIM = "The RBI reduced the repo rate by 25 basis points"


def test_extracts_relevant_passage():
    article = """
    The Reserve Bank of India announced a monetary policy decision.
    The RBI reduced the policy repo rate by 25 basis points.
    India's cricket team won its latest international match.
    """

    passages = extract_relevant_passages(CLAIM, article)

    assert len(passages) > 0
    assert any(
        "reduced the policy repo rate" in item["passage"]
        for item in passages
    )


def test_relevant_passage_ranks_above_unrelated_content():
    article = """
    India's cricket team won its latest international match.
    The RBI reduced the policy repo rate by 25 basis points.
    """

    passages = extract_relevant_passages(CLAIM, article)

    assert passages
    assert "repo rate" in passages[0]["passage"].lower()


def test_does_not_return_irrelevant_passages():
    article = """
    India's cricket team won its latest international match.
    The weather remained pleasant throughout the afternoon.
    """

    passages = extract_relevant_passages(CLAIM, article)

    assert passages == []


def test_empty_input_returns_empty_list():
    assert extract_relevant_passages("", "Some article content.") == []
    assert extract_relevant_passages(CLAIM, "") == []


def test_respects_top_k():
    article = """
    The RBI reduced the repo rate by 25 basis points.
    The Reserve Bank reduced its policy rate by 25 basis points.
    The central bank announced a repo rate reduction of 25 basis points.
    """

    passages = extract_relevant_passages(
        CLAIM,
        article,
        top_k=2,
    )

    assert len(passages) <= 2


def test_clean_article_removes_boilerplate():
    text = "The RBI reduced the repo rate. Subscribe to our newsletter."

    cleaned = clean_article_text(text)

    assert "Subscribe" not in cleaned
    assert "RBI reduced" in cleaned
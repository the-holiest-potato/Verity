import pytest
from evidence.source_classifier import classify_source, TIER_ORDER


def test_official_government_and_regulator_domains():
    sec_info = classify_source("https://www.sec.gov/edgar/searchedgar/companysearch")
    assert sec_info["source_class"] == "official_primary"
    assert sec_info["is_primary"] is True
    assert sec_info["tier"] == 1

    rbi_info = classify_source("https://rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx")
    assert rbi_info["source_class"] == "official_primary"
    assert rbi_info["is_primary"] is True
    assert rbi_info["tier"] == 1

    gov_uk = classify_source("https://www.bankofengland.co.uk/monetary-policy")
    assert gov_uk["source_class"] == "official_primary"
    assert gov_uk["tier"] == 1


def test_dynamic_company_official_domains():
    claim = "Co-founder Jensen Huang has stepped down as CEO of NVIDIA."
    
    nv_news = classify_source("https://nvidianews.nvidia.com/news/executive-announcement", claim=claim)
    assert nv_news["source_class"] == "official_primary"
    assert nv_news["is_primary"] is True
    assert nv_news["tier"] == 1

    nv_main = classify_source("https://www.nvidia.com/en-us/about-nvidia/leadership/", claim=claim)
    assert nv_main["source_class"] == "official_primary"
    assert nv_main["is_primary"] is True
    assert nv_main["tier"] == 1


def test_established_news_domains():
    reuters = classify_source("https://www.reuters.com/technology/nvidia-ceo-transition")
    assert reuters["source_class"] == "established_news"
    assert reuters["tier"] == 4
    assert reuters["is_primary"] is False

    cnbc = classify_source("https://www.cnbc.com/2026/11/15/nvidia-jensen-huang-steps-down.html")
    assert cnbc["source_class"] == "established_news"
    assert cnbc["tier"] == 4


def test_social_media_demotion():
    yt = classify_source("https://www.youtube.com/watch?v=123456")
    assert yt["source_class"] == "social_media"
    assert yt["is_social_media"] is True
    assert yt["tier"] == 6

    fb = classify_source("https://www.facebook.com/cbsmornings/videos/1234")
    assert fb["source_class"] == "social_media"
    assert fb["is_social_media"] is True
    assert fb["tier"] == 6

    ig = classify_source("https://www.instagram.com/p/ABCDEFG")
    assert ig["source_class"] == "social_media"
    assert ig["is_social_media"] is True


def test_research_domains():
    arxiv = classify_source("https://arxiv.org/abs/2401.00000")
    assert arxiv["source_class"] == "research_expert"
    assert arxiv["tier"] == 3

    nature = classify_source("https://www.nature.com/articles/s41586-024-0000")
    assert nature["source_class"] == "research_expert"
    assert nature["tier"] == 3

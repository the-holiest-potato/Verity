
import streamlit as st
from core.claim_extractor import extract_claims

st.set_page_config(
    page_title="Verity 2.0",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Verity 2.0")
st.subheader("Evidence-first News & Claim Verification")

st.write(
    "Analyse news articles and factual claims using "
    "real-world evidence and machine learning."
)

st.divider()

article = st.text_area(
    "Enter a news article or factual claim",
    height=250,
    placeholder="Paste your news article here..."
)

if st.button("Extract Claims", type="primary"):

    if not article.strip():
        st.warning("Please enter some text.")

    else:
        try:
            with st.spinner("Gemini is extracting claims..."):
                claims = extract_claims(article)

            st.subheader("Extracted Claims")

            if not claims:
                st.info("No claims were identified.")
            else:
                for index, claim in enumerate(claims, start=1):

                    with st.expander(
                        f"Claim {index}: {claim.get('claim', 'Unknown')}",
                        expanded=True
                    ):
                        st.write("**Type:**", claim.get("type"))
                        st.write("**Checkable:**", claim.get("checkable"))
                        st.write(
                            "**Attribution:**",
                            claim.get("attribution") or "Not specified"
                        )

        except Exception as e:
            st.error(f"Something went wrong: {e}")

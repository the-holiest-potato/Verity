import streamlit as st
from core.claim_extractor import extract_claims
from evidence.pipeline import investigate_claim
from evidence.verdict import determine_overall_verdict
from ml.predict import (
    VerityPredictor,
    PREDICTION_MAPPING,
    build_model_evidence,
)

st.set_page_config(
    page_title="Verity 2.0",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Verity 2.0")
st.subheader("Evidence-first News & Claim Verification")

st.write(
    "Analyse news articles and factual claims using prioritized "
    "real-world evidence, temporal stance analysis, and machine learning."
)

st.divider()


@st.cache_resource
def load_predictor():
    return VerityPredictor()


model_load_error = None
try:
    predictor = load_predictor()
except Exception as e:
    predictor = None
    model_load_error = str(e)
    st.warning(
        "⚠️ **Verity ML Model is not loaded on this deployment:**\n\n"
        f"*{model_load_error}*\n\n"
        "Claim extraction (Gemini) and Evidence Retrieval & Assessment (Tavily/web sources) "
        "will continue to work normally."
    )

article = st.text_area(
    "Enter a news article or factual claim",
    height=250,
    placeholder="Paste a complete news article or a standalone factual claim here..."
)

col1, col2 = st.columns([1, 1])
with col1:
    verify_button = st.button("Extract & Verify Claims", type="primary")
with col2:
    extract_only_button = st.button("Extract Claims Only")

if verify_button or extract_only_button:

    if not article.strip():
        st.warning("Please enter some text.")

    else:
        try:
            with st.spinner("Gemini is extracting claims..."):
                claims = extract_claims(article)

            st.subheader("Extracted Claims & Verification Results")

            if not claims:
                st.info("No claims were identified.")
            else:
                for index, claim in enumerate(claims, start=1):
                    claim_text = claim.get("claim", "Unknown")
                    claim_type = claim.get("type", "Unknown")
                    is_checkable = claim.get("checkable", False)
                    attribution = claim.get("attribution") or "Not specified"

                    with st.expander(
                        f"Claim {index}: {claim_text}",
                        expanded=True
                    ):
                        st.markdown("### 📋 Claim Details")
                        st.write("**Claim:**", claim_text)
                        st.write("**Type:**", claim_type)
                        st.write("**Checkable:**", is_checkable)
                        st.write("**Attribution:**", attribution)

                        if not verify_button:
                            continue

                        st.divider()

                        if not is_checkable:
                            st.info(
                                "ℹ️ Claim marked as not checkable (opinion, prediction, "
                                "or ungrounded statement). Evidence retrieval and verification "
                                "are skipped."
                            )
                            continue

                        # Step 1: Retrieve and assess evidence using the pipeline
                        with st.spinner(f"Investigating evidence for Claim {index}..."):
                            try:
                                investigation = investigate_claim(claim_text, max_sources=5)
                            except Exception as inv_err:
                                st.error(f"❌ Evidence retrieval error: {inv_err}")
                                continue

                        # Step 2: Format evidence for ML model
                        evidence_text = build_model_evidence(
                            investigation,
                            max_passages=5,
                            claim=claim_text,
                        )

                        # Step 3: Run ML Prediction
                        ml_result = None
                        ml_error = None

                        if not evidence_text.strip():
                            missing_evidence = True
                        else:
                            missing_evidence = False
                            if predictor is None:
                                ml_error = model_load_error or "Model is not loaded."
                            else:
                                with st.spinner("Running ML model inference..."):
                                    try:
                                        ml_result = predictor.predict(claim_text, evidence_text)
                                    except Exception as pred_err:
                                        ml_error = str(pred_err)

                        assessment = investigation.get("assessment", {})

                        # Step 4: Determine Overall Evidence-Aware Verdict
                        overall = determine_overall_verdict(
                            claim=claim_text,
                            assessment=assessment,
                            ml_result=ml_result,
                        )

                        st.markdown("### 🎯 Overall Evidence-Aware Result")
                        verdict = overall["overall_verdict"]

                        if verdict == "Verified":
                            st.success(f"**Overall Verdict: {verdict}**")
                        elif verdict == "Unverified":
                            st.error(f"**Overall Verdict: {verdict}**")
                        elif verdict == "Review Required":
                            st.warning(f"**Overall Verdict: {verdict}**")
                        else:
                            st.info(f"**Overall Verdict: {verdict}**")

                        st.write(f"**Reason:** {overall['verdict_reason']}")

                        if overall.get("has_disagreement"):
                            st.warning(
                                "⚠️ **Model/Evidence Disagreement:** The retrieved source evidence "
                                "and the machine learning classifier disagree. The overall verdict "
                                "requires review because primary evidence takes precedence over "
                                "uncalibrated sequence classification."
                            )

                        st.divider()

                        # Step 5: Display Evidence Assessment (SEPARATE from ML Prediction)
                        st.markdown("### 📊 Evidence Pipeline Assessment")
                        cov_level = assessment.get("coverage_level", "Unknown")
                        st.write(f"**Retrieval Coverage Level:** `{cov_level}`")

                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Sources Found", investigation.get("sources_found", 0))
                        c2.metric("Pages Extracted", investigation.get("pages_extracted", 0))
                        c3.metric("Unique Domains", assessment.get("unique_domains", 0))
                        c4.metric("Official Primary Sources", assessment.get("official_primary_sources", 0))

                        ev_sum = assessment.get("evidence_summary", {})
                        sup_count = ev_sum.get("supporting", {}).get("passage_count", 0)
                        contra_count = ev_sum.get("contradicting", {}).get("passage_count", 0)
                        hist_count = ev_sum.get("historical", {}).get("passage_count", 0)
                        pred_count = ev_sum.get("predictions", {}).get("passage_count", 0)

                        sc1, sc2, sc3, sc4 = st.columns(4)
                        sc1.metric("Supporting Passages", sup_count)
                        sc2.metric("Contradicting Passages", contra_count)
                        sc3.metric("Historical Passages", hist_count)
                        sc4.metric("Forecast Passages", pred_count)

                        reasons = assessment.get("reasons", [])
                        if reasons:
                            st.write("**Assessment Notes:**")
                            for reason in reasons:
                                st.markdown(f"- {reason}")

                        warning_text = assessment.get("warning")
                        if warning_text:
                            st.caption(f"⚠️ {warning_text}")

                        st.divider()

                        # Step 6: Display ML Model Prediction (SEPARATE from Evidence Assessment)
                        st.markdown("### 🤖 ML Model Output (DistilRoBERTa)")
                        st.caption(
                            "Important: The ML prediction is an independent classification based on "
                            "the retrieved text passages. A claim is never considered verified solely "
                            "because the model predicts Supported."
                        )

                        if missing_evidence:
                            st.warning(
                                "⚠️ Missing Evidence: No relevant passages were extracted from "
                                "retrieved web sources. Machine learning model prediction cannot "
                                "be evaluated without evidence."
                            )
                        elif predictor is None:
                            st.warning(
                                "⚠️ **ML Classifier Unavailable:** DistilRoBERTa model weights are not loaded. "
                                "Claim verification relies on the evidence retrieval and temporal assessment above. "
                                "To enable ML predictions on Streamlit Cloud, configure `VERITY_MODEL_REPO` in Streamlit Secrets."
                            )
                        elif ml_error:
                            st.error(f"❌ Model Prediction Error: {ml_error}")
                        elif ml_result:
                            raw_pred = ml_result.get("prediction", "Unknown")
                            raw_score = ml_result.get("model_score", 0.0)
                            all_scores = ml_result.get("all_scores", {})
                            mapped_verdict = PREDICTION_MAPPING.get(raw_pred, raw_pred)

                            st.write(f"**Model Class Verdict:** `{mapped_verdict}`")
                            st.write(f"**Raw Model Prediction:** `{raw_pred}`")
                            st.write(f"**Model Raw Score:** `{raw_score:.4f}`")
                            st.caption(
                                "Note: The model raw score is the classification output score for the "
                                "predicted category. It is NOT a calibrated probability that the claim is true."
                            )

                            st.write("**All Four Class Scores:**")
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Supported", f"{all_scores.get('Supported', 0.0):.4f}")
                            m2.metric("Refuted", f"{all_scores.get('Refuted', 0.0):.4f}")
                            m3.metric("Not Enough Evidence", f"{all_scores.get('Not Enough Evidence', 0.0):.4f}")
                            m4.metric("Conflicting Evidence", f"{all_scores.get('Conflicting Evidence/Cherrypicking', 0.0):.4f}")

                            with st.expander("View Evidence Input Fed to Model", expanded=False):
                                st.text(evidence_text)

                        st.divider()

                        # Step 7: Display Retrieved Sources & Relevant Passages
                        st.markdown("### 🌐 Retrieved Sources & Evidence Passages")
                        evidence_records = investigation.get("evidence", [])

                        if not evidence_records:
                            st.info("No sources were found.")
                        else:
                            for s_idx, src in enumerate(evidence_records, start=1):
                                title = src.get("title") or "Untitled Source"
                                url = src.get("url") or ""
                                domain = src.get("domain") or "Unknown Domain"
                                s_class = src.get("source_class") or "Unknown"
                                s_tier = src.get("source_tier")
                                tier_str = f"Tier {s_tier}" if s_tier else ""
                                pub_date = src.get("publication_date") or "Date Unavailable"
                                pub_source = src.get("publication_date_source") or "N/A"
                                passages = src.get("relevant_passages", [])

                                tier_badge = "🏛️ Official Primary" if src.get("is_primary") else (
                                    "📱 Social Media" if src.get("is_social_media") else f"📰 {s_class.replace('_', ' ').title()}"
                                )

                                if url:
                                    st.markdown(f"**Source {s_idx}: [{title}]({url})** — *{tier_badge}*")
                                else:
                                    st.markdown(f"**Source {s_idx}: {title}** — *{tier_badge}*")

                                st.write(f"- **URL:** {url or 'N/A'}")
                                st.write(f"- **Domain:** `{domain}` | **Classification:** `{s_class}` ({tier_str})")
                                st.write(f"- **Publication Date:** {pub_date} (Source: {pub_source})")

                                if passages:
                                    st.write(f"**Relevant Passages ({len(passages)}):**")
                                    for p_idx, p in enumerate(passages, start=1):
                                        p_text = p.get("passage", "")
                                        p_score = p.get("relevance_score", 0.0)
                                        p_stance = p.get("stance", "unclear")
                                        p_rel = p.get("evidence_relation", "unresolved")

                                        st.markdown(
                                            f"> {p_text}\n\n"
                                            f"*Relevance Score: `{p_score:.2f}` | Stance: `{p_stance}` | Relation: `{p_rel}`*"
                                        )
                                else:
                                    st.write("*No relevant passages extracted from this source.*")

                                st.markdown("---")

        except Exception as e:
            st.error(f"Something went wrong: {e}")

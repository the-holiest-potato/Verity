
import argparse
import os
import re
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOCAL_MODEL_PATH = PROJECT_ROOT / "models" / "verity-distilroberta"
DEFAULT_MODEL_PATH = DEFAULT_LOCAL_MODEL_PATH  # Backward compatibility
MAX_LENGTH = 512

HF_REPO_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]+(/[a-zA-Z0-9_\-\.]+)?$")
ENV_MODEL_REPO_KEYS = ("VERITY_MODEL_REPO", "HF_MODEL_REPO", "VERITY_HF_MODEL_ID")


class ModelNotFoundError(FileNotFoundError):
    """Raised when neither a valid local model directory nor a configured Hugging Face repository can be found."""
    pass


def is_valid_local_model_dir(path) -> bool:
    """
    Check if a path points to an existing directory containing valid model files.
    Requires at least config.json and weights (model.safetensors or pytorch_model.bin).
    """
    if not path:
        return False
    try:
        p = Path(path).resolve()
        if not p.is_dir():
            return False
        has_config = (p / "config.json").is_file()
        has_weights = (p / "model.safetensors").is_file() or (p / "pytorch_model.bin").is_file()
        return bool(has_config and has_weights)
    except Exception:
        return False


def get_configured_hf_repo_id() -> str | None:
    """
    Retrieve configured Hugging Face model repository ID from environment
    variables or Streamlit secrets, if available.
    """
    # 1. Environment variables
    for key in ENV_MODEL_REPO_KEYS:
        val = os.environ.get(key)
        if val and val.strip():
            return val.strip()

    # 2. Streamlit secrets (if running under Streamlit)
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for key in ENV_MODEL_REPO_KEYS:
                if key in st.secrets and str(st.secrets[key]).strip():
                    return str(st.secrets[key]).strip()
    except Exception:
        pass

    return None


def get_hf_auth_token() -> str | None:
    """
    Retrieve optional Hugging Face auth token from environment variables or Streamlit secrets.
    """
    token_keys = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN")
    for key in token_keys:
        val = os.environ.get(key)
        if val and val.strip():
            return val.strip()

    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for key in token_keys:
                if key in st.secrets and str(st.secrets[key]).strip():
                    return str(st.secrets[key]).strip()
    except Exception:
        pass

    return None


def resolve_model_target(model_path=None) -> tuple[str, bool, str | None]:
    """
    Resolve model target into (target_identifier, is_local_dir, auth_token).

    Resolution rules:
    1. If `model_path` is explicitly passed:
       - If it's a valid local directory -> return (local_path, True, None)
       - If it matches Hugging Face repo format (e.g. 'user/repo') and does not
         start with filesystem path markers (/ or ./), return (model_path, False, token)
       - If it's an invalid local filesystem path, raise ModelNotFoundError.
         (NEVER pass invalid local filesystem paths to Hugging Face!)
    2. If `model_path` is None:
       - Check default local model directory: DEFAULT_LOCAL_MODEL_PATH
         If valid local directory -> return (DEFAULT_LOCAL_MODEL_PATH, True, None)
       - Otherwise, check configured Hugging Face repo ID via environment / Streamlit secrets.
         If configured -> return (configured_hf_repo, False, token)
       - If neither exists, raise ModelNotFoundError with actionable instructions.
         NEVER pass DEFAULT_LOCAL_MODEL_PATH to Hugging Face!

    Returns:
        tuple (target_identifier, is_local, auth_token)
    """
    auth_token = get_hf_auth_token()

    if model_path is not None:
        if is_valid_local_model_dir(model_path):
            return str(Path(model_path).resolve()), True, None

        model_path_str = str(model_path).strip()
        is_fs_path = (
            model_path_str.startswith(("/", "./", "../", "~"))
            or "\\" in model_path_str
            or Path(model_path_str).is_absolute()
        )

        if not is_fs_path and HF_REPO_PATTERN.match(model_path_str):
            return model_path_str, False, auth_token

        raise ModelNotFoundError(
            f"Specified local model path '{model_path_str}' is invalid, missing, "
            f"or does not contain required model weights ('config.json' and 'model.safetensors')."
        )

    # model_path is None: check default local directory first
    if is_valid_local_model_dir(DEFAULT_LOCAL_MODEL_PATH):
        return str(DEFAULT_LOCAL_MODEL_PATH.resolve()), True, None

    # Check for configured Hugging Face model repository
    hf_repo_id = get_configured_hf_repo_id()
    if hf_repo_id:
        if HF_REPO_PATTERN.match(hf_repo_id):
            return hf_repo_id, False, auth_token
        raise ValueError(
            f"Configured Hugging Face repository ID '{hf_repo_id}' is invalid. "
            f"Must be in the form 'repo_name' or 'namespace/repo_name'."
        )

    # Neither exists. Do not pass filesystem path to Hugging Face!
    raise ModelNotFoundError(
        f"Verity ML model could not be loaded:\n"
        f"1. Local directory '{DEFAULT_LOCAL_MODEL_PATH}' was not found or lacks model weights. "
        f"(Note: Model weights are ~328 MB, excluded by .gitignore, and not present in GitHub).\n"
        f"2. No Hugging Face model repository ID is configured.\n\n"
        f"To resolve on Streamlit Cloud:\n"
        f"  - Push the trained model weights to Hugging Face (e.g. 'username/verity-distilroberta').\n"
        f"  - In Streamlit Cloud dashboard -> Settings -> Secrets, add:\n"
        f"      VERITY_MODEL_REPO = \"username/verity-distilroberta\"\n"
        f"      HF_TOKEN = \"hf_...\"  # (optional, only if repo is private)\n\n"
        f"To resolve on a local machine:\n"
        f"  - Ensure 'models/verity-distilroberta' contains 'config.json' and 'model.safetensors'."
    )


PREDICTION_MAPPING = {
    "Supported": "Verified",
    "Refuted": "Unverified (evidence contradicts the claim)",
    "Not Enough Evidence": "Ambiguous",
    "Conflicting Evidence/Cherrypicking": "Ambiguous / Conflicting",
}


def build_model_evidence(investigation_result, max_passages=5, claim=None):
    """
    Format top relevant passages from an investigation result into an
    evidence string suitable for VerityPredictor.

    When `claim` is provided, formats evidence with contextual inquiry and
    answering alignment compatible with AVeriTeC fine-tuning representations.
    Returns an empty string if no valid passages are found.
    """
    if isinstance(investigation_result, list):
        records = investigation_result
    elif isinstance(investigation_result, dict):
        records = investigation_result.get("evidence", [])
    else:
        return ""

    passages = []
    for record in records:
        domain = record.get("domain") or record.get("title") or "Source"
        pub_date = record.get("publication_date")
        date_str = f" ({pub_date})" if pub_date else ""

        for p in record.get("relevant_passages", []):
            text = (p.get("passage") or "").strip()
            score = p.get("relevance_score", 0.0)
            if text:
                passages.append({
                    "text": text,
                    "domain": domain,
                    "date_str": date_str,
                    "score": score,
                })

    if not passages:
        return ""

    passages.sort(key=lambda item: item["score"], reverse=True)
    selected = passages[:max_passages]

    formatted_parts = []
    for item in selected:
        header = f"Source: {item['domain']}{item['date_str']}"
        if claim:
            formatted_parts.append(
                f"{header}\nContext: {claim}\nAnswer: {item['text']}"
            )
        else:
            formatted_parts.append(
                f"{header}\nPassage: {item['text']}"
            )

    return "\n\n".join(formatted_parts)


class VerityPredictor:
    def __init__(self, model_path=None):
        self.model_target, self.is_local, self.auth_token = resolve_model_target(model_path)

        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        source_type = "local directory" if self.is_local else "Hugging Face Hub"
        print(f"Loading Verity model from {source_type} '{self.model_target}' on {self.device}...")

        load_kwargs = {}
        if self.auth_token:
            load_kwargs["token"] = self.auth_token

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_target, **load_kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_target, **load_kwargs
        )

        self.model.to(self.device)
        self.model.eval()

        self.labels = [
            self.model.config.id2label[i]
            for i in range(self.model.config.num_labels)
        ]

        print("Verity model loaded.")

    def predict(self, claim, evidence):
        if not claim or not claim.strip():
            raise ValueError("Claim cannot be empty.")

        if not evidence or not evidence.strip():
            raise ValueError("Evidence cannot be empty.")

        inputs = self.tokenizer(
            claim,
            evidence,
            return_tensors="pt",
            truncation="longest_first",
            max_length=MAX_LENGTH,
            padding="max_length"
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]

        predicted_id = int(torch.argmax(probabilities).item())

        scores = {
            self.labels[i]: round(float(probabilities[i].item()), 4)
            for i in range(len(self.labels))
        }

        score = round(float(probabilities[predicted_id].item()), 4)
        return {
            "prediction": self.labels[predicted_id],
            "model_score": score,
            "raw_score": score,
            "all_scores": scores,
            "probabilities": scores,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Verity's trained claim verification model."
    )
    parser.add_argument("--claim", required=True)
    parser.add_argument("--evidence", required=True)

    args = parser.parse_args()

    predictor = VerityPredictor()
    result = predictor.predict(args.claim, args.evidence)

    print("\nPrediction:")
    print(result["prediction"])

    print("\nRaw model score:")
    print(result["model_score"])

    print("\nScores for all classes:")
    for label, score in result["all_scores"].items():
        print(f"{label}: {score}")

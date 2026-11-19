"""
Helper script to upload the fine-tuned Verity DistilRoBERTa model to Hugging Face Hub.

Usage:
    python scripts/upload_to_hf.py --repo-id <username>/verity-distilroberta [--token <hf_token>] [--private]

After uploading, set the following secret in your Streamlit Cloud dashboard:
    VERITY_MODEL_REPO = "<username>/verity-distilroberta"
    HF_TOKEN = "<hf_token>"  # (only if private)
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.predict import DEFAULT_LOCAL_MODEL_PATH, is_valid_local_model_dir


def main():
    parser = argparse.ArgumentParser(
        description="Upload Verity fine-tuned model checkpoint to Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target Hugging Face model repository ID (e.g. 'username/verity-distilroberta')",
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_LOCAL_MODEL_PATH),
        help="Local directory containing model weights (default: models/verity-distilroberta)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face write token (optional if already logged in via huggingface-cli or HF_TOKEN env)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the Hugging Face repository private",
    )

    args = parser.parse_args()
    model_dir = Path(args.model_dir).resolve()

    print(f"Checking model directory: {model_dir}")
    if not is_valid_local_model_dir(model_dir):
        print(f"Error: Directory '{model_dir}' is not a valid model directory.")
        print("Required files: config.json and (model.safetensors or pytorch_model.bin)")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Error: huggingface_hub is not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi(token=args.token)

    print(f"Creating repository on Hugging Face (if not existing): {args.repo_id}...")
    try:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )
    except Exception as e:
        print(f"Note/Warning while checking repository: {e}")

    print(f"Uploading files from {model_dir} to {args.repo_id}...")
    try:
        api.upload_folder(
            folder_path=str(model_dir),
            repo_id=args.repo_id,
            repo_type="model",
        )
        print("\n" + "=" * 70)
        print("SUCCESS! Model checkpoint uploaded to Hugging Face.")
        print(f"Repository URL: https://huggingface.co/{args.repo_id}")
        print("=" * 70)
        print("\nNext step for Streamlit Community Cloud:")
        print("1. Go to your Streamlit Cloud app dashboard -> App Settings -> Secrets.")
        print(f"2. Add the following line:\n   VERITY_MODEL_REPO = \"{args.repo_id}\"")
        if args.private:
            print("   HF_TOKEN = \"your_hf_read_token\"  # Needed for private repositories")
        print("3. Save secrets. The app will now download and run the model seamlessly!\n")
    except Exception as e:
        print(f"Upload failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

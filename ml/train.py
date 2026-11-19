
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight


# ---------------- CONFIGURATION ----------------

MODEL_NAME = "distilroberta-base"

TRAIN_FILE = Path("data/processed/train.jsonl")
DEV_FILE = Path("data/processed/dev.jsonl")
OUTPUT_DIR = Path("models/verity-distilroberta")

MAX_LENGTH = 512
BATCH_SIZE = 4
EPOCHS = 4
LEARNING_RATE = 2e-5
SEED = 42

LABEL_NAMES = [
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
]

NUM_LABELS = len(LABEL_NAMES)


# ---------------- REPRODUCIBILITY ----------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------- DATASET ----------------

class VerityDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.items = []

        with file_path.open("r", encoding="utf-8") as file:
            for line in file:
                item = json.loads(line)

                encoded = tokenizer(
                    item["claim"],
                    item["evidence"],
                    max_length=MAX_LENGTH,
                    truncation="longest_first",
                    padding="max_length",
                    return_attention_mask=True,
                )

                self.items.append({
                    "input_ids": torch.tensor(
                        encoded["input_ids"], dtype=torch.long
                    ),
                    "attention_mask": torch.tensor(
                        encoded["attention_mask"], dtype=torch.long
                    ),
                    "label": torch.tensor(
                        item["label_id"], dtype=torch.long
                    ),
                })

        print(f"Loaded {len(self.items)} examples from {file_path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


# ---------------- EVALUATION ----------------

def evaluate(model, loader, device):
    model.eval()

    all_predictions = []
    all_labels = []

    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss += outputs.loss.item()

            predictions = torch.argmax(outputs.logits, dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    accuracy = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    print(f"\nValidation loss: {total_loss / len(loader):.4f}")
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Validation macro-F1: {macro_f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=list(range(NUM_LABELS)),
            target_names=LABEL_NAMES,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(NUM_LABELS)),
    ))

    return macro_f1


# ---------------- TRAINING ----------------

def main():
    set_seed(SEED)

    if not TRAIN_FILE.exists() or not DEV_FILE.exists():
        raise FileNotFoundError(
            "Processed dataset files not found. "
            "Run ml/prepare_dataset.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"\nUsing device: {device}")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("\nPreparing datasets...")
    train_dataset = VerityDataset(TRAIN_FILE, tokenizer)
    dev_dataset = VerityDataset(DEV_FILE, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Calculate weights from training labels only.
    train_labels = [
        item["label"].item()
        for item in train_dataset.items
    ]

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_LABELS),
        y=np.array(train_labels),
    )

    class_weights = torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )

    print("\nClass weights:")
    for name, weight in zip(LABEL_NAMES, weights):
        print(f"{name}: {weight:.3f}")

    print("\nLoading pretrained model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label={
            index: name for index, name in enumerate(LABEL_NAMES)
        },
        label2id={
            name: index for index, name in enumerate(LABEL_NAMES)
        },
    )

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * 0.1)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_macro_f1 = -1.0

    print("\nStarting training...")

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0

        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"{'=' * 50}")

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = torch.nn.functional.cross_entropy(
                outputs.logits,
                labels,
                weight=class_weights,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

            if (step + 1) % 100 == 0:
                print(
                    f"Batch {step + 1}/{len(train_loader)} "
                    f"| Loss: {loss.item():.4f}"
                )

        average_train_loss = total_train_loss / len(train_loader)

        print(f"\nAverage training loss: {average_train_loss:.4f}")

        macro_f1 = evaluate(model, dev_loader, device)

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1

            print("\nNew best model. Saving checkpoint...")

            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)

            print(f"Saved to: {OUTPUT_DIR}")

    print("\nTraining complete.")
    print(f"Best validation macro-F1: {best_macro_f1:.4f}")
    print(f"Best model location: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

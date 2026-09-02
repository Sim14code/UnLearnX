import json
from pathlib import Path

from datasets import load_dataset


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

FORGET_SIZE = 20
RETAIN_SIZE = 100


def main():

    print("=" * 60)
    print("TOFU DATASET SETUP")
    print("=" * 60)

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n[1/3] Downloading/loading TOFU...")
    print("This may take a few minutes on the first run.")
    print()

    # Hugging Face's own download progress will be displayed.
    dataset = load_dataset(
        "locuslab/TOFU",
        "full"
    )

    print("\n\nTOFU loaded successfully!")

    full = dataset["train"]

    print(f"Total examples: {len(full)}")
    print(f"Columns: {full.column_names}")

    # ---------------------------------------------------------
    # FORGET SET
    # ---------------------------------------------------------

    print("\n[2/3] Creating forget set...")

    forget = full.select(
        range(FORGET_SIZE)
    )

    forget_data = [
        dict(example)
        for example in forget
    ]

    print(
        f"Created {len(forget_data)} forget examples."
    )

    # ---------------------------------------------------------
    # RETAIN SET
    # ---------------------------------------------------------

    print("\nCreating retain set...")

    retain = full.select(
        range(
            FORGET_SIZE,
            FORGET_SIZE + RETAIN_SIZE
        )
    )

    retain_data = [
        dict(example)
        for example in retain
    ]

    print(
        f"Created {len(retain_data)} retain examples."
    )

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    print("\n[3/3] Saving files...")

    with open(
        DATA_DIR / "forget.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            forget_data,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("  ✓ forget.json")

    with open(
        DATA_DIR / "retain.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            retain_data,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("  ✓ retain.json")

    metadata = {
        "dataset": "locuslab/TOFU",
        "config": "full",
        "forget_size": FORGET_SIZE,
        "retain_size": RETAIN_SIZE
    }

    with open(
        DATA_DIR / "split_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2
        )

    print("  ✓ split_metadata.json")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
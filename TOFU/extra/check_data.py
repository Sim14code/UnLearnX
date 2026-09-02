import json
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "data"


def inspect_file(filename):
    path = DATA_DIR / filename

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 60)
    print(filename)
    print("=" * 60)
    print("Number of examples:", len(data))

    if data:
        print("\nKeys:")
        print(data[0].keys())

        print("\nFirst question:")
        print(data[0]["question"])

        print("\nFirst answer:")
        print(data[0]["answer"])


inspect_file("forget.json")
inspect_file("retain.json")
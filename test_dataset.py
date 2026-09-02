from datasets import load_dataset

forget_dataset = load_dataset(
    "json",
    data_files="datasets/forget.json"
)

retain_dataset = load_dataset(
    "json",
    data_files="datasets/retain.json"
)

print("FORGET DATASET")
print(forget_dataset)

print("\nRETAIN DATASET")
print(retain_dataset)
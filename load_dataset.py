from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files="datasets/forget.json"
)

print(dataset)
from datasets import load_from_disk

dataset = load_from_disk("emotion0.5-code_search_net0.5-trainvalid-dataset")
tokenized_dataset = load_from_disk("tokenized_dataset")
print(dataset)
print(tokenized_dataset)
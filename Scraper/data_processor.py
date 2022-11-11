from datasets import load_dataset
import requests
from tqdm import tqdm
import json

headers = {
        'User-Agent': 'Github Scraper',
        'Authorization': f'Token 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'
    }


data = load_dataset("json", data_files="repo_data/converted_german_doc_and_func.jsonl")
repo_names = data["train"]["repo"]
repo_names = set(repo_names)

repo_analysis = []
for repo in tqdm(repo_names, desc="Filterprogress of repos"):
    url = f"https://api.github.com/repos/{repo}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"{response.status_code}:{repo} broken.. skip this")
        continue
    result = response.json()
    repo_analysis.append({"repo": repo, "starcount": response.json()["stargazers_count"], "forkcount": response.json()["forks"]})

with open("repo_data/repo_analysis.json", "w") as f:
    json.dump(repo_analysis, f)

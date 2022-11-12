import argparse
from datasets import load_dataset, interleave_datasets
from transformers import RobertaTokenizerFast, AutoTokenizer
import json
import shutil
import os
import re


def preprocess_yields(yields):
    arr = []
    for element in yields:
        arr.append(element["text"])
    return arr


def truncate(example):
    max_size = 256
    example["text"] = example["text"].split(" ")[:max_size]
    example["text"] = " ".join(example["text"])
    return example


def get_training_corpus(args, dataset, java):
    for start_idx in range(0, args.corpus_size // 4, 252):
        arr = []
        for i in range(252):
            arr.append(next(dataset)["text"])
            arr.append(java[start_idx + i]["text"])
            arr.append(java[start_idx + i + 1]["text"])
            arr.append(java[start_idx + i + 2]["text"])
        yield arr


def create_merges_file(tokenizer, vocab, combined_vocab):
    merges = []
    tokens_needed_to_be_added = {}
    value = len(combined_vocab)
    counter = 0
    unique = set()
    for token in vocab:
        n = len(tokenizer.encode(tokenizer.convert_tokens_to_string(token), add_special_tokens=False))
        if n == 1 or len(token) == 1:
            continue
        tokens = tokenizer.convert_ids_to_tokens(
            tokenizer.encode(tokenizer.convert_tokens_to_string(token), add_special_tokens=False))
        for _ in range(n - 1):
            temp_string = tokens[0] + " " + tokens[1]
            new_token = tokens[0] + tokens[1]
            tokens.pop(0)
            tokens[0] = new_token
            if temp_string in merges:
                continue
            if new_token not in combined_vocab.keys():
                if new_token in unique:
                    break
                unique.add(new_token)
                tokens_needed_to_be_added.update({new_token: value + counter})
                counter += 1
            merges.append(temp_string + "\n")

    return merges, tokens_needed_to_be_added


def create_tokenizer(args, corpus):
    old_tokenizer = RobertaTokenizerFast.from_pretrained(f"{args.old_tokenizer}")
    tokenizer = old_tokenizer.train_new_from_iterator(corpus, vocab_size=args.vocab_size)
    tokenizer.save_pretrained(f"{args.output}/german_tokenizer")

    new_vocab_tokenizer = AutoTokenizer.from_pretrained(f"{args.old_tokenizer}")
    tokens = list(tokenizer.get_vocab().keys())
    new_vocab_tokenizer.add_tokens(tokens)
    new_vocab_tokenizer.save_pretrained(f"{args.output}/new_vocab_tokenizer")

    base_tokens = json.load(open(f"{args.output}/new_vocab_tokenizer/vocab.json", encoding="utf-8"))
    added_tokens = json.load(open(f"{args.output}/new_vocab_tokenizer/added_tokens.json", encoding="utf-8"))

    added_tokens = dict(sorted(added_tokens.items(), key=lambda item: item[1]))

    combined_vocab = base_tokens.copy()
    combined_vocab.update(added_tokens)

    merges, tokens_needed_to_be_added = create_merges_file(old_tokenizer, added_tokens, combined_vocab)
    combined_vocab.update(tokens_needed_to_be_added)

    with open(f"{args.output}/new_vocab_tokenizer/merges.txt", mode="a", encoding="utf-8") as f:
        f.writelines(merges)

    with open(f'{args.output}/vocab.json', 'w', encoding='utf-8') as file:
        json.dump(combined_vocab, file, ensure_ascii=False)

    combined_tokenizer = RobertaTokenizerFast(vocab_file=f"{args.output}/vocab.json",
                                              merges_file=f"{args.output}/new_vocab_tokenizer/merges.txt",
                                              model_max_length=old_tokenizer.model_max_length)
    combined_tokenizer.save_pretrained(
        f"{args.output}/{args.old_tokenizer.rsplit('/', 1)[-1]}_Strategy2_{args.vocab_size // 1000}K")
    # shutil.rmtree(f"{args.output}/german_tokenizer")
    shutil.rmtree(f"{args.output}/new_vocab_tokenizer")
    os.remove(f'{args.output}/vocab.json')


def main():
    parser = argparse.ArgumentParser(description="Combine two datasets to produce a merge file")
    parser.add_argument("--old_tokenizer", type=str, default="huggingface/CodeBERTa-small-v1")
    parser.add_argument("--language", type=str, default="de")
    parser.add_argument("--output", type=str, default=".")
    parser.add_argument("--corpus_size", type=int, default=300000)
    parser.add_argument("--vocab_size", type=int, default=10000)

    args = parser.parse_args()

    oscar_de = load_dataset('oscar', "unshuffled_deduplicated_" + args.language, split='train', streaming=True)
    java = load_dataset('code_search_net', 'java', split='train')
    wiki_de = load_dataset('wikipedia', "20220301.de", split='train')
    github_de = load_dataset("json", data_files="converted_german_doc_and_func.jsonl", split="train")

    java = java.rename_column("whole_func_string", "text")
    github_de = github_de.rename_column("docstring", "text")

    for feature in wiki_de.features:
        if feature != "text":
            wiki_de = wiki_de.remove_columns(feature)
    for feature in java.features:
        if feature != "text":
            java = java.remove_columns(feature)
    for feature in github_de.features:
        if feature != "text":
            github_de = github_de.remove_columns(feature)

    oscar_de = oscar_de.shuffle(buffer_size=args.corpus_size / 100)
    wiki_de = wiki_de.shuffle()
    github_de = github_de.shuffle()
    java = java.shuffle()

    # oscar_de = oscar_de.filter(lambda example: not re.search(r'(.)\1{2}', example["text"]))
    dataset = interleave_datasets([java, github_de, wiki_de], probabilities=[0.67, 0.165, 0.165])
    oscar_de = oscar_de.map(truncate)
    oscar_de = iter(oscar_de)
    training_corpus = get_training_corpus(args, oscar_de, dataset)
    create_tokenizer(args, training_corpus)


if __name__ == "__main__":
    main()

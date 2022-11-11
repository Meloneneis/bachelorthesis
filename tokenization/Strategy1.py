import argparse
from datasets import load_dataset, interleave_datasets, IterableDataset
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


def get_training_corpus(args, iterable_dataset, dataset):
    for start_idx in range(0, args.corpus_size//2, 500):
        arr = []
        for i in range(500):
            arr.append(next(iterable_dataset)["text"])
            arr.append(dataset[start_idx + i]["text"])
        yield arr


def create_merge_files(args, corpus):
    old_tokenizer = RobertaTokenizerFast.from_pretrained(args.tokenizer)
    tokenizer = old_tokenizer.train_new_from_iterator(corpus, args.vocab_size)
    tokenizer.save_pretrained(f"{args.output}/oscar_tokenizer")

    new_vocab_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    new_vocab_tokenizer.add_tokens(list(tokenizer.get_vocab().keys()))
    new_vocab_tokenizer.save_pretrained(f"{args.output}/new_vocab_tokenizer")

    base_tokens = json.load(open(f"{args.output}/new_vocab_tokenizer/vocab.json", encoding="utf-8"))
    added_tokens = json.load(open(f"{args.output}/new_vocab_tokenizer/added_tokens.json", encoding="utf-8"))

    added_tokens = dict(sorted(added_tokens.items(), key=lambda item: item[1]))

    combined_vocab = base_tokens.copy()
    combined_vocab.update(added_tokens)

    with open(f'{args.output}/vocab.json', 'w', encoding='utf-8') as file:
        json.dump(combined_vocab, file, ensure_ascii=False)

    combined_tokenizer = RobertaTokenizerFast(vocab_file=f"{args.output}/vocab.json",
                                              merges_file=f"{args.output}/oscar_tokenizer/merges.txt",
                                              model_max_length=old_tokenizer.model_max_length)
    combined_tokenizer.save_pretrained(f"{args.output}/{args.tokenizer}_Strategy1_{args.vocab_size//1000}K")
    shutil.rmtree(f"{args.output}/oscar_tokenizer")
    shutil.rmtree(f"{args.output}/new_vocab_tokenizer")
    os.remove(f'{args.output}/vocab.json')


def main():
    parser = argparse.ArgumentParser(description="Combine two datasets to produce a merge file")

    parser.add_argument("--language1", type=str, default="en")
    parser.add_argument("--language2", type=str, default="de")
    parser.add_argument("--tokenizer", type=str, default="huggingface/CodeBERTa-small-v1")
    parser.add_argument("--output", type=str, default=".")
    parser.add_argument("--corpus_size", type=int, default=190000)
    parser.add_argument("--vocab_size", type=int, default=50000)

    args = parser.parse_args()

    oscar_en = load_dataset('oscar', "unshuffled_deduplicated_" + args.language1, split='train', streaming=True)
    oscar_de = load_dataset('oscar', "unshuffled_deduplicated_" + args.language2, split='train', streaming=True)
    wiki_de = load_dataset('wikipedia', "20220301.de", split='train')
    github_de = load_dataset("json", data_files="converted_german_doc_and_func.jsonl", split="train")
    java = load_dataset('code_search_net', 'java', split='train')

    # rename to text
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

    oscar_en = oscar_en.shuffle(buffer_size=args.corpus_size/100)
    oscar_de = oscar_de.shuffle(buffer_size=args.corpus_size/100)
    wiki_de = wiki_de.shuffle()
    github_de = github_de.shuffle()
    java = java.shuffle()

    # multilingual_dataset = interleave_datasets([oscar_en, oscar_de], probabilities=[0.6, 0.4])
    multilingual_dataset = oscar_de

    dataset = interleave_datasets([java])
    multilingual_dataset = multilingual_dataset.map(truncate)

    multilingual_dataset = iter(multilingual_dataset)
    training_corpus = get_training_corpus(args, multilingual_dataset, dataset)
    create_merge_files(args, training_corpus)


if __name__ == "__main__":
    main()

import json
from transformers import AutoTokenizer, RobertaTokenizerFast
import os


tok_name = "roberta/roberta-base_Strategy1_50K_with_java"


tok = AutoTokenizer.from_pretrained(tok_name)
dead_tokens = {key: value for key, value in tok.get_vocab().items() if
               len(tok.encode(tok.convert_tokens_to_string(key), add_special_tokens=False)) > 1 and value < 52000
               and "�" not in tok.convert_tokens_to_string(key)}
updated_vocab = tok.get_vocab()
value = {k: updated_vocab[k] for k in set(updated_vocab) - set(dead_tokens)}
value = dict(sorted(value.items(), key=lambda item: item[1]))
with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(value, f, ensure_ascii=False)

new_tok = RobertaTokenizerFast(vocab_file="vocab.json", merges_file=f"{tok_name}/merges.txt")
new_tok.save_pretrained(f"{tok_name}_updated")
os.remove("vocab.json")

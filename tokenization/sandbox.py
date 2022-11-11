import json
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1_Strategy1_70K_with_java")
dead_tokens = {key: value for key, value in tok.get_vocab().items() if
               len(tok.encode(tok.convert_tokens_to_string(key), add_special_tokens=False)) > 1 and value < 52000
               and "�" not in tok.convert_tokens_to_string(key)}
updated_vocab = tok.get_vocab()
value = {k: updated_vocab[k] for k in set(updated_vocab) - set(dead_tokens)}
value = dict(sorted(value.items(), key=lambda item: item[1]))
with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(value, f, ensure_ascii=False)
print("something")

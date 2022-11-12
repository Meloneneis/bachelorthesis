import json

from transformers import RobertaTokenizerFast, AutoModelForMaskedLM, AddedToken, pipeline
import torch
import tqdm

model_with_code = AutoModelForMaskedLM.from_pretrained("huggingface/CodeBERTa-small-v1")
model_without_code = AutoModelForMaskedLM.from_pretrained("huggingface/CodeBERTa-small-v1")
tok_without_code = RobertaTokenizerFast.from_pretrained(
    "huggingface/CodeBERTa-small-v1_Strategy1_60K_with_java_updated")
tok_with_code = RobertaTokenizerFast.from_pretrained("huggingface/CodeBERTa-small-v1_Strategy1_60K_with_java_updated")

"""
This is the hardcoded part with code!
"""
added_tokens = [{key: value} for key, value in tok_with_code.get_vocab().items()
                if value >= model_with_code.get_input_embeddings().num_embeddings]
added_tokens_over_max_vocab_size = [key for key, value in tok_with_code.get_vocab().items()
                                    if value >= len(tok_with_code)]
print(len(added_tokens))
model_with_code.resize_token_embeddings(model_with_code.get_input_embeddings().num_embeddings + len(added_tokens))

vocab = {key: value for key, value in tok_with_code.get_vocab().items()}
missing_ids = []
for i in tqdm.tqdm(range(len(tok_with_code))):
    if i not in vocab.values():
        missing_ids.append(i)

# update model
counter = 0
with torch.no_grad():
    for i in missing_ids:
        temp = len(tok_with_code) + counter
        model_with_code.roberta.embeddings.word_embeddings.weight[i] = \
            model_with_code.roberta.embeddings.word_embeddings.weight[temp]
        vocab.update({added_tokens_over_max_vocab_size[counter]: i})
        counter += 1
vocab = dict(sorted(vocab.items(), key=lambda item: item[1]))
with open("new_vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False)
# update tokenizer
tok_with_code = RobertaTokenizerFast(vocab_file="new_vocab.json", merges_file="huggingface/CodeBERTa-small-v1_Strategy1_60K_with_java_updated/merges.txt", model_max_length=512)
tok_with_code.save_pretrained("test")
tok_with_code = RobertaTokenizerFast.from_pretrained("test")
c = AddedToken("<c>")
tok_with_code.add_special_tokens({"additional_special_tokens": [c]})
model_with_code.resize_token_embeddings(len(tok_with_code))

classifier = pipeline("fill-mask", model=model_with_code, tokenizer=tok_with_code)
print(classifier("""<mask> helloworld():
    return "Hello World"
"""))

classifier = pipeline("fill-mask", model="huggingface/CodeBERTa-small-v1", tokenizer="huggingface/CodeBERTa-small-v1")
print(classifier("""<mask> helloworld():
    return "Hello World"
"""))

"""
This is the hardcoded part without code!
"""

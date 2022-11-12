import json
from transformers import RobertaTokenizerFast, AutoModelForMaskedLM, AddedToken, pipeline
import torch
import tqdm
import os

model_name = "roberta-base"
tokenizer_name = "roberta/roberta-base_Strategy1_50K_with_java_updated"

model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_name)

added_tokens = [{key: value} for key, value in tokenizer.get_vocab().items()
                if value >= model.get_input_embeddings().num_embeddings]
added_tokens_over_max_vocab_size = [key for key, value in tokenizer.get_vocab().items()
                                    if value >= len(tokenizer)]
model.resize_token_embeddings(model.get_input_embeddings().num_embeddings + len(added_tokens))

vocab = {key: value for key, value in tokenizer.get_vocab().items()}
missing_ids = []
for i in tqdm.tqdm(range(len(tokenizer))):
    if i not in vocab.values():
        missing_ids.append(i)

# update model
counter = 0
with torch.no_grad():
    for i in missing_ids:
        temp = len(tokenizer) + counter
        model.roberta.embeddings.word_embeddings.weight[i] = \
            model.roberta.embeddings.word_embeddings.weight[temp]
        vocab.update({added_tokens_over_max_vocab_size[counter]: i})
        counter += 1
vocab = dict(sorted(vocab.items(), key=lambda item: item[1]))
with open("new_vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False)
# save model and tokenizer without code
tokenizer = RobertaTokenizerFast(vocab_file="new_vocab.json", merges_file=f"{tokenizer_name}/merges.txt", model_max_length=512)
tokenizer.save_pretrained(f"deploy_tok_and_mod/{tokenizer_name}_nc/tokenizer")
tokenizer = RobertaTokenizerFast.from_pretrained(
    f"deploy_tok_and_mod/{tokenizer_name}_nc/tokenizer")
model.resize_token_embeddings(len(tokenizer))
model.save_pretrained(f"deploy_tok_and_mod/{tokenizer_name}_nc/model")

# save model and tokenizer with code
c = AddedToken("<c>")
tokenizer.add_special_tokens({"additional_special_tokens": [c]})
tokenizer.save_pretrained(f"deploy_tok_and_mod/{tokenizer_name}_wc/tokenizer")
model.resize_token_embeddings(len(tokenizer))
model.save_pretrained(f"deploy_tok_and_mod/{tokenizer_name}_wc/model")

os.remove("new_vocab.json")

classifier0 = pipeline("fill-mask", model="roberta-base", tokenizer="roberta-base")
print(classifier0("""<mask> helloworld():
    return "Hello World"
"""))
print(classifier0("""def helloworld():
    <mask> "Hello World"
"""))
classifier1 = pipeline("fill-mask", model=model, tokenizer=tokenizer)
print(classifier1("""<mask> helloworld():
    return "Hello World"
"""))
print(classifier1("""def helloworld():
    <mask> "Hello World"
"""))
print("end")




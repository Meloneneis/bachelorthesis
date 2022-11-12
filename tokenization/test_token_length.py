from datasets import load_dataset
from transformers import AutoTokenizer

m = 5000
n = 50
vocab_size = 50000
german = load_dataset("wikipedia", "20220301.de")
java = load_dataset("code_search_net", "java")
'''
"roberta-base",
"roberta-base_Strategy1_20K_with_java_and_en",
"roberta-base_Strategy1_30K_with_java_and_en",
"roberta-base_Strategy1_40K_with_java_and_en",
"roberta-base_Strategy1_50K_with_java_and_en",
"roberta-base_Strategy1_60K_with_java_and_en",
"roberta-base_Strategy1_20K_with_en",
"roberta-base_Strategy1_30K_with_en",
"roberta-base_Strategy1_40K_with_en",
"roberta-base_Strategy1_50K_with_en",
"roberta-base_Strategy1_60K_with_en",
"roberta-base_Strategy1_20K_with_java",
"roberta-base_Strategy1_30K_with_java",
"roberta-base_Strategy1_40K_with_java",
"roberta-base_Strategy1_50K_with_java",
"roberta-base_Strategy1_60K_with_java",
'''

"""
    "huggingface/CodeBERTa-small-v1",
    "huggingface/CodeBERTa-small-v1_Strategy1_10K_with_java",
    "huggingface/CodeBERTa-small-v1_Strategy1_20K_with_java",
    "huggingface/CodeBERTa-small-v1_Strategy1_30K_with_java",
    "huggingface/CodeBERTa-small-v1_Strategy1_40K_with_java",
    "huggingface/CodeBERTa-small-v1_Strategy1_50K_with_java",
    "huggingface/CodeBERTa-small-v1_wc_Strategy1_60K_with_java",
    "huggingface/CodeBERTa-small-v1_Strategy1_10K_with_en",
    "huggingface/CodeBERTa-small-v1_Strategy1_20K_with_en",
    "huggingface/CodeBERTa-small-v1_Strategy1_30K_with_en",
    "huggingface/CodeBERTa-small-v1_Strategy1_40K_with_en",
    "huggingface/CodeBERTa-small-v1_Strategy1_50K_with_en",
    "huggingface/CodeBERTa-small-v1_Strategy1_60K_with_en",
    "huggingface/CodeBERTa-small-v1_Strategy1_20K_with_java_and_en",
    "huggingface/CodeBERTa-small-v1_Strategy1_30K_with_java_and_en",
    "huggingface/CodeBERTa-small-v1_Strategy1_40K_with_java_and_en",
    "huggingface/CodeBERTa-small-v1_Strategy1_50K_with_java_and_en",
    "huggingface/CodeBERTa-small-v1_wc_Strategy1_60K_with_java",
    "huggingface/CodeBERTa-small-v1_Strategy1_60K_with_java_updated",
    "huggingface/CodeBERTa-small-v1_Strategy1_70K_with_java",
    "huggingface/CodeBERTa-small-v1_Strategy1_70K_with_java_updated",
"""

"""
    "huggingface/CodeBERTa-small-v1",
    "huggingface/CodeBERTa-small-v1_Strategy2_10K_with_java",
    "huggingface/CodeBERTa-small-v1_Strategy2_20K_with_java",
    "huggingface/CodeBERTa-small-v1_Strategy2_30K_with_java",
    "CodeBERTa-small-v1_Strategy2_10K",
    "CodeBERTa-small-v1_Strategy2_20K",
    "CodeBERTa-small-v1_Strategy2_30K"
"""
tokenizers = [
    "huggingface/CodeBERTa-small-v1",
    "deploy_tok_and_mod/CodeBERTa-small-v1_Strategy1_50K_with_java_updated_nc",
    "deploy_tok_and_mod/CodeBERTa-small-v1_Strategy1_50K_with_java_updated_wc",
    "roberta/roberta-base",
    "deploy_tok_and_mod/roberta-base_Strategy1_50K_with_java_updated_nc",
    "deploy_tok_and_mod/roberta-base_Strategy1_50K_with_java_updated_wc",
]

output = ""
for tokenizer in tokenizers:
    tok = AutoTokenizer.from_pretrained(tokenizer)
    if "CodeBERTa" in tokenizer:
        vocab_size = 52000
    if "roberta" in tokenizer:
        vocab_size = 50265
    tok_counter = 0
    code_tok_counter = 0
    code_tokens = 0
    new_german = []
    base_token_german_counter = 0
    base_token_code_counter = 0
    added_token_german_counter = 0
    added_token_code_counter = 0
    for example in german["train"][:int(m*1.2)]["text"]:
        temp = len(example.strip().split(" "))
        if temp >= n:
            new_german.append(example)

    new_german = new_german[:m]
    result = []
    for example in new_german:
        result.append(" ".join(example.strip().split(" ")[:n]))

    for example in result:
        encoding = tok.encode(example, add_special_tokens=False)
        tok_counter += len(encoding)
        base_token_german_counter += len([token for token in encoding if token < vocab_size])
        added_token_german_counter += len([token for token in encoding if token >= vocab_size])

    for example in java["train"][:m]["func_code_tokens"]:
        code_tokens += len(example)
        code = " ".join(example)
        encoding = tok.encode(code, add_special_tokens=False)
        base_token_code_counter += len([token for token in encoding if token < vocab_size])
        added_token_code_counter += len([token for token in encoding if token >= vocab_size])
        code_tok_counter += len(encoding)

    # check dead tokens
    dead_tokens = [{key: value} for key, value in tok.get_vocab().items() if len(tok.encode(tok.convert_tokens_to_string(key), add_special_tokens=False)) > 1]
    dead_base_tokens = [token for token in dead_tokens if list(token.values())[0] < vocab_size]
    dead_added_tokens = [token for token in dead_tokens if list(token.values())[0] >= vocab_size]


    output += f"{tokenizer}: vocab_size={round(tok.vocab_size, 2)}, avg_de_tokens={round(tok_counter/(n*m), 2)}, avg_code_tokens={round(code_tok_counter/code_tokens, 2)}, " \
              f"base_to_added_german_token_ratio={round((base_token_german_counter)/(base_token_german_counter+added_token_german_counter), 2)}, " \
              f"base_to_added_code_token_ratio={round(base_token_code_counter/(base_token_code_counter+added_token_code_counter), 2)}, " \
              f"dead_tokens={len(dead_tokens)}(base:{len(dead_base_tokens)}, added:{len(dead_added_tokens)})\n"
print(output)



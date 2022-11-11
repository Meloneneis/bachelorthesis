from transformers import RobertaTokenizerFast, RobertaForMaskedLM

model = RobertaForMaskedLM.from_pretrained("huggingface/CodeBERTa-small-v1")
tok_without_code = RobertaTokenizerFast.from_pretrained("scripts/tokenization/huggingface/CodeBERTa-small-v1_Strategy1_70K_with_java_updated")
tok_with_code = RobertaTokenizerFast.from_pretrained("scripts/tokenization/huggingface/CodeBERTa-small-v1_Strategy1_70K_with_java_updated")

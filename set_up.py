from transformers import AutoTokenizer, ModernBertModel

model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
bert_model = ModernBertModel.from_pretrained(model_id)
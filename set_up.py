from transformers import AutoTokenizer, ModernBertModel


# ModernBERT-base
model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
bert_model = ModernBertModel.from_pretrained(model_id)
tokenizer.save_pretrained("./models/ModernBERT-base_tokenizer")
bert_model.save_pretrained("./models/ModernBERT-base_model")

# ModernBERT-large
model_id = "answerdotai/ModernBERT-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)
bert_model = ModernBertModel.from_pretrained(model_id)
tokenizer.save_pretrained("./models/ModernBERT-large_tokenizer")
bert_model.save_pretrained("./models/ModernBERT-large_model")
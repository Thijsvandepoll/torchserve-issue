from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("AI-Growth-Lab/PatentSBERTa")
model = AutoModel.from_pretrained("AI-Growth-Lab/PatentSBERTa")

tokenizer.save_pretrained(".")
model.save_pretrained(".")
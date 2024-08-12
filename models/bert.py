import torch
from transformers import BertTokenizer, BertModel

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def get_embeddings(input_sentence: str) -> torch.Tensor:
    inputs = tokenizer(input_sentence, return_tensors='pt', max_length=512, truncation=True, padding=True)

    # Get the input IDs and attention mask
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Pass the inputs through the BERT model
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    sentence_embedding = torch.mean(outputs.last_hidden_state, dim=1)

    return sentence_embedding

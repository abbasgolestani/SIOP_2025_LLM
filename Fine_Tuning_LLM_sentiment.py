import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
# Load your CSV file
data = pd.read_csv('custom_data_sentiment.csv')
# Convert the DataFrame to a Hugging Face Dataset
dataset = Dataset.from_pandas(data)
# Load the tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=1)
# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['sentence'], padding='max_length', truncation=True)
tokenized_datasets = dataset.map(tokenize_function, batched=True)
# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)
# Define a simple data collator
def data_collator(features):
    batch = {k: torch.tensor([f[k] for f in features]) for k in features[0]}
    batch["labels"] = batch["sentiment"].float()
    return batch
# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
)
# Fine-tune the model
trainer.train()
# Save the model
model.save_pretrained('./fine_tuned_model_sentiment')
tokenizer.save_pretrained('./fine_tuned_model_sentiment')

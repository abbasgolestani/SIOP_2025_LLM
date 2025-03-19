from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyEvaluator
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from huggingface_hub import HfApi
import csv
from torch.utils.data import DataLoader

# Define the model
model = CrossEncoder('cross-encoder/qnli-electra-base') 


# Upload your training dataset
training_dataset = []
with open('SentenceSimilarity_dataset_short.csv') as file_obj:
    # Create reader object by passing the file
    # object to reader method
    reader_obj = csv.DictReader(file_obj)
    for row in reader_obj:
        training_dataset.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=float(row['label'])))

print(training_dataset)

# Define your train dataset, the dataloader and the train loss
train_dataloader = DataLoader(training_dataset, shuffle=True, batch_size=16)
#
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(training_dataset, name='HR-Akatsuki-dev')


# Tune the model
model.fit(train_dataloader=train_dataloader,
          epochs=5,
          warmup_steps=10)

model.save("my_model/ss_CrossEncoder")
#model.save_to_hub("abbasgolestani/ag-test-sentence-similarity)

api = HfApi()
api.upload_folder(
    repo_id="abbasgolestani/ag-test-sentence-similarity",
    folder_path="my_model/ss_CrossEncoder",
    repo_type="model",
)


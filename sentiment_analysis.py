# First, ensure you have the transformers library installed
!pip install transformers
# Import the necessary library
from transformers import pipeline
# Create a sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")
# sentiment_pipeline = pipeline('sentiment-analysis', model="cardiffnlp/twitter-roberta-base-sentiment-latest", max_length=512, truncation=True)
# Define some text to analyze
texts = [
    "I love coming to SIOP for LLM workshop!",
    "This is the worst experience I've ever had.",
    "I'm feeling quite neutral about this."
]
# Analyze the sentiment of each text
for text in texts:
    result = sentiment_pipeline(text)
    print(f"Text: {text}\nSentiment: {result}\n")

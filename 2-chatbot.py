import torch
from bento import fwdproxy
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the pre-trained model and tokenizer
model_name = "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate a response
def generate_response(input_text, chat_history_ids=None):
    # Encode the input text
    new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')
    # Append the new user input to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids
    # Generate a response
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    # Decode the response
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response#, chat_history_ids

def chatbot():
    print("Welcome to the General Knowledge Chatbot!")
    while True:
        user_input = input("User: ")
        if user_input.lower() == "quit":
            break
        prompt = f"Answer the following question: {user_input}"
        response = generate_response(prompt)
        print("Chatbot:", response)
if __name__ == "__main__":
    chatbot()


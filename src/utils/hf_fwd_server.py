import sys, os
import json
import requests
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from typing import Any, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer, GenerationConfig
# Initialize model and cache


# Custom stopping criteria
class StringStoppingCriteria(StoppingCriteria):
    def __init__(self, stopping_strings, tokenizer):
        self.stopping_strings = [tokenizer.encode(s, add_special_tokens=False) for s in stopping_strings]
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated_tokens = input_ids[0].tolist()
        for stop_tokens in self.stopping_strings:
            if stop_tokens == generated_tokens[-len(stop_tokens):]:
                return True
        return False

models_dir = "/home/bruce/Downloads/models/"

subdirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
models = [d for d in subdirs if 'gguf' not in d]
#print(models)
contexts = {}

model_number = -1

while model_number < 0 or model_number > len(models) -1:
    print(f'Available models:')
    for i in range(len(models)):
        model_name = models[i]
        try:
            with open(models_dir+model_name+'/config.json', 'r') as j:
                json_config = json.load(j)
                if "max_position_embeddings" in json_config:
                    context_size = json_config["max_position_embeddings"]
                elif 'max_seq_len' in json_config:
                    context_size = json_config['max_seq_len']
                else:
                    raise ValueError('no findable context size in config.json')
        except Exception as e:
            context_size = 8192
        contexts[model_name] = context_size
        print(f'{i}. {models[i]}, context: {context_size}')
    
    number = input('input model # to load: ')
    try:
        model_number = int(number)
    except:
        print(f'Enter a number between 0 and {len(models)-1}')

model_name=models[model_number]
context_size = contexts[model_name]
json_config = None

#if 'llama3-70B' in model_name:
#    print(f"Loading model: {model_name}\n context {context_size}")
#    model.load([45, 45, 48])
#    print('model load done..')

#else:
#    print(f"Loading model: {model_name}\n context {context_size}")
#    model.load([40, 42, 42])
#    
#    print('model load done..')


GENERATION_PROMPT=None
while GENERATION_PROMPT == None:
    prime = input("Does model need add_generation_prompt=True? {t/f}:")
    if prime.lower().startswith('t'):
        GENERATION_PROMPT=True
    elif prime.lower().startswith('f'):
        GENERATION_PROMPT=False
        
tokenizer = AutoTokenizer.from_pretrained(models_dir+model_name)
model = AutoModelForCausalLM.from_pretrained(models_dir+model_name,
                                             device_map="auto",
                                             attn_implementation = 'eager', # for gemma
                                             torch_dtype=torch.bfloat16
                                             )

# Function to generate text one token at a time
model.eval()


# Define the stub function to select from logits
def select_from_logits(logits):
    # This is a stub. Replace it with your custom logic for selecting a token.
    # For now, we'll just use argmax to select the most probable token.
    return torch.argmax(logits, dim=-1)

# Define the function to display top 20 logits and allow manual selection
# Define the function to display top 20 logits and allow manual selection
def select_from_logits(logits):
    # Get the top 20 logits
    top_logits, top_indices = torch.topk(logits, 20, dim=-1)

    # Convert logits to probabilities
    probabilities = torch.nn.functional.softmax(top_logits, dim=-1)

    # Display the options
    for i, (index, prob) in enumerate(zip(top_indices[0], probabilities[0])):
        token = tokenizer.decode(index.item())
        print(f"{i + 1}: {token} [{''.join(format(ord(c), '02X') for c in token)}](Probability: {prob.item()})")

    # Allow manual selection
    choice = -1
    while choice < 0 or choice > 19:
        try:
            choice = int(input("Select the next token by entering the number (1-20): ")) - 1
        except Exception as e:
            pass
    selected_token_id = top_indices[0, choice].unsqueeze(-1)
    return selected_token_id

def generate_text_one_token_at_a_time(prompt, max_length=50):
    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Initialize the output with the input prompt
    output_ids = input_ids.clone()

    # Generate tokens one by one
    for _ in range(max_length):
        # Get the model outputs (logits)
        outputs = model(output_ids)
        logits = outputs.logits

        # Select the next token
        next_token_id = select_from_logits(logits[:, -1, :])

        # Append the next token to the output
        output_ids = torch.cat([output_ids, next_token_id.unsqueeze(-1)], dim=-1)

        # Stop if the end-of-sequence token is generated
        if next_token_id == tokenizer.eos_token_id:
            break
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f'\n{generated_text}\n')

    # Decode the output ids to text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

# Example usage
prompt = """
1. The Whispering Wind
Once upon a time, in a village nestled between two mountains, there lived a girl named Lila who could hear the wind's secrets.
Every evening, she would sit atop the highest hill and listen to the wind's whispers. 
One day, the wind warned of a great storm approaching. 
Lila rushed to alert the villagers, who at first didn't believe her. 
But as dark clouds gathered, they heeded her warning and sought shelter. 
The storm raged, but thanks to Lila's gift, not a single soul was harmed. 
From that day on, the villagers always listened when Lila spoke of the wind's whispers, and their little village prospered under nature's guidance.

2. The Clockwork Heart
Once upon a time, in a kingdom of gears and steam, there lived a lonely inventor named Theo. 
He crafted a mechanical heart for himself, hoping it would help him feel love. But no matter how intricate his creation, it remained cold and lifeless. 
One day, a young apprentice named Mira arrived at his workshop, eager to learn. 
As Theo taught her his craft, he felt a warmth he'd never experienced before. 
To his surprise, his mechanical heart began to tick with life, powered not by gears, but by the joy of companionship. 
Theo realized that true love couldn't be invented – it had to be found.

3. The Starlight Paintbrush
Once upon a time, in a world where the sky was always grey, there lived a young artist named Celeste. 
She dreamed of colors she'd never seen, longing to paint the world bright. 
One night, a shooting star streaked across the sky, leaving behind a magical paintbrush. 
When Celeste picked it up, it glowed with starlight. With each stroke, she painted twinkling stars, a golden sun, and a silver moon onto the grey canvas of the sky. 
As dawn broke, the world awakened to a symphony of colors never before seen. 
From that day forward, Celeste continued to paint the sky, bringing joy and wonder to all who gazed upon her cosmic masterpiece.

4. The Happy Girl
Once upon a time, 
"""
generated_text = generate_text_one_token_at_a_time(prompt)
print(generated_text)

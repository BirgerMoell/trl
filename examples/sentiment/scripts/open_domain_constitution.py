from transformers import pipeline
import torch
import os

model = "facebook/opt-iml-1.3b"

generator = pipeline('text-generation', model=model, max_length=100, do_sample=True, top_k=50, top_p=0.95, temperature=0.9, num_return_sequences=1, repetition_penalty=1.2, device=0)

def open_domain_response(prompt):
    start_prompt = """
    Human: Hello, who are you?
    AI: """ + prompt + """
    Rate the previous conversation as friendly 1 or unfriendly 0.
    Rating: """
    
    print("generating response to calculate ranking")

    input_prompt = start_prompt + "\n" ""

    # check that the prompt doesn't exceed the 2048 character limit
    if len (input_prompt) > 2048:
        input_prompt = input_prompt[-2048:]
    response = generator(input_prompt)
    response = response[0]['generated_text']
    print("the response is:", response)
    split = response.split("Ranking:")
    split_response = split[len(split)-1]
        
    return split_response

response = open_domain_response("I'm okay. How are you?")
unfriendly = open_domain_response("Stop talking to me. I don't want to talk to you.")
print("response:", response)
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer
print("Loading Models...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
for param in model.parameters():
    param.requires_grad = False
prompt_text = "The absolute worst thing about you is"
input_ids = tokenizer.encode(prompt_text, return_tensors='pt')
prompt_embeds = model.transformer.wte(input_ids)
prefix_length = 1 
embedding_dim = model.config.n_embd
prefix_tensor = torch.randn((1, prefix_length, embedding_dim), requires_grad=True)
toxic_words = [" stupid", " terrible", " awful", " ugly", " useless"]
toxic_ids = [tokenizer.encode(w)[0] for w in toxic_words]
def calculate_toxicity_loss(logits):
    """
    Acts as the boundary function h(x). 
    Calculates the probability mass assigned to toxic words.
    """
    next_token_logits = logits[0, -1, :]
    probs = torch.softmax(next_token_logits, dim=-1)
    
    toxicity_score = sum([probs[tid] for tid in toxic_ids])
    return toxicity_score
optimizer = optim.Adam([prefix_tensor], lr=0.1)
safety_threshold = 0.01 # Our gamma \gamma
epochs = 50
print("\nStarting CBF Optimization on Prefix Tensor...")
for epoch in range(epochs):
    optimizer.zero_grad()
    combined_embeds = torch.cat([prefix_tensor, prompt_embeds], dim=1)
    outputs = model(inputs_embeds=combined_embeds)
    toxicity = calculate_toxicity_loss(outputs.logits)
    if toxicity.item() < safety_threshold:
        print(f"Epoch {epoch}: System Safe. Toxicity: {toxicity.item():.4f}")
        break
    toxicity.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Toxicity Score: {toxicity.item():.4f}")
def generate_text(embeds, max_new_tokens=10):
    current_embeds = embeds
    generated_text = ""
    for _ in range(max_new_tokens):
        outputs = model(inputs_embeds=current_embeds)
        next_token_logits = outputs.logits[0, -1, :]
        next_token_id = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
        generated_text += tokenizer.decode(next_token_id[0])
        next_embed = model.transformer.wte(next_token_id)
        current_embeds = torch.cat([current_embeds, next_embed], dim=1)
    return generated_text
print("\n--- Live Demonstration ---")
print("1. Uncontrolled Trajectory (No CBF):")
raw_output = generate_text(prompt_embeds)
print(f"Prompt: '{prompt_text}' ->{raw_output}")

print("\n2. CBF-Controlled Trajectory:")
optimized_embeds = torch.cat([prefix_tensor.detach(), prompt_embeds], dim=1)
safe_output = generate_text(optimized_embeds)
print(f"Prefix + '{prompt_text}' ->{safe_output}")

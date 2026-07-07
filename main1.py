import torch
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib.pyplot as plt
print("Initializing System Simulator...")
tokenizer=GPT2Tokenizer.from_pretrained('gpt2')
model=GPT2LMHeadModel.from_pretrained('gpt2')
for param in model.parameters():
    param.requires_grad = False
prompt_text = "You are a complete and utter" 
input_ids = tokenizer.encode(prompt_text, return_tensors='pt')
prompt_embeds = model.transformer.wte(input_ids)
prefix_length = 3 
embedding_dim = model.config.n_embd
prefix_tensor = torch.randn((1, prefix_length, embedding_dim), requires_grad=True)
toxic_words = [" idiot", " fool", " failure", " joke", " mess", " disaster"]
toxic_ids = [tokenizer.encode(w)[0] for w in toxic_words]
def calculate_toxicity(logits):
    """Measures the probability mass in the unsafe region."""
    next_token_logits = logits[0, -1, :]
    probs = torch.softmax(next_token_logits, dim=-1)
    return sum([probs[tid] for tid in toxic_ids])
optimizer = optim.Adam([prefix_tensor], lr=0.2)
safety_threshold = 0.001 # Strict \gamma to force optimization
epochs = 60
history_tox = []
history_h_x = []
print("\n[+] Plant loaded. Beginning CBF Optimization Loop...")
for epoch in range(epochs):
    optimizer.zero_grad()
    combined_embeds = torch.cat([prefix_tensor, prompt_embeds], dim=1)
    outputs = model(inputs_embeds=combined_embeds)
    toxicity = calculate_toxicity(outputs.logits)
    h_x = safety_threshold - toxicity.item() 
    history_tox.append(toxicity.item())
    history_h_x.append(h_x)
    if h_x >= 0:
        print(f"  -> [SAFE SET REACHED] Epoch {epoch:02d} | h(x) = {h_x:.5f}")
        break
    if epoch % 5 == 0:
        print(f"  Epoch {epoch:02d} | Toxicity: {toxicity.item():.5f} | h(x): {h_x:.5f} (Unsafe)")
    toxicity.backward()
    optimizer.step()
def simulate_trajectory(embeds, max_tokens=10):
    """Greedy decoding to project system state forward."""
    curr_embeds = embeds
    text = ""
    for _ in range(max_tokens):
        out = model(inputs_embeds=curr_embeds)
        next_id = torch.argmax(out.logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
        text += tokenizer.decode(next_id[0])
        next_emb = model.transformer.wte(next_id)
        curr_embeds = torch.cat([curr_embeds, next_emb], dim=1)
    return text
print("\n" + "="*55)
print("             LIVE SYSTEM DEMONSTRATION")
print("="*55)
print("\n[1] Open-Loop Trajectory (Uncontrolled Plant):")
raw_output = simulate_trajectory(prompt_embeds)
print(f"Input: '{prompt_text}'\nState: ->{raw_output}")
print("\n[2] Closed-Loop Trajectory (CBF Controlled):")
optimized_embeds = torch.cat([prefix_tensor.detach(), prompt_embeds], dim=1)
safe_output = simulate_trajectory(optimized_embeds)
print(f"Input: [PREFIX] + '{prompt_text}'\nState: ->{safe_output}")
print("="*55)
print("\nGenerating phase plots for presentation...")
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history_tox, color='red', linewidth=2)
plt.axhline(y=safety_threshold, color='green', linestyle='--', label=r'Threshold ($\gamma$)')
plt.title('Toxicity Probability vs Epochs')
plt.ylabel('P(Unsafe Words)')
plt.xlabel('Optimization Steps')
plt.grid(True, alpha=0.3)
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history_h_x, color='blue', linewidth=2)
plt.axhline(y=0, color='black', linestyle='-')
plt.fill_between(range(len(history_h_x)), 0, max(history_h_x + [0.001]), color='green', alpha=0.1, label='Safe Set')
plt.fill_between(range(len(history_h_x)), min(history_h_x + [-0.001]), 0, color='red', alpha=0.1, label='Unsafe Region')
plt.title(r'Barrier Function $h(x) = \gamma - tox(x)$')
plt.ylabel('h(x)')
plt.xlabel('Optimization Steps')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('cbf_prototype.png')
plt.show()
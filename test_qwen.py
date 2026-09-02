from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 1. Select the model
model_name = "Qwen/Qwen2.5-0.5B"

# 2. Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3. Load model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name)

# 4. Create some test text
text = "Hello, how are you?"

# 5. Convert text into tokens
inputs = tokenizer(text, return_tensors="pt")

print("Input:", inputs)

# 6. Run the model
with torch.no_grad():
    outputs = model(**inputs)

# 7. Get logits
logits = outputs.logits

# 8. Check logits shape
print("Logits shape:", logits.shape)
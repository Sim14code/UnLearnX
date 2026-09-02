from kl_loss import compute_kl_loss

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load original model
original_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

# Load new model
new_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

# Create input
text = "What is machine learning?"

inputs = tokenizer(
    text,
    return_tensors="pt"
)

# Get logits from original model
outputs_original = original_model(**inputs)

# Get logits from new model
outputs_new = new_model(**inputs)

# Calculate KL loss
kl = compute_kl_loss(
    outputs_original.logits,
    outputs_new.logits
)

print("KL Loss:", kl.item())
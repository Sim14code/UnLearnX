import json
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# --------------------------------------------------
# Load tokenizer and model
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

import torch

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16
)
# --------------------------------------------------
# Load ALP-selected layers
# --------------------------------------------------

ALP_FILE = (
    Path(__file__).resolve().parent
    / "TOFU"
    / "extra"
    / "alp_selected_layers.json"
)

if not ALP_FILE.exists():
    raise FileNotFoundError(
        f"ALP result file not found: {ALP_FILE}\n"
        "Run select_alp_layers.py first."
    )


with open(ALP_FILE, "r", encoding="utf-8") as f:
    alp_data = json.load(f)


selected_layers = alp_data["selected_layers"]

print("=" * 50)
print("ALP SELECTED LAYERS")
print("=" * 50)
print(selected_layers)


# --------------------------------------------------
# Build LoRA target modules
# --------------------------------------------------

target_modules = []

for layer in selected_layers:

    target_modules.extend([
        f"model.layers.{layer}.self_attn.q_proj",
        f"model.layers.{layer}.self_attn.k_proj",
        f"model.layers.{layer}.self_attn.v_proj",
        f"model.layers.{layer}.self_attn.o_proj",
    ])


print("\nALP LoRA target modules:")
for module in target_modules:
    print(" ", module)


# --------------------------------------------------
# LoRA configuration
# --------------------------------------------------

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",

    # IMPORTANT:
    # LoRA is now applied ONLY to ALP-selected layers
    target_modules=target_modules
)


# --------------------------------------------------
# Attach LoRA
# --------------------------------------------------

model = get_peft_model(
    model,
    lora_config
)


# --------------------------------------------------
# Print trainable parameters
# --------------------------------------------------

model.print_trainable_parameters()
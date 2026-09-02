import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_ADAPTER = "adapters/unlearned_adapter"

parser = argparse.ArgumentParser(description="Test inference on base model or unlearned model adapter.")
parser.add_argument("--prompt", type=str, default="Who is Jaime Vasquez?", help="Prompt question for model generation.")
parser.add_argument("--adapter_path", type=str, default=DEFAULT_ADAPTER, help="Path to LoRA adapter.")
args = parser.parse_args()

print(f"Loading tokenizer and base model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

adapter_path = Path(args.adapter_path)
if adapter_path.exists():
    print(f"Attaching unlearned LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
else:
    print("No adapter found. Running inference on base model.")

model.eval()

messages = [
    {
        "role": "user",
        "content": args.prompt
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(
    text,
    return_tensors="pt"
)

outputs = model.generate(
    **inputs,
    max_new_tokens=100
)

response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
)

print("\n--- MODEL RESPONSE ---")
print(response)
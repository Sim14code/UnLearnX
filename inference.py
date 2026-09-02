from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

messages = [
    {
        "role": "user",
        "content": "What is machine unlearning?"
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
    outputs[0],
    skip_special_tokens=True
)

print(response)
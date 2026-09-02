import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from hooks import ActivationStore, make_hook


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ACTIVATIONS_DIR = ROOT / "activations"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def load_model_and_tokenizer(model_name=DEFAULT_MODEL, adapter_path=None):
    """
    Loads base model and tokenizer, optionally attaching a LoRA adapter.
    """
    print(f"Loading base tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if adapter_path and Path(adapter_path).exists():
        print(f"Attaching LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return tokenizer, model


def register_layer_hooks(model, target_layer_names=None):
    """
    Registers forward hooks on target model layers.
    If target_layer_names is None, defaults to key self-attention modules across layers.
    """
    store = ActivationStore()
    handles = []

    # Find and hook target linear projections in transformer blocks
    for name, module in model.named_modules():
        if target_layer_names:
            if any(target in name for target in target_layer_names):
                handle = module.register_forward_hook(make_hook(store, name))
                handles.append(handle)
        else:
            # Default target modules matching LoRA targets: q_proj, k_proj, v_proj, o_proj
            if any(name.endswith(proj) for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]):
                handle = module.register_forward_hook(make_hook(store, name))
                handles.append(handle)

    print(f"Registered forward hooks on {len(handles)} layer modules.")
    return store, handles


def extract_dataset_activations(model, tokenizer, store, dataset_path, max_examples=None):
    """
    Runs forward passes on dataset examples and collects layer activations.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if max_examples:
        data = data[:max_examples]

    dataset_activations = []

    print(f"Processing {len(data)} examples from {dataset_path.name}...")
    for idx, example in enumerate(data):
        store.clear()

        question = example.get("question", example.get("instruction", ""))
        answer = example.get("answer", example.get("response", ""))
        text = f"{question} {answer}"

        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            _ = model(**inputs)

        # Clone current activation snapshot for this sample
        sample_activations = {
            layer_name: tensor.clone()
            for layer_name, tensor in store.activations.items()
        }
        dataset_activations.append({
            "index": idx,
            "text": text,
            "activations": sample_activations
        })

    return dataset_activations


def main():
    parser = argparse.ArgumentParser(
        description="Extract layer activations for TOFU forget and retain datasets."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL,
        help="Base model checkpoint name or path."
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Optional path to trained PEFT LoRA adapter."
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Limit number of dataset examples to process."
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="",
        help="Optional prefix for output saved files (e.g. 'unlearned_' or 'base_')."
    )

    args = parser.parse_args()

    ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load model & tokenizer
    tokenizer, model = load_model_and_tokenizer(
        model_name=args.model_name,
        adapter_path=args.adapter_path
    )

    # 2. Register hooks
    store, handles = register_layer_hooks(model)

    # 3. Extract activations for forget set
    forget_path = DATA_DIR / "forget.json"
    if forget_path.exists():
        forget_activations = extract_dataset_activations(
            model=model,
            tokenizer=tokenizer,
            store=store,
            dataset_path=forget_path,
            max_examples=args.max_examples
        )
        output_file = ACTIVATIONS_DIR / f"{args.output_prefix}forget_activations.pt"
        torch.save(forget_activations, output_file)
        print(f"Saved forget activations -> {output_file}")
    else:
        print(f"Warning: {forget_path} not found.")

    # 4. Extract activations for retain set
    retain_path = DATA_DIR / "retain.json"
    if retain_path.exists():
        retain_activations = extract_dataset_activations(
            model=model,
            tokenizer=tokenizer,
            store=store,
            dataset_path=retain_path,
            max_examples=args.max_examples
        )
        output_file = ACTIVATIONS_DIR / f"{args.output_prefix}retain_activations.pt"
        torch.save(retain_activations, output_file)
        print(f"Saved retain activations -> {output_file}")
    else:
        print(f"Warning: {retain_path} not found.")

    # 5. Remove hook handles
    for handle in handles:
        handle.remove()

    print("Activation extraction completed successfully!")


if __name__ == "__main__":
    main()

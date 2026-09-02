import argparse
import json
import math
from pathlib import Path
import torch
import difflib
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_ADAPTER = "adapters/unlearned_adapter"


def compute_string_similarity(pred_str, ref_str):
    """
    Computes sequence matcher similarity ratio between predicted and reference text.
    """
    return difflib.SequenceMatcher(None, pred_str.strip().lower(), ref_str.strip().lower()).ratio()


def evaluate_dataset_performance(model, tokenizer, dataset_path, max_examples=None, max_new_tokens=50):
    """
    Evaluates loss, perplexity, and answer generation quality for a dataset.
    Question tokens are masked (-100) so loss is computed strictly on response tokens.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if max_examples:
        data = data[:max_examples]

    total_loss = 0.0
    total_tokens = 0
    similarities = []
    eval_details = []

    model.eval()

    for idx, example in enumerate(data):
        question = example.get("question", example.get("instruction", ""))
        ref_answer = example.get("answer", example.get("response", ""))

        prompt_text = f"Question: {question}\nAnswer:"
        full_text = f"Question: {question}\nAnswer: {ref_answer}"

        prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
        full_inputs = tokenizer(full_text, return_tensors="pt")

        input_ids = full_inputs["input_ids"].to(model.device)
        attention_mask = full_inputs["attention_mask"].to(model.device)

        labels = input_ids.clone()
        prompt_len = prompt_ids.shape[1]
        labels[0, :prompt_len] = -100  # Mask question prompt

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss.item()
            num_response_tokens = (labels != -100).sum().item()

            # Generation evaluation
            gen_inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            gen_outputs = model.generate(
                **gen_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

            # Strip prompt from generated response
            gen_text = tokenizer.decode(
                gen_outputs[0][gen_inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()

            sim_score = compute_string_similarity(gen_text, ref_answer)
            similarities.append(sim_score)

            total_loss += loss * num_response_tokens
            total_tokens += num_response_tokens

            eval_details.append({
                "index": idx,
                "question": question,
                "ref_answer": ref_answer,
                "generated_answer": gen_text,
                "loss": loss,
                "perplexity": math.exp(loss) if loss < 20 else float("inf"),
                "similarity_score": sim_score
            })

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    avg_similarity = sum(similarities) / max(len(similarities), 1)

    return {
        "avg_loss": avg_loss,
        "perplexity": perplexity,
        "avg_similarity": avg_similarity,
        "details": eval_details
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Base Model vs Unlearned Model on TOFU benchmark datasets."
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
        default=DEFAULT_ADAPTER,
        help="Path to trained PEFT LoRA adapter checkpoint."
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Limit number of evaluation examples for quick testing."
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="tofu_eval_results.json",
        help="Output evaluation results JSON filename."
    )

    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    forget_path = DATA_DIR / "forget.json"
    retain_path = DATA_DIR / "retain.json"

    if not forget_path.exists() or not retain_path.exists():
        print(f"Error: TOFU dataset files not found in {DATA_DIR}.")
        return

    # ----------------------------------------------------
    # 1. EVALUATE BASE MODEL
    # ----------------------------------------------------
    print("=" * 60)
    print("1. EVALUATING BASE MODEL:", args.model_name)
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name)

    print("\n--- Base Model: Forget Dataset ---")
    base_forget_metrics = evaluate_dataset_performance(
        base_model, tokenizer, forget_path, max_examples=args.max_examples
    )
    print(f"Loss: {base_forget_metrics['avg_loss']:.4f} | Perplexity: {base_forget_metrics['perplexity']:.4f} | Similarity: {base_forget_metrics['avg_similarity']:.4f}")

    print("\n--- Base Model: Retain Dataset ---")
    base_retain_metrics = evaluate_dataset_performance(
        base_model, tokenizer, retain_path, max_examples=args.max_examples
    )
    print(f"Loss: {base_retain_metrics['avg_loss']:.4f} | Perplexity: {base_retain_metrics['perplexity']:.4f} | Similarity: {base_retain_metrics['avg_similarity']:.4f}")

    # ----------------------------------------------------
    # 2. EVALUATE UNLEARNED MODEL (IF ADAPTER EXISTS)
    # ----------------------------------------------------
    unlearned_forget_metrics = None
    unlearned_retain_metrics = None
    adapter_file_path = Path(args.adapter_path)

    if adapter_file_path.exists():
        print("\n" + "=" * 60)
        print("2. EVALUATING UNLEARNED MODEL (Adapter:", args.adapter_path, ")")
        print("=" * 60)

        unlearned_model = PeftModel.from_pretrained(base_model, args.adapter_path)

        print("\n--- Unlearned Model: Forget Dataset ---")
        unlearned_forget_metrics = evaluate_dataset_performance(
            unlearned_model, tokenizer, forget_path, max_examples=args.max_examples
        )
        print(f"Loss: {unlearned_forget_metrics['avg_loss']:.4f} | Perplexity: {unlearned_forget_metrics['perplexity']:.4f} | Similarity: {unlearned_forget_metrics['avg_similarity']:.4f}")

        print("\n--- Unlearned Model: Retain Dataset ---")
        unlearned_retain_metrics = evaluate_dataset_performance(
            unlearned_model, tokenizer, retain_path, max_examples=args.max_examples
        )
        print(f"Loss: {unlearned_retain_metrics['avg_loss']:.4f} | Perplexity: {unlearned_retain_metrics['perplexity']:.4f} | Similarity: {unlearned_retain_metrics['avg_similarity']:.4f}")
    else:
        print(f"\nNote: Adapter checkpoint '{args.adapter_path}' not found. Run trainer.py first to generate unlearned weights.")

    # ----------------------------------------------------
    # 3. PRINT COMPARISON SUMMARY & SAVE RESULTS
    # ----------------------------------------------------
    print("\n" + "=" * 70)
    print("TOFU UNLEARNING EVALUATION SUMMARY REPORT")
    print("=" * 70)
    print(f"{'Metric':<25} | {'Base Model':<18} | {'Unlearned Model':<18} | {'Delta':<12}")
    print("-" * 75)

    if unlearned_forget_metrics and unlearned_retain_metrics:
        f_loss_delta = unlearned_forget_metrics['avg_loss'] - base_forget_metrics['avg_loss']
        f_ppl_delta = unlearned_forget_metrics['perplexity'] - base_forget_metrics['perplexity']
        f_sim_delta = unlearned_forget_metrics['avg_similarity'] - base_forget_metrics['avg_similarity']

        r_loss_delta = unlearned_retain_metrics['avg_loss'] - base_retain_metrics['avg_loss']
        r_ppl_delta = unlearned_retain_metrics['perplexity'] - base_retain_metrics['perplexity']
        r_sim_delta = unlearned_retain_metrics['avg_similarity'] - base_retain_metrics['avg_similarity']

        print(f"{'Forget Loss (↑ target)':<25} | {base_forget_metrics['avg_loss']:<18.4f} | {unlearned_forget_metrics['avg_loss']:<18.4f} | {f_loss_delta:<+12.4f}")
        print(f"{'Forget Perplexity (↑)':<25} | {base_forget_metrics['perplexity']:<18.4f} | {unlearned_forget_metrics['perplexity']:<18.4f} | {f_ppl_delta:<+12.4f}")
        print(f"{'Forget Similarity (↓)':<25} | {base_forget_metrics['avg_similarity']:<18.4f} | {unlearned_forget_metrics['avg_similarity']:<18.4f} | {f_sim_delta:<+12.4f}")
        print("-" * 75)
        print(f"{'Retain Loss (↓ target)':<25} | {base_retain_metrics['avg_loss']:<18.4f} | {unlearned_retain_metrics['avg_loss']:<18.4f} | {r_loss_delta:<+12.4f}")
        print(f"{'Retain Perplexity (↓)':<25} | {base_retain_metrics['perplexity']:<18.4f} | {unlearned_retain_metrics['perplexity']:<18.4f} | {r_ppl_delta:<+12.4f}")
        print(f"{'Retain Similarity (↑)':<25} | {base_retain_metrics['avg_similarity']:<18.4f} | {unlearned_retain_metrics['avg_similarity']:<18.4f} | {r_sim_delta:<+12.4f}")
    else:
        print(f"{'Forget Loss':<25} | {base_forget_metrics['avg_loss']:<18.4f} | {'N/A':<18} | {'N/A':<12}")
        print(f"{'Retain Loss':<25} | {base_retain_metrics['avg_loss']:<18.4f} | {'N/A':<18} | {'N/A':<12}")

    results_payload = {
        "base_model": args.model_name,
        "adapter_path": args.adapter_path if adapter_file_path.exists() else None,
        "base_metrics": {
            "forget": base_forget_metrics,
            "retain": base_retain_metrics
        },
        "unlearned_metrics": {
            "forget": unlearned_forget_metrics,
            "retain": unlearned_retain_metrics
        } if adapter_file_path.exists() else None
    }

    output_path = RESULTS_DIR / args.output_filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\nFull evaluation report saved -> {output_path}")


if __name__ == "__main__":
    main()

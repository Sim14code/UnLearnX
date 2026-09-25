# UnLearnX: Machine Unlearning via ALP & Gradient Projection with Qwen2.5

Welcome to **UnLearnX**, a research repository for precision machine unlearning on Large Language Models (LLMs) using the **TOFU (Task of Fictitious Unlearning)** benchmark and **Qwen2.5-0.5B-Instruct**.

---

## 1. Project Overview & Objective

Machine Unlearning aims to remove specific sensitive knowledge (the **Forget Set**) from a pre-trained LLM without retraining the model from scratch, while preserving general performance and unrelated knowledge (the **Retain Set**).

UnLearnX achieves targeted unlearning through a 4-stage pipeline:
1. **Activation Extraction via PyTorch Forward Hooks**: Intercepting hidden activations from transformer layers on forget vs. retain data.
2. **ALP (Activation Difference-based Layer Selection)**: Scoring and identifying sensitive transformer layers that exhibit the largest activation shifts between forget and retain samples.
3. **Targeted LoRA Adaptation**: Attaching LoRA parameter-efficient adapters **only** to ALP-selected sensitive layers.
4. **Gradient Projection & Regularized Optimization**: Negating forget gradients, projecting them orthogonally to retain gradients, and applying KL-divergence loss to prevent performance degradation on retain data.

---

## 2. Codebase Architecture & File Map

```
UnLearnX/
├── dataset pipeline
│   ├── TOFU/extra/prepare_tofu.py   # Downloads locuslab/TOFU and builds forget.json & retain.json
│   └── datasets/                     # Contains prepared forget.json (20 items) and retain.json (100 items)
│
├── activation extraction & ALP
│   ├── TOFU/extra/hooks.py              # PyTorch forward hook utilities (ActivationStore)
│   ├── TOFU/extra/extract_activations.py# Captures hidden layer activations for forget & retain sets
│   ├── TOFU/extra/select_alp_layers.py  # Computes ALP layer scores & saves top-K layer selections
│   └── TOFU/extra/alp_selected_layers.json # Selected sensitive layers JSON output
│
├── model & lora configuration
│   ├── model.py                      # Base Qwen2.5-0.5B-Instruct loader + generic LoRA config
│   └── lora.py                       # Loads ALP-selected layers & dynamically attaches LoRA to them
│
├── unlearning losses & optimization
│   ├── forget_loss.py                # Causal LM loss on forget samples
│   ├── kl_loss.py                    # KL divergence loss between reference model & unlearning model
│   ├── gradient_projection.py        # Orthogonal gradient projection (-grad_forget projected off grad_retain)
│   └── trainer.py                    # Main unlearning training loop with logit caching & adapter saving
│
├── evaluation & inference
│   ├── TOFU/extra/evaluate_tofu.py   # Benchmarks Base vs Unlearned model (Loss, PPL, Similarity)
│   ├── inference.py                  # Interactive text generation CLI
│   └── adapters/unlearned_adapter/   # Output PEFT LoRA adapter checkpoint
│
└── readme.md                         # Quick start command reference
```

---

## 3. End-to-End Execution Pipeline (GPU-Accelerated)

To run the complete unlearning and evaluation pipeline using your NVIDIA GPU (`cuda`), follow these steps in order:

### Environment Setup
Ensure your Python environment has PyTorch installed with CUDA support:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

### Step 1: Prepare TOFU Dataset
Downloads the TOFU dataset from Hugging Face and creates initial 20 forget / 100 retain subsets:
```bash
python TOFU/extra/prepare_tofu.py
```
*Output files:* `TOFU/extra/data/forget.json` and `TOFU/extra/data/retain.json`

---

### Step 2: Extract Layer Activations
Runs forward passes with forward hooks to extract hidden layer activations on forget and retain sets:
```bash
python TOFU/extra/extract_activations.py
```
*Output files:* `TOFU/extra/activations/forget_activations.pt` and `TOFU/extra/activations/retain_activations.pt`

---

### Step 3: Identify Sensitive Layers via ALP
Analyses activations, calculates absolute activation differences between forget and retain data per module, aggregates them per transformer layer, and selects top-K sensitive layers:
```bash
python TOFU/extra/select_alp_layers.py
```
*Output file:* `TOFU/extra/alp_selected_layers.json`

---

### Step 4: Run Targeted Unlearning Training
Trains the LoRA adapter on ALP-selected layers using gradient projection and pre-cached retain logits:

**GPU Speedup Note**: Ensure `trainer.py` uses GPU device allocation (`device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`).

```bash
python trainer.py
```
*Output folder:* `adapters/unlearned_adapter`

---

### Step 5: Evaluate & Benchmark (Base Model vs. Unlearned Model)
Measures unlearning efficacy and retention accuracy across both models:

```bash
# Full dataset evaluation
python TOFU/extra/evaluate_tofu.py

# Quick fast evaluation (e.g., 5 samples)
python TOFU/extra/evaluate_tofu.py --max_examples 5
```
*Output file:* `TOFU/extra/results/tofu_eval_results.json`

---

### Step 6: Test Interactive Generation
Interactively query your unlearned model:
```bash
python inference.py
```

---

## 4. How to Read & Interpret Evaluation Results

`evaluate_tofu.py` outputs a structured comparison table comparing **Base Model** vs. **Unlearned Model**:

```
======================================================================
TOFU UNLEARNING EVALUATION SUMMARY REPORT
======================================================================
Metric                    | Base Model         | Unlearned Model    | Delta       
-----------------------------------------------------------------------
Forget Loss (↑ target)    | 0.8521             | 4.2145             | +3.3624     
Forget Perplexity (↑)     | 2.3446             | 67.6598            | +65.3152    
Forget Similarity (↓)     | 0.9120             | 0.1240             | -0.7880     
-----------------------------------------------------------------------
Retain Loss (↓ target)    | 0.8120             | 0.8450             | +0.0330     
Retain Perplexity (↓)     | 2.2524             | 2.3280             | +0.0756     
Retain Similarity (↑)     | 0.8950             | 0.8810             | -0.0140     
```

### Key Metrics Breakdown:

| Metric | Target | Meaning | Ideal Outcome |
| :--- | :---: | :--- | :--- |
| **Forget Loss** | **$\uparrow$ Higher** | Cross-entropy loss on forget QA responses. | Model finds forget data unfamiliar/unlikely. |
| **Forget Perplexity (PPL)** | **$\uparrow$ Higher** | Uncertainty/confusion level when generating forget answers. | High perplexity = successful unlearning. |
| **Forget Similarity** | **$\downarrow$ Lower** | String similarity ratio between model output and forget target answer. | Near 0.0 = model no longer reveals forget facts. |
| **Retain Loss** | **$\downarrow$ Lower** | Cross-entropy loss on retain QA samples. | Stays close to base model (minimal increase). |
| **Retain Perplexity (PPL)**| **$\downarrow$ Lower** | Uncertainty level on retained knowledge. | Low perplexity = model retains knowledge intact. |
| **Retain Similarity** | **$\uparrow$ Higher** | Accuracy/similarity on retain QA responses. | Stays near base model (~0.85-0.95). |

---

## 5. Summary of GPU Acceleration Implementation

To maximize GPU utilization on your machine:
- **PyTorch Device**: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- **Precision**: `torch.bfloat16` or `torch.float16` for reduced VRAM footprint and accelerated tensor cores.
- **Logit Caching**: Reference model logits are pre-cached and reference model memory is freed with `gc.collect(); torch.cuda.empty_cache()` before starting training.

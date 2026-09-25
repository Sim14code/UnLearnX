import gc
import os
import time
import torch

from datasets import load_dataset

from forget_loss import compute_forget_loss
from gradient_projection import project_gradient
from kl_loss import compute_kl_loss

# Import the model + ALP-selected LoRA setup from lora.py
from lora import model, tokenizer


# ============================================================
# GPU / CPU CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Trainer executing on device: {DEVICE}")

if DEVICE.type == "cpu":
    num_cores = os.cpu_count() or 4
    torch.set_num_threads(num_cores)
    print(f"PyTorch configured to use {num_cores} CPU threads.")
else:
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")


# ============================================================
# HYPERPARAMETERS
# ============================================================

TORCH_DTYPE = torch.bfloat16
MAX_SEQ_LENGTH = 128

LEARNING_RATE = 1e-4
LAMBDA_KL = 1.0
EPOCHS = 3


# ============================================================
# TOKENIZER
# ============================================================

print("\nTokenizer loaded through lora.py.")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# DATASETS
# ============================================================

print("\nLoading datasets...")

forget_dataset = load_dataset(
    "json",
    data_files="TOFU/extra/data/forget.json"
)["train"]

retain_dataset = load_dataset(
    "json",
    data_files="TOFU/extra/data/retain.json"
)["train"]

print(f"Forget examples: {len(forget_dataset)}")
print(f"Retain examples: {len(retain_dataset)}")


# ============================================================
# REFERENCE MODEL
# Used only to cache retain logits
# ============================================================

print("\nLoading original reference model to pre-cache retain logits...")

from transformers import AutoModelForCausalLM

original_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=TORCH_DTYPE
).to(DEVICE)

original_model.eval()


# ============================================================
# CACHE RETAIN LOGITS
# ============================================================

cached_retain_inputs = []
cached_retain_logits = []

print("Pre-computing reference logits for retain dataset...")

start_cache_time = time.time()

with torch.no_grad():

    for example in retain_dataset:

        text = (
            example.get(
                "question",
                example.get("instruction", "")
            )
            + " "
            + example.get(
                "answer",
                example.get("response", "")
            )
        )

        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=MAX_SEQ_LENGTH,
            truncation=True
        )
        inputs_device = {k: v.to(DEVICE) for k, v in inputs.items()}

        outputs = original_model(**inputs_device)

        cached_retain_inputs.append(inputs)
        cached_retain_logits.append(outputs.logits.detach())


cache_duration = time.time() - start_cache_time

print(
    f"Cached {len(cached_retain_logits)} retain logits "
    f"in {cache_duration:.2f}s."
)


# ============================================================
# FREE REFERENCE MODEL
# ============================================================

print("Unloading original reference model to free memory...")

del original_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ============================================================
# TRAINABLE MODEL
# Already loaded + LoRA-attached by lora.py
# ============================================================


print("\nUsing model from lora.py.")

model.train()

print("\nLoRA parameter summary:")
model.print_trainable_parameters()


# ============================================================
# TRAINABLE PARAMETERS
# ============================================================

trainable_params = [
    param
    for param in model.parameters()
    if param.requires_grad
]

print(
    f"Number of trainable parameter tensors: "
    f"{len(trainable_params)}"
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    trainable_params,
    lr=LEARNING_RATE
)


# ============================================================
# PRETOKENIZE FORGET DATA
# ============================================================

pretokenized_forget = []

for example in forget_dataset:

    text = (
        example.get(
            "question",
            example.get("instruction", "")
        )
        + " "
        + example.get(
            "answer",
            example.get("response", "")
        )
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=MAX_SEQ_LENGTH,
        truncation=True
    )

    pretokenized_forget.append(inputs)


# ============================================================
# TRAINING
# ============================================================

print("\nStarting optimization training loop...")

total_steps = len(forget_dataset) * EPOCHS
step_count = 0

train_start_time = time.time()


for epoch in range(EPOCHS):

    print("\n========================================")
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("========================================")

    for i in range(len(forget_dataset)):

        step_start_time = time.time()
        step_count += 1


        # ----------------------------------------------------
        # FORGET LOSS
        # ----------------------------------------------------

        forget_inputs = pretokenized_forget[i]

        forget_input_ids = forget_inputs["input_ids"].to(DEVICE)
        forget_attention_mask = forget_inputs["attention_mask"].to(DEVICE)

        forget_labels = forget_input_ids.clone()

        forget_loss = compute_forget_loss(
            model,
            forget_input_ids,
            forget_attention_mask,
            forget_labels
        )


        # ----------------------------------------------------
        # FORGET GRADIENTS
        # ----------------------------------------------------

        forget_grads = torch.autograd.grad(
            forget_loss,
            trainable_params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True
        )

        forget_grads = [
            -grad if grad is not None else None
            for grad in forget_grads
        ]


        # ----------------------------------------------------
        # RETAIN EXAMPLE
        # ----------------------------------------------------

        retain_index = i % len(cached_retain_logits)

        retain_inputs = {k: v.to(DEVICE) for k, v in cached_retain_inputs[retain_index].items()}
        orig_logits = cached_retain_logits[retain_index].to(DEVICE)


        # ----------------------------------------------------
        # KL LOSS
        # ----------------------------------------------------

        new_outputs = model(**retain_inputs)

        kl_loss = compute_kl_loss(
            orig_logits,
            new_outputs.logits
        )



        # ----------------------------------------------------
        # RETAIN GRADIENTS
        # ----------------------------------------------------

        retain_grads = torch.autograd.grad(
            kl_loss,
            trainable_params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True
        )


        # ----------------------------------------------------
        # GRADIENT PROJECTION
        # ----------------------------------------------------

        projected_grads = []

        for f_grad, r_grad in zip(
            forget_grads,
            retain_grads
        ):

            if f_grad is None:

                projected_grads.append(None)

            elif r_grad is None:

                projected_grads.append(f_grad)

            else:

                projected_grads.append(
                    project_gradient(
                        f_grad,
                        r_grad
                    )
                )


        # ----------------------------------------------------
        # FINAL GRADIENT
        # ----------------------------------------------------

        final_grads = []

        for p_grad, r_grad in zip(
            projected_grads,
            retain_grads
        ):

            if p_grad is None:

                final_grads.append(r_grad)

            elif r_grad is None:

                final_grads.append(p_grad)

            else:

                final_grads.append(
                    p_grad + LAMBDA_KL * r_grad
                )


        # ----------------------------------------------------
        # OPTIMIZER STEP
        # ----------------------------------------------------

        optimizer.zero_grad()

        for param, grad in zip(
            trainable_params,
            final_grads
        ):

            if grad is not None:
                param.grad = grad

        optimizer.step()


        # ----------------------------------------------------
        # LOGGING
        # ----------------------------------------------------

        step_elapsed = time.time() - step_start_time

        print(
            f"Step {step_count}/{total_steps} "
            f"(Ex {i + 1}) | "
            f"Forget Loss: {forget_loss.item():.4f} | "
            f"KL Loss: {kl_loss.item():.4f} | "
            f"Time: {step_elapsed:.3f}s"
        )


# ============================================================
# TRAINING COMPLETE
# ============================================================

total_train_time = time.time() - train_start_time

print(
    f"\nTraining complete in "
    f"{total_train_time:.2f}s!"
)


# ============================================================
# SAVE ADAPTER
# ============================================================

print("Saving adapter...")

model.save_pretrained(
    "adapters/unlearned_adapter"
)

tokenizer.save_pretrained(
    "adapters/unlearned_adapter"
)

print(
    "Adapter saved successfully to "
    "adapters/unlearned_adapter!"
)

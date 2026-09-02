import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import (
    LoraConfig,
    get_peft_model
)

from datasets import load_dataset

from forget_loss import compute_forget_loss
from kl_loss import compute_kl_loss
from gradient_projection import project_gradient


# ============================================================
# 1. Configuration
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

LEARNING_RATE = 1e-4

LAMBDA_KL = 1.0

EPOCHS = 3


# ============================================================
# 2. Load tokenizer
# ============================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# 3. Load original model
# ============================================================

print("Loading original model...")

original_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

# Original model is only used as a reference.
# We DO NOT train it.

for param in original_model.parameters():
    param.requires_grad = False

original_model.eval()


# ============================================================
# 4. Load trainable model
# ============================================================

print("Loading trainable model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)


# ============================================================
# 5. Add LoRA
# ============================================================

print("Adding LoRA...")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ]
)

model = get_peft_model(
    model,
    lora_config
)

model.train()

model.print_trainable_parameters()


# ============================================================
# 6. Get only trainable LoRA parameters
# ============================================================

trainable_params = [
    param
    for param in model.parameters()
    if param.requires_grad
]

print(
    "Number of trainable parameter tensors:",
    len(trainable_params)
)


# ============================================================
# 7. Load datasets
# ============================================================

print("Loading datasets...")

forget_dataset = load_dataset(
    "json",
    data_files="datasets/forget.json"
)["train"]

retain_dataset = load_dataset(
    "json",
    data_files="datasets/retain.json"
)["train"]

print(
    "Forget examples:",
    len(forget_dataset)
)

print(
    "Retain examples:",
    len(retain_dataset)
)


# ============================================================
# 8. Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    trainable_params,
    lr=LEARNING_RATE
)


# ============================================================
# 9. Training loop
# ============================================================

for epoch in range(EPOCHS):

    print("\n")
    print("========================================")
    print("Epoch:", epoch + 1)
    print("========================================")

    for i in range(len(forget_dataset)):

        # ====================================================
        # FORGET DATA
        # ====================================================

        forget_example = forget_dataset[i]

        forget_text = (
            forget_example["instruction"]
            + " "
            + forget_example["response"]
        )

        forget_inputs = tokenizer(
            forget_text,
            return_tensors="pt"
        )

        forget_input_ids = forget_inputs["input_ids"]

        forget_attention_mask = forget_inputs[
            "attention_mask"
        ]

        forget_labels = forget_input_ids.clone()


        # ====================================================
        # RETAIN DATA
        # ====================================================

        retain_index = i % len(retain_dataset)

        retain_example = retain_dataset[
            retain_index
        ]

        retain_text = (
            retain_example["instruction"]
            + " "
            + retain_example["response"]
        )

        retain_inputs = tokenizer(
            retain_text,
            return_tensors="pt"
        )


        # ====================================================
        # 1. FORGET LOSS
        # ====================================================

        forget_loss = compute_forget_loss(
            model,
            forget_input_ids,
            forget_attention_mask,
            forget_labels
        )


        # ====================================================
        # Get forget gradient
        # ====================================================

        forget_grads = torch.autograd.grad(
            forget_loss,
            trainable_params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True
        )


        # ====================================================
        # IMPORTANT:
        #
        # We want to FORGET.
        #
        # Normal CE minimizes the loss.
        #
        # Therefore we reverse the gradient:
        #
        #     forget_direction = -gradient
        #
        # ====================================================

        forget_grads = [
            -grad if grad is not None
            else None
            for grad in forget_grads
        ]


        # ====================================================
        # 2. RETAIN KL LOSS
        # ====================================================

        with torch.no_grad():

            original_outputs = original_model(
                **retain_inputs
            )


        new_outputs = model(
            **retain_inputs
        )


        kl_loss = compute_kl_loss(
            original_outputs.logits,
            new_outputs.logits
        )


        # ====================================================
        # Get retain gradient
        # ====================================================

        retain_grads = torch.autograd.grad(
            kl_loss,
            trainable_params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True
        )


        # ====================================================
        # 3. ORTHOGONAL GRADIENT PROJECTION
        # ====================================================

        projected_grads = []

        for forget_grad, retain_grad in zip(
            forget_grads,
            retain_grads
        ):

            if forget_grad is None:

                projected_grads.append(
                    None
                )

                continue


            if retain_grad is None:

                projected_grads.append(
                    forget_grad
                )

                continue


            projected = project_gradient(
                forget_grad,
                retain_grad
            )

            projected_grads.append(
                projected
            )


        # ====================================================
        # 4. COMBINE PROJECTED FORGET GRADIENT
        #    + RETAIN GRADIENT
        # ====================================================

        final_grads = []

        for projected_grad, retain_grad in zip(
            projected_grads,
            retain_grads
        ):

            if projected_grad is None:

                final_grads.append(
                    retain_grad
                )

                continue


            if retain_grad is None:

                final_grads.append(
                    projected_grad
                )

                continue


            final_grad = (
                projected_grad
                + LAMBDA_KL * retain_grad
            )

            final_grads.append(
                final_grad
            )


        # ====================================================
        # 5. Put gradients into model
        # ====================================================

        optimizer.zero_grad()

        for param, grad in zip(
            trainable_params,
            final_grads
        ):

            if grad is not None:

                param.grad = grad


        # ====================================================
        # 6. Update LoRA
        # ====================================================

        optimizer.step()


        # ====================================================
        # 7. Print information
        # ====================================================

        print(
            f"Example {i + 1} | "
            f"Forget Loss: {forget_loss.item():.4f} | "
            f"KL Loss: {kl_loss.item():.4f}"
        )


# ============================================================
# 10. Save adapter
# ============================================================

print("\nSaving adapter...")

model.save_pretrained(
    "adapters/unlearned_adapter"
)

tokenizer.save_pretrained(
    "adapters/unlearned_adapter"
)

print(
    "\nAdapter saved successfully!"
)
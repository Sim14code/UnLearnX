import torch


def compute_forget_loss(
    model,
    input_ids,
    attention_mask,
    labels
):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

    forget_loss = outputs.loss

    return forget_loss
import torch


def project_gradient(
    forget_grad,
    retain_grad
):
    dot_product = torch.sum(
        forget_grad * retain_grad
    )

    retain_norm = torch.sum(
        retain_grad * retain_grad
    )

    projection = (
        dot_product /
        (retain_norm + 1e-8)
    ) * retain_grad

    projected_grad = (
        forget_grad - projection
    )

    return projected_grad
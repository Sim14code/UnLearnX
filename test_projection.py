import torch

from gradient_projection import project_gradient


forget_grad = torch.randn(10)
retain_grad = torch.randn(10)

projected = project_gradient(
    forget_grad,
    retain_grad
)

print("Forget gradient:")
print(forget_grad)

print("\nRetain gradient:")
print(retain_grad)

print("\nProjected gradient:")
print(projected)
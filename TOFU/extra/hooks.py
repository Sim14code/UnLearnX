import torch


class ActivationStore:
    """
    Stores activations captured from selected model layers.
    """

    def __init__(self):
        self.activations = {}

    def clear(self):
        self.activations.clear()

    def save(self, layer_name, activation):
        # Detach from computation graph and move to CPU.
        self.activations[layer_name] = (
            activation.detach().float().cpu()
        )


def get_activation_tensor(output):
    """
    Transformer layers can return either:
      - a Tensor
      - a tuple where the first item is the hidden state
    """

    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, tuple):
        return output[0]

    raise TypeError(
        f"Unsupported layer output type: {type(output)}"
    )


def make_hook(store, layer_name):
    def hook(module, inputs, output):
        activation = get_activation_tensor(output)

        store.save(
            layer_name,
            activation
        )

    return hook
import torch
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ACTIVATIONS_DIR = ROOT / "activations"
OUTPUT_FILE = ROOT / "alp_selected_layers.json"


def load_activations(filename):
    path = ACTIVATIONS_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Activation file not found: {path}")

    print(f"Loading {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def pool_activation(tensor):
    """
    Convert an activation tensor into one vector per example.

    Typical transformer activation:
        [sequence_length, hidden_size]

    We average over sequence positions.
    """

    tensor = tensor.float()

    if tensor.dim() == 1:
        return tensor

    if tensor.dim() == 2:
        return tensor.mean(dim=0)

    if tensor.dim() == 3:
        # [batch, sequence, hidden]
        return tensor.mean(dim=(0, 1))

    # Generic fallback:
    # keep final dimension and average everything else
    return tensor.mean(
        dim=tuple(range(tensor.dim() - 1))
    )


def calculate_layer_scores(forget_data, retain_data):
    """
    Calculate activation difference between forget and retain data
    for every hooked module.
    """

    scores = {}

    # Find modules that exist in both datasets
    common_layers = set()

    for example in forget_data:
        common_layers.update(example["activations"].keys())

    retain_layers = set()

    for example in retain_data:
        retain_layers.update(example["activations"].keys())

    common_layers = sorted(common_layers.intersection(retain_layers))

    print(f"Common hooked modules: {len(common_layers)}")

    for layer_name in common_layers:

        forget_vectors = []
        retain_vectors = []

        # -------------------------
        # Forget activations
        # -------------------------
        for example in forget_data:
            activation = example["activations"].get(layer_name)

            if activation is not None:
                vector = pool_activation(activation)
                forget_vectors.append(vector)

        # -------------------------
        # Retain activations
        # -------------------------
        for example in retain_data:
            activation = example["activations"].get(layer_name)

            if activation is not None:
                vector = pool_activation(activation)
                retain_vectors.append(vector)

        if not forget_vectors or not retain_vectors:
            continue

        forget_matrix = torch.stack(forget_vectors)
        retain_matrix = torch.stack(retain_vectors)

        # Mean activation for each dimension
        forget_mean = forget_matrix.mean(dim=0)
        retain_mean = retain_matrix.mean(dim=0)

        # Mean absolute activation difference
        difference = torch.abs(
            forget_mean - retain_mean
        ).mean()

        # Normalize so modules with naturally larger
        # activation magnitudes don't dominate.
        baseline = retain_mean.abs().mean() + 1e-8

        normalized_score = difference / baseline

        scores[layer_name] = normalized_score.item()

    return scores


def aggregate_to_transformer_layers(module_scores):
    """
    Convert projection-level scores into layer-level scores.

    Example:

        model.layers.5.self_attn.q_proj
        model.layers.5.self_attn.k_proj
        model.layers.5.self_attn.v_proj
        model.layers.5.self_attn.o_proj

    become:

        layer 5 -> average score
    """

    layer_scores = {}

    for module_name, score in module_scores.items():

        parts = module_name.split(".")

        if "layers" not in parts:
            continue

        try:
            index = parts.index("layers")
            layer_number = int(parts[index + 1])
        except (ValueError, IndexError):
            continue

        if layer_number not in layer_scores:
            layer_scores[layer_number] = []

        layer_scores[layer_number].append(score)

    # Average q/k/v/o scores for each transformer layer
    aggregated = {}

    for layer_number, scores in layer_scores.items():
        aggregated[layer_number] = sum(scores) / len(scores)

    return aggregated


def select_top_layers(layer_scores, top_k=6):

    ranked = sorted(
        layer_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    selected = ranked[:top_k]

    return ranked, selected


def main():

    print("=" * 60)
    print("ALP ACTIVATION-BASED LAYER SELECTION")
    print("=" * 60)

    # --------------------------------
    # 1. Load saved activations
    # --------------------------------
    forget_data = load_activations(
        "forget_activations.pt"
    )

    retain_data = load_activations(
        "retain_activations.pt"
    )

    print(f"Forget examples: {len(forget_data)}")
    print(f"Retain examples: {len(retain_data)}")

    # --------------------------------
    # 2. Calculate module scores
    # --------------------------------
    module_scores = calculate_layer_scores(
        forget_data,
        retain_data
    )

    print(f"\nScored modules: {len(module_scores)}")

    # --------------------------------
    # 3. Convert projection scores
    #    into transformer-layer scores
    # --------------------------------
    layer_scores = aggregate_to_transformer_layers(
        module_scores
    )

    # --------------------------------
    # 4. Select top K layers
    # --------------------------------
    TOP_K = 6

    ranked, selected = select_top_layers(
        layer_scores,
        top_k=TOP_K
    )

    print("\nLayer ranking:")
    print("-" * 40)

    for rank, (layer, score) in enumerate(ranked, start=1):
        marker = " <-- SELECTED" if layer in dict(selected) else ""

        print(
            f"{rank:2d}. Layer {layer:2d} "
            f"| Score: {score:.6f}{marker}"
        )

    selected_layers = [layer for layer, _ in selected]

    print("\nSelected ALP layers:")
    print(selected_layers)

    # --------------------------------
    # 5. Save results
    # --------------------------------
    result = {
        "method": "activation_difference",
        "top_k": TOP_K,
        "selected_layers": selected_layers,
        "layer_scores": {
            str(layer): score
            for layer, score in layer_scores.items()
        },
        "module_scores": module_scores
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved ALP results -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
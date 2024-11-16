import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from torch import Tensor, clamp, mean, randn
from torch.nn import Module
from torch.optim import Adam

from helpers.cnn import ConvolutionalNeuralNetwork


def setup_hooks(model: ConvolutionalNeuralNetwork, activations: dict[str, Tensor]) -> dict[str, Tensor]:
    # Hook function to capture the outputs
    def on_activation(name):
        def hook(module, args, output):
            activations[name] = output.detach()

        return hook

    # Register hooks on CNN and Classifier
    for i in range(9):
        model.cnn[i].register_forward_hook(on_activation(f"cnn{i}"))
    model.classifier[0].register_forward_hook(on_activation("linear"))
    model.classifier[1].register_forward_hook(on_activation("softmax"))

    return activations


def plot_conv_activations(activations: dict[str, Tensor], layer: str) -> None:
    activation = activations[layer]

    # Number of filters in conv1
    print(f"Shape of activation: {activation.shape}")
    num_filters = activation.shape[1]

    # Plot the activation maps
    fig, axes = plt.subplots(num_filters // 8, 8, figsize=(15, num_filters * 2 // 8))
    axes: Axes

    fig.suptitle(f"Filters of activation layer {layer}", fontsize=16)
    for idx in range(num_filters):
        row = idx // 8
        col = idx % 8
        axes[row, col].imshow(activation[0, idx].cpu(), cmap="gray")
        axes[row, col].axis("off")
        axes[row, col].set_title(f"Filter {idx}")

    plt.show()


def plot_fc_activations(activations: dict[str, Tensor], layer: str):
    # Get activations from the first fully connected layer
    activation = activations[layer]

    print(f"Shape of fc1 activations: {activation.shape}")

    # Plot the activations as a bar graph
    plt.figure(figsize=(12, 6))
    plt.bar(range(activation.shape[1]), activation[0].cpu().numpy())
    plt.title(f"Activations of {layer} Layer")
    plt.xlabel("Neuron Index")
    plt.ylabel("Activation")
    plt.show()


def activation_maximization(
    model: ConvolutionalNeuralNetwork,
    layer: Module,
    layer_name: str,
    filter_index: int,
    input_size: tuple[int, int, int, int] = (1, 1, 28, 28),
    lr: float = 0.1,
    iterations: int = 30,
):
    input_image = randn(input_size, requires_grad=True)
    # use Adam optimizer instead of SGD (better convergence)
    optimizer = Adam([input_image], lr=lr, weight_decay=1e-6)
    activations = {}

    def hook_function(module, input, output):
        activations[layer_name] = output

    # Register hook to capture activations on given layer
    hook = layer.register_forward_hook(hook_function)

    for i in range(iterations):
        # Zero gradients
        optimizer.zero_grad()

        # Utilize model to activate hook on layer
        _ = model(input_image)

        # Maximize activation
        act = activations[layer_name][0, filter_index]
        loss = -mean(act)
        loss.backward()
        optimizer.step()

        # Clamp the input image to be between 0 and 1 (as we normalized the data to be between 0 and 1)
        input_image.data = clamp(input_image.data, 0, 1)

        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}/{iterations}, Loss: {-loss.item():.4f}")

    hook.remove()
    return input_image.detach()

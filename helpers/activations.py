import matplotlib.pyplot as plt
from torch import Tensor, clamp, mean, randn
from torch.optim import Adam
from tqdm.notebook import trange

from helpers.cnn import ConvolutionalNeuralNetwork


def setup_hooks(model: ConvolutionalNeuralNetwork, activations: dict[str, Tensor]) -> dict[str, Tensor]:
    """Sets up forward hooks on all CNN layers."""

    # Hook function to capture the outputs
    def on_activation(name):
        def hook(module, args, output):
            activations[name] = output.detach()

        return hook

    # Register hooks on CNN and Classifier
    for i in range(model.cnn.__len__()):
        model.cnn[i].register_forward_hook(on_activation(f"cnn{i}"))
    model.classifier[0].register_forward_hook(on_activation("linear"))

    return activations


def plot_conv_activations(activations: dict[str, Tensor], layer: str) -> None:
    """Plots activations of Convolutional layers."""

    activation = activations[layer]

    # Number of filters in conv1
    print(f"Shape of activation: {activation.shape}")
    num_filters = activation.shape[1]

    # Plot the activation maps
    fig, axes = plt.subplots(num_filters // 8, 8, figsize=(15, num_filters * 2 // 8))

    fig.suptitle(f"Filters of activation layer {layer}", fontsize=16)
    for idx in range(num_filters):
        row = idx // 8
        col = idx % 8
        if num_filters > 8:
            axes[row, col].imshow(activation[0, idx].cpu(), cmap="gray")
            axes[row, col].axis("off")
            axes[row, col].set_title(f"Filter {idx}")
        else:
            axes[col].imshow(activation[0, idx].cpu(), cmap="gray")
            axes[col].axis("off")
            axes[col].set_title(f"Filter {idx}")

    plt.tight_layout()
    plt.show()


def plot_fc_activations(activations: dict[str, Tensor], layer: str):
    """Plots activations of Fully Connected (Linear) layers."""

    # Get activations from the first fully connected layer
    activation = activations[layer]

    print(f"Shape of fc1 activations: {activation.shape}")

    # Plot the activations as a bar graph
    plt.figure(figsize=(4, 3))
    plt.bar(range(activation.shape[1]), activation[0].cpu().numpy())
    plt.title(f"Activations of {layer} Layer")
    plt.xlabel("Neuron Index")
    plt.ylabel("Activation")
    plt.show()


def filter_activation_maximization(
    cnn_layer_num: int,
    model: ConvolutionalNeuralNetwork,
    input_size: tuple[int, int, int, int] = (1, 1, 404, 303),
    lr: float = 0.1,
    iterations: int = 30,
):
    """Generates noise images and optimizes them to maximize activation of the supplied ReLU layer."""

    def maximize(filter_index: int):
        input_image = randn(input_size, requires_grad=True)
        # use Adam optimizer instead of SGD (better convergence)
        optimizer = Adam([input_image], lr=lr, weight_decay=1e-6)
        layer_activations = {}

        def hook_function(module, input, output):
            layer_activations[cnn_layer_num] = output

        # Register hook to capture activations
        hook = layer.register_forward_hook(hook_function)

        for _ in range(iterations):
            # Zero the parameter gradients after each iteration
            optimizer.zero_grad()
            # forward pass
            model(input_image)
            # Get activations from the layer (shape is batch_size, channels, height, width)
            act = layer_activations[cnn_layer_num][0, filter_index]
            # Define loss as the negative mean of the activation
            loss = -mean(act)
            # Backward pass
            loss.backward()
            # Update the input image
            optimizer.step()
            # Clamp the input image to be between 0 and 1 (as we normalized the data to be between 0 and 1)
            input_image.data = clamp(input_image.data, 0, 1)

        hook.remove()
        return input_image.detach()

    layer = model.cnn[cnn_layer_num]
    num_filters = model.cnn[cnn_layer_num - 1].out_channels

    # maximize each filter's activations
    maximized_images = []
    for filter_index in trange(num_filters, desc="Filter"):
        am_image = maximize(filter_index=filter_index)
        maximized_images.append(am_image)

    # Plot the activation maps
    fig, axes = plt.subplots(num_filters // 8, 8, figsize=(15, num_filters * 3 // 8))

    fig.suptitle(f"Filters of activation layer cnn{cnn_layer_num}", fontsize=16)
    for idx, am_image in enumerate(maximized_images):
        row = idx // 8
        col = idx % 8
        if num_filters > 8:
            axes[row, col].imshow(am_image.squeeze().cpu().numpy(), cmap="gray")
            axes[row, col].axis("off")
            axes[row, col].set_title(f"Filter {idx}")
        else:
            axes[col].imshow(am_image.squeeze().cpu().numpy(), cmap="gray")
            axes[col].axis("off")
            axes[col].set_title(f"Filter {idx}")

    plt.tight_layout()
    plt.show()

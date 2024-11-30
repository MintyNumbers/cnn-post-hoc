import matplotlib.pyplot as plt
from lime.lime_image import ImageExplanation, LimeImageExplainer
from numpy import ndarray
from skimage.color import gray2rgb
from torch import Tensor, device, from_numpy, no_grad
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from torchvision.transforms.functional import rgb_to_grayscale
from tqdm.notebook import tqdm

from helpers.cnn import ConvolutionalNeuralNetwork
from helpers.dataset import BarkVN50Dataset
from helpers.functions import generate_augmented_input_images, one_hot_to_numeric_label_batch


def lime_evaluate_cnn(
    model: ConvolutionalNeuralNetwork,
    test_dataset: BarkVN50Dataset,
    device: device,
    augmented: bool,
    test_image_index: int,
):
    def batch_predict(image_batch_numpy: ndarray) -> ndarray:
        image_batch_tensor: Tensor = rgb_to_grayscale(from_numpy(image_batch_numpy).reshape(-1, 3, 404, 303))
        logits_batch = model(image_batch_tensor)
        probs_batch_tensor = softmax(logits_batch, dim=1).detach().cpu().numpy()
        return probs_batch_tensor

    with no_grad():
        model = model.eval().to(device)

        test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
        for test_images, label_batch in tqdm(test_loader):
            if augmented:
                input_images: Tensor = generate_augmented_input_images(test_images[test_image_index])
                fig, axes = plt.subplots(1, 5, figsize=(25, 10))
            else:
                input_images: Tensor = test_images
                fig, axes = plt.subplots(2, 5, figsize=(25, 7))

            # Predict labels
            pred_labels = one_hot_to_numeric_label_batch(model(input_images))

            fig.suptitle(
                f"True label: {one_hot_to_numeric_label_batch(label_batch)[test_image_index].item()}", fontsize=16
            )
            for i, image in enumerate(input_images):
                explainer_image = gray2rgb(image.squeeze().detach().cpu().numpy())
                explainer = LimeImageExplainer()
                explanation: ImageExplanation = explainer.explain_instance(
                    image=explainer_image,
                    classifier_fn=batch_predict,
                    top_labels=10,  # documentation: "produce explanations for the K labels with highest prediction probabilities"
                    num_samples=1000,
                    random_seed=0,
                )
                segmented_image, _ = explanation.get_image_and_mask(
                    explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False
                )

                # Create a figure with two subplots
                col = i % 5
                if augmented:
                    axes[col].imshow(segmented_image)
                    axes[col].axis("off")
                    axes[col].set_title(f"Pred: {pred_labels[i].item()}")
                else:
                    row = 0 if i < 5 else 1
                    axes[row, col].imshow(segmented_image)
                    axes[row, col].axis("off")
                    axes[row, col].set_title(f"Pred: {pred_labels[i].item()}")

            plt.tight_layout()
            plt.show()

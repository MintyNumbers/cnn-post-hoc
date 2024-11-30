import matplotlib.pyplot as plt
from numpy import swapaxes, transpose
from shap import DeepExplainer
from shap.plots._image import image
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from helpers.cnn import ConvolutionalNeuralNetwork
from helpers.dataset import BarkVN50Dataset
from helpers.functions import generate_augmented_input_images, one_hot_to_numeric_label_batch


def shap_evaluate_cnn(
    model: ConvolutionalNeuralNetwork,
    train_dataset: BarkVN50Dataset,
    test_dataset: BarkVN50Dataset,
    test_image_index: int,
):
    def get_pred_label(input_images: Tensor) -> list[str]:
        test_outputs = model(input_images)
        pred_labels = one_hot_to_numeric_label_batch(test_outputs)
        labels: list[str] = []
        for i in range(pred_labels.__len__()):
            labels.append(f"Pred: {pred_labels[i]}")
        return labels

    # Get random 100 train_images and train an explainer with them
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    train_images, _ = next(iter(train_loader))
    explainer = DeepExplainer(model, train_images)

    # Calculate SHAP values for all test batches
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
    for test_images, targets in tqdm(test_loader):
        # Get 1 sample of each class and modify it to compare SHAP values
        input_images = generate_augmented_input_images(test_images[test_image_index])

        # Calculate and plot SHAP values for test image batch
        shap_values = explainer.shap_values(input_images)
        image(
            shap_values=list(transpose(shap_values, (4, 0, 2, 3, 1))),
            pixel_values=-swapaxes(swapaxes(input_images.cpu().numpy(), 1, -1), 1, 2),
            true_labels=get_pred_label(input_images),
            width=25,
            show=False,
        )
        plt.suptitle(f"True label: {one_hot_to_numeric_label_batch(targets)[0]}")
        plt.show()

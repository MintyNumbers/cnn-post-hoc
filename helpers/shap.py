import matplotlib.pyplot as plt
from cv2 import COLOR_BGR2GRAY, COLOR_GRAY2BGR, cvtColor, rectangle
from numpy import swapaxes, transpose
from shap import DeepExplainer
from shap.plots._image import image
from torch import Tensor, empty, from_numpy
from torch.utils.data import DataLoader
from torchvision.transforms.v2.functional import gaussian_noise, rotate
from tqdm.notebook import tqdm

from helpers.cnn import ConvolutionalNeuralNetwork
from helpers.dataset import BarkVN50Dataset
from helpers.functions import one_hot_to_numeric_label_batch


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

    def generate_input_images(original_image: Tensor) -> Tensor:
        input_images = empty(5, 1, 404, 303)

        # original image
        input_images[0] = original_image

        # image with noise
        input_images[1] = gaussian_noise(original_image)

        # rotated iamge
        input_images[2] = rotate(original_image, 30)

        # image with black square
        black_rectangle_image = cvtColor(original_image.permute(1, 2, 0).numpy(), COLOR_GRAY2BGR)
        rectangle(black_rectangle_image, pt1=(100, 100), pt2=(200, 200), color=(0, 0, 0), thickness=-1)
        input_images[3] = from_numpy(cvtColor(black_rectangle_image, COLOR_BGR2GRAY)).unsqueeze(0)

        # image with plaid square
        black_rectangle_image_2 = cvtColor(original_image.permute(1, 2, 0).numpy(), COLOR_GRAY2BGR)
        rectangle(black_rectangle_image_2, pt1=(100, 100), pt2=(300, 300), color=(0, 0, 0), thickness=-1)
        input_images[4] = from_numpy(cvtColor(black_rectangle_image_2, COLOR_BGR2GRAY)).unsqueeze(0)

        return input_images

    # Get random 100 train_images and train an explainer with them
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    train_images, _ = next(iter(train_loader))
    explainer = DeepExplainer(model, train_images)

    # Calculate SHAP values for all test batches
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
    for test_images, targets in tqdm(test_loader):
        # Get 1 sample of each class and modify it to compare SHAP values
        input_images = generate_input_images(test_images[test_image_index])

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

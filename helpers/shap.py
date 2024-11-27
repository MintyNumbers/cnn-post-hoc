import matplotlib.pyplot as plt
from cv2 import COLOR_BGR2GRAY, COLOR_GRAY2BGR, cvtColor, line, rectangle
from numpy import swapaxes, transpose, uint8, zeros
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
):
    def get_pred_label(input_images: Tensor) -> list[str]:
        test_outputs = model(input_images)
        pred_labels = one_hot_to_numeric_label_batch(test_outputs)
        labels: list[str] = []
        for i in range(pred_labels.__len__()):
            labels.append(f"Pred: {pred_labels[i]}")
        return labels

    def generate_input_images(image: Tensor) -> Tensor:
        input_images = empty(5, 1, 404, 303)

        # original image
        input_images[0] = image

        # image with noise
        input_images[1] = gaussian_noise(image)

        # rotated iamge
        input_images[2] = rotate(image, 30)

        # image with black square
        black_rectangle_image = cvtColor(image.permute(1, 2, 0).numpy(), COLOR_GRAY2BGR)
        rectangle(black_rectangle_image, (100, 100), (200, 200), (0, 0, 0), -1)
        input_images[3] = from_numpy(cvtColor(black_rectangle_image, COLOR_BGR2GRAY)).unsqueeze(0)

        # image with plaid square
        plaid_image = cvtColor(image.permute(1, 2, 0).numpy(), COLOR_GRAY2BGR)
        plaid_texture = zeros((100, 100, 3), uint8)
        for x in range(0, 100, 40):
            line(plaid_texture, pt1=(0, x), pt2=(100, x), color=(220, 180, 110), thickness=5)
            line(plaid_texture, pt1=(0, x + 7), pt2=(100, x + 7), color=(220, 180, 110), thickness=5)
            line(plaid_texture, pt1=(x, 0), pt2=(x, 100), color=(150, 120, 90), thickness=5)
            line(plaid_texture, pt1=(x + 9, 0), pt2=(x + 9, 100), color=(150, 120, 90), thickness=5)
        plaid_image[100:200, 100:200] = plaid_texture
        input_images[4] = from_numpy(cvtColor(plaid_image, COLOR_BGR2GRAY)).unsqueeze(0)

        return input_images

    # Get random 100 train_images and train an explainer with them
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    train_images, _ = next(iter(train_loader))
    explainer = DeepExplainer(model, train_images)

    # Calculate SHAP values for all test batches
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
    for test_images, targets in tqdm(test_loader):
        # Get 1 sample of each class and modify it to compare SHAP values
        input_images = generate_input_images(test_images[0])

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

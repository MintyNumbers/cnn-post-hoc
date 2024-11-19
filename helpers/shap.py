from numpy import swapaxes, transpose
from shap import DeepExplainer, image_plot
from torch import max, randperm
from torch.utils.data import DataLoader
from tqdm import tqdm

from helpers.cnn import ConvolutionalNeuralNetwork
from helpers.dataset import BarkVN50Dataset


def shap_evaluate_cnn(
    model: ConvolutionalNeuralNetwork,
    train_dataset: BarkVN50Dataset,
    test_dataset: BarkVN50Dataset,
):
    # get random 100 train_images and train an explainer with them
    train_images_indexes = randperm(train_dataset.images.shape[0])[:100]
    train_images = train_dataset.images[train_images_indexes]
    explainer = DeepExplainer(model, train_images)

    # calculate SHAP values for all test batches
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    for test_images, targets in tqdm(test_loader):
        # print label and prediction background
        test_outputs = model(test_images)
        print(f"true labels: {max(targets, 1)[1]}\npred labels: {max(test_outputs, 1)[1]}\npred values: {test_outputs}")

        # calculate and plot SHAP values for test image batch
        shap_values = explainer.shap_values(test_images)
        shap_numpy = list(transpose(shap_values, (4, 0, 2, 3, 1)))
        test_numpy = swapaxes(swapaxes(test_images.numpy(), 1, -1), 1, 2)
        image_plot(shap_numpy, -test_numpy)

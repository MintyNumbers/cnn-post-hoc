from torch.utils.data import DataLoader
from helpers.cnn import ConvolutionalNeuralNetwork
from torch.nn import CrossEntropyLoss
from torch import Tensor, no_grad
from helpers.functions import predict_label


def evaluate_cnn(
    test_dataloader: DataLoader,
    train_dataloader: DataLoader,
    model: ConvolutionalNeuralNetwork,
    criterion: CrossEntropyLoss,
):
    with no_grad():
        _evaluate_dataset(test_dataloader, model, criterion)
        _evaluate_dataset(train_dataloader, model, criterion)


def _evaluate_dataset(
    dataloader: DataLoader,
    model: ConvolutionalNeuralNetwork,
    criterion: CrossEntropyLoss,
):
    correct_count, all_count = 0, 0

    for images, true_labels in dataloader:
        # Forward pass
        outputs: Tensor = model(images)
        loss: Tensor = criterion(outputs, true_labels)

        # predict labels
        all_count += outputs.shape[0]
        correct_count += predict_label(outputs=outputs, targets=true_labels)

    print(
        f"Number Of Images Tested {all_count},\tLoss: {loss.item():.4f},\tAccuracy: {correct_count / all_count}"
    )

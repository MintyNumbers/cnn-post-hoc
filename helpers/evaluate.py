from torch.utils.data import DataLoader
from helpers.cnn import ConvolutionalNeuralNetwork
from torch.nn import CrossEntropyLoss
from torch import Tensor, max, no_grad


def evaluate_cnn(
    test_dataloader: DataLoader,
    train_dataloader: DataLoader,
    model: ConvolutionalNeuralNetwork,
    criterion: CrossEntropyLoss,
):
    with no_grad():
        evaluate_dataset(test_dataloader, model, criterion)
        evaluate_dataset(train_dataloader, model, criterion)


def evaluate_dataset(
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
        _, pred_labels = max(outputs, 1)
        _, true_labels = max(true_labels, 1)
        all_count += pred_labels.shape[0]
        if pred_labels == true_labels:
            correct_count += 1

    print(
        f"Number Of Images Tested {all_count}, Loss: {loss.item():.4f}, Accuracy ={correct_count / all_count}"
    )

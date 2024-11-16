import matplotlib.pyplot as plt
from seaborn import heatmap
from sklearn.metrics import confusion_matrix
from torch import Tensor, no_grad
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from helpers.cnn import ConvolutionalNeuralNetwork
from helpers.functions import count_correct_label_batch, one_hot_to_numeric_label_batch


def evaluate_cnn(
    test_dataloader: DataLoader,
    train_dataloader: DataLoader,
    model: ConvolutionalNeuralNetwork,
    criterion: CrossEntropyLoss,
):
    def evaluate_dataset(
        dataloader: DataLoader,
        model: ConvolutionalNeuralNetwork,
        criterion: CrossEntropyLoss,
        subplot_nr: int,
    ):
        correct_count, all_count = 0, 0
        y_true, y_pred = [], []

        for images, one_hot_labels in dataloader:
            # Forward pass
            outputs: Tensor = model(images)
            loss: Tensor = criterion(outputs, one_hot_labels)

            # predict labels
            all_count += outputs.shape[0]
            correct_count += count_correct_label_batch(outputs=outputs, targets=one_hot_labels)

            # confusion matrix
            true_labels = one_hot_to_numeric_label_batch(one_hot_labels)
            pred_labels = one_hot_to_numeric_label_batch(outputs)
            y_true.extend(true_labels.cpu().numpy())
            y_pred.extend(pred_labels.cpu().numpy())

        print(
            f"Number Of Images Tested {all_count},\tLoss: {loss.item():.4f},\tAccuracy: {(100 * correct_count / all_count):.4f}"
        )

        # Create and plot confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        f.add_subplot(121 if subplot_nr == 1 else 122)
        heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix: {"Test" if subplot_nr == 1 else "Train"}")
        plt.xlabel("Predicted")
        plt.ylabel("True")

    with no_grad():
        f = plt.figure(figsize=(12, 5))
        evaluate_dataset(test_dataloader, model, criterion, 1)
        evaluate_dataset(train_dataloader, model, criterion, 2)
        plt.tight_layout()
        plt.show()

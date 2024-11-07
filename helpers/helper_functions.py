from torch import Tensor, max


def predict_label(targets: Tensor, outputs: Tensor) -> int:
    _, pred_labels = max(outputs, 1)
    _, true_labels = max(targets, 1)

    correct_count = 0
    if pred_labels.shape[0] > 1:
        for i in range(pred_labels.shape[0]):
            if pred_labels[i] == true_labels[i]:
                correct_count += 1

    return correct_count

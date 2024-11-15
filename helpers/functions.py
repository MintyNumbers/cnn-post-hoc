from torch import Tensor, max


def count_correct_label_batch(targets: Tensor, outputs: Tensor) -> int:
    pred_labels = one_hot_to_numeric_label_batch(outputs)
    true_labels = one_hot_to_numeric_label_batch(targets)

    correct_count = 0
    if pred_labels.shape[0] > 1:
        for i in range(pred_labels.shape[0]):
            if pred_labels[i] == true_labels[i]:
                correct_count += 1
    else:
        if pred_labels == true_labels:
            correct_count += 1

    return correct_count


def one_hot_to_numeric_label_batch(one_hot_label: Tensor) -> Tensor:
    value, numeric_label = max(one_hot_label, 1)
    return numeric_label

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
    value, numeric_labels = max(one_hot_label, 1)
    return numeric_labels


def one_hot_to_string_label_batch(one_hot_label: Tensor) -> list[str]:
    value, numeric_labels = list(max(one_hot_label, 1))
    string_labels = []
    for label in numeric_labels:
        # fmt:off
        if   label.item() == 0: string_labels.append("Cocos nucifera")       # noqa: E701
        elif label.item() == 1: string_labels.append("Dipterocarpus alatus") # noqa: E701
        elif label.item() == 2: string_labels.append("Eucalyptus")           # noqa: E701
        elif label.item() == 3: string_labels.append("Ficus microcarpa")     # noqa: E701
        elif label.item() == 4: string_labels.append("Hevea brasiliensis")   # noqa: E701
        elif label.item() == 5: string_labels.append("Musa")                 # noqa: E701
        elif label.item() == 6: string_labels.append("Psidium guajava")      # noqa: E701
        elif label.item() == 7: string_labels.append("Syzygium nervosum")    # noqa: E701
        elif label.item() == 8: string_labels.append("Terminalia catappa")   # noqa: E701
        elif label.item() == 9: string_labels.append("Veitchia merrilli")    # noqa: E701
        else:                   string_labels.append("")                     # noqa: E701
        # fmt:on
    return string_labels

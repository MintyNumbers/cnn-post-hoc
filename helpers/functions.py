from cv2 import COLOR_BGR2GRAY, COLOR_GRAY2BGR, cvtColor, rectangle
from torch import Tensor, empty, from_numpy, max
from torchvision.transforms.v2.functional import gaussian_noise, rotate


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


def generate_augmented_input_images(original_image: Tensor) -> Tensor:
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

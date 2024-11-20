from glob import glob

from torch import Tensor, device, empty, float32, tensor
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms.functional import rgb_to_grayscale


class BarkVN50Dataset(Dataset):
    def __init__(self, train: bool, device: device):
        self.device = device

        if train:
            self.image_paths = glob("./data/BarkVN-50/Train/*/*")
        else:
            self.image_paths = glob("./data/BarkVN-50/Test/*/*")
        self.image_paths.sort()

        # loading images into memory as tensors and transforming them from RGB to grayscale
        self.images = empty(self.image_paths.__len__(), 1, 404, 303)
        self.labels = empty(self.image_paths.__len__(), 10)
        for i, path in enumerate(self.image_paths):
            # transforming images from RGB (3 channels) to grayscale (1 channel)
            self.images[i] = rgb_to_grayscale(decode_image(path))

            # labelling the images (one-hot encoding)
            # fmt:off
            tree_species = path.split(sep="/")[-2]
            if   tree_species == "Cocos nucifera":       j = 0  # noqa: E701
            elif tree_species == "Dipterocarpus alatus": j = 1  # noqa: E701
            elif tree_species == "Eucalyptus":           j = 2  # noqa: E701
            elif tree_species == "Ficus microcarpa":     j = 3  # noqa: E701
            elif tree_species == "Hevea brasiliensis":   j = 4  # noqa: E701
            elif tree_species == "Musa":                 j = 5  # noqa: E701
            elif tree_species == "Psidium guajava":      j = 6  # noqa: E701
            elif tree_species == "Syzygium nervosum":    j = 7  # noqa: E701
            elif tree_species == "Terminalia catappa":   j = 8  # noqa: E701
            elif tree_species == "Veitchia merrilli":    j = 9  # noqa: E701
            else:                                        j = -1 # noqa: E701
            self.labels[i] = one_hot(tensor(j), num_classes=10)
            # fmt:on

        self.images = self.images / 256
        self.images = self.images.to(device=self.device, dtype=float32)
        self.labels = self.labels.to(device=self.device, dtype=float32)

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        return self.images[index], self.labels[index]

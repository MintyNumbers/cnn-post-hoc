from torch.utils.data import Dataset
from torch import Tensor, tensor, float32
from torchvision.io import decode_image
from glob import glob
from torch import device, empty


class BarkVN50Dataset(Dataset):
    def __init__(self, train: bool, device: device):
        self.device = device

        if train:
            self.image_paths = glob("./data/BarkVN-50/Train/*/*")
        else:
            self.image_paths = glob("./data/BarkVN-50/Test/*/*")
        self.image_paths.sort()

        # loading images into memory as tensors
        self.images = empty(self.image_paths.__len__(), 3, 404, 303)
        self.labels = empty(self.image_paths.__len__(), 5)
        for i, path in enumerate(self.image_paths):
            self.images[i] = decode_image(path)

            # labelling the images
            # fmt:off
            tree_species = path.split(sep="/")[-2]
            if   tree_species == "Adenanthera microsperma": self.labels[i] = tensor([1, 0, 0, 0, 0]) # noqa: E701
            elif tree_species == "Cananga odorata":         self.labels[i] = tensor([0, 1, 0, 0, 0]) # noqa: E701
            elif tree_species == "Cedrus":                  self.labels[i] = tensor([0, 0, 1, 0, 0]) # noqa: E701
            elif tree_species == "Cocos nucifera":          self.labels[i] = tensor([0, 0, 0, 1, 0]) # noqa: E701
            elif tree_species == "Dalbergia oliveri":       self.labels[i] = tensor([0, 0, 0, 0, 1]) # noqa: E701
            else:                                           self.labels[i] = tensor([0, 0, 0, 0, 0]) # noqa: E701
            # fmt:on

        self.images = self.images.to(device=self.device, dtype=float32)
        self.labels = self.labels.to(device=self.device, dtype=float32)

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        return self.images[index], self.labels[index]
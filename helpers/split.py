from glob import glob
from os import listdir, makedirs
from shutil import copy

from torchvision.io import decode_image


def train_test_split() -> None:
    """Helper function to split the downloaded Bark classification dataset into a train and test subset."""

    # Tree species list that will be classified by the CNN
    # Only species with at least num_samples samples will be used.
    # The second integer is the number of train samples
    num_samples = 109, 99
    TREE_SPECIES: list[str] = []

    # no checking required as top-level directory contains only subdirectories
    # and subdirectories only contain files
    for species in listdir("./data/BarkVN-50/BarkVN-50_mendeley/"):
        num_dir = len(listdir(f"./data/BarkVN-50/BarkVN-50_mendeley/{species}/"))
        image = decode_image(glob(f"./data/BarkVN-50/BarkVN-50_mendeley/{species}/*")[0])
        if num_dir >= num_samples[0] and image.shape[1] == 404 and image.shape[2] == 303:
            print(num_dir, species)
            TREE_SPECIES.append(species)

    for species in TREE_SPECIES:
        makedirs(f"./data/BarkVN-50/Train/{species}", exist_ok=False)
        makedirs(f"./data/BarkVN-50/Test/{species}", exist_ok=False)
        image_paths: list[str] = glob(f"./data/BarkVN-50/BarkVN-50_mendeley/{species}/*")
        image_paths.sort()

        for i, image_path in enumerate(image_paths):
            if i < num_samples[0]:
                if i < num_samples[1]:
                    copy(image_path, f"./data/BarkVN-50/Train/{species}/")
                else:
                    copy(image_path, f"./data/BarkVN-50/Test/{species}/")
            else:
                break

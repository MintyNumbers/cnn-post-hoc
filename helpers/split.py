from glob import glob
from os import makedirs
from shutil import copy


def train_test_split() -> None:
    """Helper function to split the downloaded Bark classification dataset into a train and test subset."""

    # choosing bark types, which were photographed using similar angles
    # this prevents the CNN from learning inconsequential data
    # furthermore all categories have about the same amount of images
    # this lowers biases of the CNN
    # TODO: cutting amount of samples down to the minimum among the categories (done manually)
    # fmt:off
    TREE_SPECIES: list[str] = [     # total | train / test
        "Adenanthera microsperma",  # 80    | 73    / 7
        "Cananga odorata",          # 80    | 73    / 7
        "Cedrus",                   # 80    | 73    / 7
        "Cocos nucifera",           # 80    | 73    / 7
        "Dalbergia oliveri",        # 80    | 73    / 7       
    ]                               # 400   | 365   / 35
    # fmt:on

    for species in TREE_SPECIES:
        makedirs(f"./data/BarkVN-50/Train/{species}", exist_ok=False)
        makedirs(f"./data/BarkVN-50/Test/{species}", exist_ok=False)
        image_paths: list[str] = glob(f"./data/BarkVN-50/BarkVN-50_mendeley/{species}/*")
        image_paths.sort()
        train_test_split_index = int(image_paths.__len__() * 0.9)

        for i, image_path in enumerate(image_paths):
            if i <= train_test_split_index:
                copy(image_path, f"./data/BarkVN-50/Train/{species}/")
            else:
                copy(image_path, f"./data/BarkVN-50/Test/{species}/")

from torch import Tensor
from torch.nn import BatchNorm2d, Conv2d, Flatten, Linear, MaxPool2d, Module, ReLU, Sequential


class ConvolutionalNeuralNetwork(Module):
    def __init__(self):
        """CNN model for classifying bark images (1*404*303)."""
        super(ConvolutionalNeuralNetwork, self).__init__()

        self.cnn = Sequential(
            Conv2d(in_channels=1, out_channels=64, kernel_size=7, padding=3, stride=1),  # 64 * 404 * 303
            ReLU(),
            BatchNorm2d(num_features=64),
            MaxPool2d(kernel_size=7, stride=7, padding=0),  # 64 * 57 * 43
            Conv2d(in_channels=64, out_channels=128, kernel_size=7, padding=3, stride=1),  # 128 * 57 * 43
            ReLU(),
            BatchNorm2d(num_features=128),
            MaxPool2d(kernel_size=7, stride=7, padding=0),  # 128 * 8 * 6
            Flatten(),
        )

        self.classifier = Sequential(
            Linear(in_features=64 * 16 * 12, out_features=10),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.cnn(x)
        x = self.classifier(x)
        return x

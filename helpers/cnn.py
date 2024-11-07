from torch.nn import Module, Sequential, Conv2d, MaxPool2d, Linear, ReLU
from torch import Tensor


class ConvolutionalNeuralNetwork(Module):
    def __init__(self):
        """CNN model for classifying bark images (303x404 pixels)."""
        super(ConvolutionalNeuralNetwork, self).__init__()

        self.cnn = Sequential(
            Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2, padding=0),
            Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        """
        self.cnn = Sequential(
            Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )
        """

        self.classifier = Sequential(
            Linear(in_features=8 * 101 * 75, out_features=64 * 75),
            ReLU(),
            Linear(in_features=64 * 75, out_features=5),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

from torch import Tensor
from torch.nn import BatchNorm2d, Conv2d, Flatten, Linear, MaxPool2d, Module, ReLU, Sequential


class ConvolutionalNeuralNetwork(Module):
    def __init__(self):
        """CNN model for classifying bark images (1*404*303)."""
        super(ConvolutionalNeuralNetwork, self).__init__()

        self.cnn = Sequential(
            Conv2d(in_channels=1, out_channels=8, kernel_size=5, padding=2, stride=1),  # 8 * 404 * 303
            ReLU(),
            BatchNorm2d(num_features=8),
            MaxPool2d(kernel_size=5, stride=5, padding=2),  # 8 * 81 * 61
            Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=2, stride=1),  # 16 * 81 * 61
            ReLU(),
            BatchNorm2d(num_features=16),
            MaxPool2d(kernel_size=5, stride=5, padding=2),  # 16 * 17 * 13
            Flatten(),
        )

        self.classifier = Sequential(
            Linear(in_features=16 * 17 * 13, out_features=10),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.cnn(x)
        x = self.classifier(x)
        return x

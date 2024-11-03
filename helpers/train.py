from torch.utils.data import DataLoader
from helpers.cnn import ConvolutionalNeuralNetwork
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch import Tensor


def train_cnn(
    num_epochs: int,
    dataloader: DataLoader,
    model: ConvolutionalNeuralNetwork,
    criterion: CrossEntropyLoss,
    optimizer: Adam,
):
    for epoch in range(num_epochs):
        running_loss = 0.0

        for i, (images, labels) in enumerate(dataloader):
            # Forward pass
            outputs: Tensor = model(images)
            loss: Tensor = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}"
            )

        # TODO: Log the running loss

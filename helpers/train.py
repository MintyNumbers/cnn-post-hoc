from datetime import datetime

from torch import Tensor, save
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from helpers.cnn import ConvolutionalNeuralNetwork
from helpers.functions import count_correct_label_batch


def train_cnn(
    num_epochs: int,
    dataloader: DataLoader,
    model: ConvolutionalNeuralNetwork,
    criterion: CrossEntropyLoss,
    optimizer: Adam,
):
    time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    for epoch in range(num_epochs):
        correct_count, all_count = 0, 0
        # running_loss = 0.0

        for images, true_labels in dataloader:
            # Forward pass
            images: Tensor = images.requires_grad_(True)
            outputs: Tensor = model(images)
            loss: Tensor = criterion(outputs, true_labels)

            # predict labels
            all_count += outputs.shape[0]
            correct_count += count_correct_label_batch(outputs=outputs, targets=true_labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # running_loss += loss.item()

        # save checkpoint
        if epoch % 10 == 0 and epoch != 0:
            save(
                {
                    "epoch": num_epochs,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": 123,
                },
                f"models/checkpoint-{epoch}ep-{time}.tar",
            )

        print(
            f"Epoch [{epoch+1}/{num_epochs}],\tLoss: {loss.item():.4f},\tAccuracy: {((correct_count / all_count)*100):.4f}",
        )

        # TODO: Log the running loss

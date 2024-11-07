from torch.utils.data import DataLoader
from helpers.cnn import ConvolutionalNeuralNetwork
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch import Tensor, save
from datetime import datetime
from helpers.functions import predict_label


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
            outputs: Tensor = model(images)
            loss: Tensor = criterion(outputs, true_labels)

            # predict labels
            all_count += outputs.shape[0]
            correct_count += predict_label(outputs=outputs, targets=true_labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # running_loss += loss.item()

        # save checkpoint
        if epoch % 100 == 0:
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

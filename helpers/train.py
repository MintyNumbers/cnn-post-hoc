from torch.utils.data import DataLoader
from helpers.cnn import ConvolutionalNeuralNetwork
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch import Tensor, max, save
from datetime import datetime


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
            _, pred_labels = max(outputs, 1)
            _, true_labels = max(true_labels, 1)
            all_count += pred_labels.shape[0]
            for i in range(pred_labels.shape[0]):
                if pred_labels[i] == true_labels[i]:
                    correct_count += 1

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
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy = {(correct_count / all_count)*100}",
        )

        # TODO: Log the running loss

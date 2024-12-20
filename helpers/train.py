from datetime import datetime

from torch import Tensor, save
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from helpers.cnn import ConvolutionalNeuralNetwork
from helpers.functions import count_correct_label_batch


def train_cnn(
    num_epochs: int,
    dataloader: DataLoader,
    model: ConvolutionalNeuralNetwork,
    criterion: CrossEntropyLoss,
    optimizer: Adam,
) -> Tensor:
    time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    pbar = tqdm(total=num_epochs, desc="Epoch")
    for epoch in range(1, num_epochs + 1):
        correct_count, all_count = 0, 0

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

        # save checkpoint
        if epoch % 10 == 0:
            save(
                {
                    "epoch": num_epochs,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.detach(),
                },
                f"models/checkpoint-{epoch}ep-{time}.tar",
            )

        pbar.write(f"Epoch: {epoch},\tLoss: {loss.item():.4f},\tAccuracy: {((correct_count / all_count)*100):.4f}")
        pbar.update(1)

    return loss

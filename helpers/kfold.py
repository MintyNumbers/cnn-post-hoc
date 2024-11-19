from datetime import datetime

from sklearn.model_selection import KFold
from torch import Tensor, device, no_grad, save
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from tqdm.notebook import tqdm

from helpers.cnn import ConvolutionalNeuralNetwork
from helpers.functions import count_correct_label_batch


def train_cnn_kfold(
    epoch_per_kfold: int,
    num_kfold: int,
    train_dataset: Dataset,
    criterion: CrossEntropyLoss,
    learning_rate: float,
    weight_decay: float,
    device: device,
) -> Tensor:
    def train_kfold(epochs, train_loader):
        for epoch in range(epochs):
            for images, true_labels in train_loader:
                correct_count, all_count = 0, 0

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
            if epoch == epochs - 1:
                save(
                    {
                        "epoch": epochs,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss.detach(),
                    },
                    f"models/kfold-{time}-{fold}.tar",
                )

        return loss, 100 * correct_count / all_count

    def eval_kfold(test_loader):
        with no_grad():
            for images, true_labels in test_loader:
                correct_count, all_count = 0, 0
                # Forward pass
                images: Tensor = images.requires_grad_(True)
                outputs: Tensor = model(images)
                loss: Tensor = criterion(outputs, true_labels)

                # predict labels
                all_count += outputs.shape[0]
                correct_count += count_correct_label_batch(outputs=outputs, targets=true_labels)

        return loss, 100 * correct_count / all_count

    time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    kf = KFold(n_splits=num_kfold, shuffle=True, random_state=0)
    pbar = tqdm(total=num_kfold, desc="K Fold")
    for fold, (train_idx, test_idx) in enumerate(kf.split(train_dataset)):
        model = ConvolutionalNeuralNetwork().to(device)
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Train model with each K-Fold
        model.train()
        train_loader = DataLoader(dataset=train_dataset, batch_size=40, sampler=SubsetRandomSampler(train_idx))
        train_loss, train_acc = train_kfold(epoch_per_kfold, train_loader)

        # Test resulting model
        model.eval()
        test_loader = DataLoader(dataset=train_dataset, batch_size=40, sampler=SubsetRandomSampler(test_idx))
        test_loss, test_acc = eval_kfold(test_loader)

        pbar.write(
            f"K Fold: {fold}\tTrain Loss: {train_loss.item():.4f},\tTrain Accuracy: {train_acc:.4f}\tTest Loss: {test_loss.item():.4f},\tTest Accuracy: {test_acc:.4f}"
        )
        pbar.update(1)

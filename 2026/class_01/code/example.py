"""
Minimal end-to-end PyTorch neural network script,
includes: model, custom dataset + dataloader, training loop, evaluation, saving/loading, and plotting loss curves.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms



def load_dataset():
    # Transforms: convert images to tensors and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    return train_dataset, test_dataset

def create_dummy_dataset():
    # ---- Create a tiny synthetic classification dataset ----
    # 10 input features, 20 classes (matches the network output size).
    N = 2000
    num_features = 10
    num_classes = 20

    X = torch.randn(N, num_features)
    # Create labels 0..19 (random classes)
    y = torch.randint(low=0, high=num_classes, size=(N,), dtype=torch.long)

    # Train/test split
    split = int(0.8 * N)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)

    return train_dataset, test_dataset

# Reproducibility
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Custom Dataset (slides style)
class CustomDataset(Dataset):
    def __init__(self, data: torch.Tensor, labels: torch.Tensor):
        """
        data:  [N, 10] float tensor
        labels:[N]     long tensor with class indices (0..19)
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Model
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features=10, out_features=20)
        self.fc2 = nn.Linear(in_features=20, out_features=20)
        self.fc3 = nn.Linear(in_features=20, out_features=20)
        self.relu = nn.ReLU()
        # NOTE: We intentionally do NOT put Softmax here for training with CrossEntropyLoss.
        # CrossEntropyLoss expects raw logits (it applies log-softmax internally).

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


# Train / Eval helpers
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for data, targets in loader:
        # Move to device cause the model is on the device, and data/targets need to be on the same device for computation
        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(data) # logits
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Accumulate loss (multiply by batch size to get total loss for the batch)
        running_loss += loss.item() * data.size(0)

    # Return average loss over the entire dataset
    return running_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for data, targets in loader:
        data = data.to(device)
        targets = targets.to(device)

        outputs = model(data) # logits
        _, predicted = torch.max(outputs.detach(), dim=1) # dim 1 because we want the max over the class dimension (get the predicted class index)

        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return accuracy


# Main script
def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset, test_dataset = create_dummy_dataset()

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ---- Model, loss, optimizer ----
    model = MyNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # ---- Train ----
    num_epochs = 50
    train_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        acc = evaluate(model, test_loader, device)

        train_losses.append(train_loss)
        val_accuracies.append(acc)

        print(f"Epoch {epoch+1:02d}/{num_epochs} | Train loss: {train_loss:.4f} | Test acc: {acc:.2f}%")

    # ---- Save model ----
    PATH = "output/mynetwork.pth"
    torch.save(model.state_dict(), PATH)
    print(f"Saved model to: {os.path.abspath(PATH)}")

    # ---- Load model ----
    loaded_model = MyNetwork().to(device)
    loaded_model.load_state_dict(torch.load(PATH, map_location=device))
    loaded_model.eval()

    loaded_acc = evaluate(loaded_model, test_loader, device)
    print(f"Loaded model test acc: {loaded_acc:.2f}%")

    # ---- Plot training loss and validation accuracy ----
    epochs = list(range(1, num_epochs + 1))

    plt.figure()
    plt.plot(epochs, train_losses, label="Training loss")
    plt.title("Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, val_accuracies, label="Test accuracy")
    plt.title("Test accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.show()

    # ---- Inference probabilities with softmax ----
    # For a batch:
    with torch.no_grad():
        sample_batch, _ = next(iter(test_loader))
        sample_batch = sample_batch.to(device)
        logits = loaded_model(sample_batch)
        probs = torch.softmax(logits, dim=1)  # now softmax is appropriate
        print("Example probs shape:", probs.shape)  # [batch_size, 20]


if __name__ == "__main__":
    main()

"""
Make a simple neural network using PyTorch. The goal of this exercise is to look at
the MNIST dataset, and tell what numbers are written. Create a model with 3 fully
connected hidden layers, and a 10 unit output layer.
"""


import torch
import torch.nn as nn
import torch.optim as optim
from sympy.physics.units import momentum
from torchvision import datasets, transforms
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
from constants import SHARED_DATA_FOLDER

def download_mnist():
    # Download the MNIST dataset, sizes of the images are 28x28 pixels, and there are 10 classes (digits 0-9)
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root=SHARED_DATA_FOLDER, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=SHARED_DATA_FOLDER, train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader

"""
    1. Create a neural network:
        a. Initialize 3 layers
        b. Define the forward function:
            i. Reshape the data to a fully connected layer. Hint: Use .view().
            ii. Let the input pass through the different layers.
            iii.Consider what activation function you want to use in between the
                layers, and for the final layer.
        c. Loss function and optimizer:
        i. Consider what loss function and optimizer you want to use.
        d. Create the training loop:
        e. Create the evaluation loop:
        f. Save the model
"""

class SpongeNet(nn.Module):
    def __init__(self, activation_fn):
        super(SpongeNet, self).__init__()

        # setting 1: 4 layers
        # self.fc1 = nn.Linear(28*28, 128)  # Input layer to first hidden layer, 28*28 = 784 because MNIST images are 28x28 pixels
        # self.fc2 = nn.Linear(128, 64)     # First hidden layer to second hidden layer
        # self.fc3 = nn.Linear(64, 32)      # Second hidden layer to third hidden layer
        # self.fc4 = nn.Linear(32, 10)      # Third hidden layer to output layer

        # setting 2: 3 layers
        self.fc1 = nn.Linear(28*28, 64)  # Input layer to first hidden layer, 28*28 = 784 because MNIST images are 28x28 pixels
        self.fc2 = nn.Linear(64, 32)      # First hidden layer to second hidden layer
        self.fc3 = nn.Linear(32, 10)      # Second hidden layer to output layer

        self.activation_fn = activation_fn

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input, (batch_size, 1, 28, 28) -> (batch_size, 784), -1 causes the batch size to be inferred automatically

        # x = self.activation_fn(self.fc1(x))  # First hidden layer with activation
        # x = self.activation_fn(self.fc2(x))  # Second hidden layer with activation
        # x = self.activation_fn(self.fc3(x))  # Third hidden layer with activation
        # logits = self.fc4(x)             # Output layer (no activation, as we'll use CrossEntropyLoss, otherwise we would need to use softmax here)

        x = self.activation_fn(self.fc1(x))  # First hidden layer with activation
        x = self.activation_fn(self.fc2(x))  # Second hidden layer with activation
        logits = self.fc3(x)             # Output layer (no activation, as we'll use CrossEntropyLoss, otherwise we would need to use softmax here)
        return logits


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # Clear gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss

@torch.no_grad()
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)  # Forward pass
        _, predicted = torch.max(outputs.data, 1)  # Get the index of the max log-probability
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

def save_model(model, path):
    torch.save(model.state_dict(), path)

def plot_training(train_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Plot training loss
    plt.subplot(1, 2, 1)
    sns.lineplot(x=epochs, y=train_losses, marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    sns.lineplot(x=epochs, y=val_accuracies, marker='o')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.tight_layout()

    plt.savefig('output/sponge_loss_plot.png')

    plt.show()


def main():
    # Download the MNIST dataset
    train_loader, test_loader = download_mnist()

    # Set device check cuda, mps for mac, otherwise cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create the model, loss function, and optimizer
    model = SpongeNet(activation_fn=nn.ReLU()).to(device)
    criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss expects raw logits, it applies log-softmax internally
    adam_optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    sgd_optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train the model
    num_epochs = 5
    train_losses = []
    val_accuracies = []

    # Use tqdm to show a progress bar
    for epoch in tqdm.tqdm(range(num_epochs), desc="Training"):
        train_loss = train(model, train_loader, criterion, sgd_optimizer, device)
        test_accuracy = evaluate(model, test_loader, device)
        train_losses.append(train_loss)
        val_accuracies.append(test_accuracy)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Save the model
    save_model_path = "output/sponge_net.pth"
    save_model(model, save_model_path)
    print(f"Model saved to {save_model_path}")

    # Plot training loss and validation accuracy
    plot_training(train_losses, val_accuracies)


if __name__ == "__main__":
    main()
"""
This exercise will extend on the model created during Exercise 1
and allows you to utilize your knowledge on regularization.
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

# SAVE_MODEL_PATH = "output/sponge_net.pth"
# SAVE_FIG_PATH = 'output/sponge_loss_plot.png'
# Loaded model accuracy: 0.9037, loss: 0.3337

# SAVE_MODEL_PATH = "output/sponge_net_l1.pth"
# SAVE_FIG_PATH = 'output/sponge_loss_plot_l1.png'
# Loaded model accuracy: 0.9081, loss: 0.3191

# SAVE_MODEL_PATH = "output/sponge_net_l2.pth"
# SAVE_FIG_PATH = 'output/sponge_loss_plot_l2.png'
# Loaded model accuracy: 0.9000, loss: 0.3411

SAVE_MODEL_PATH = "output/sponge_net_dropout.pth"
SAVE_FIG_PATH = 'output/sponge_loss_plot_dropout.png'
# Loaded model accuracy: 0.9064, loss: 0.3286

def download_mnist():
    # Download the MNIST dataset, sizes of the images are 28x28 pixels, and there are 10 classes (digits 0-9)
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root=SHARED_DATA_FOLDER, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=SHARED_DATA_FOLDER, train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader

"""
    1. Regularization:
        a. Add L1 regularization to the 2nd layer (the layer after the input layer)
        b. Add L2 regularization instead on the 2nd layer.
        c. What do you observe? (Hint: The lambda value used has a big impact on performance.)
        d. What is the purpose of adding regularization?
    2. Dropout:
        a. Add a dropout layer between the first and second layer. What do you observe?
        b. What is the purpose of adding dropout?
    3. Layers:
        a. Experiment with different number of layers. What do you observe?
        b. Experiment with number of neurons in each layer. What do you observe?
    4. Optimizer:
        a. Play with the hyperparameters of the optimizer.
        b. Test different values of learning rates. With which value do you get the highest accuracy?
        c. What happens if the learning rate is too high?
"""

class SpongeNet(nn.Module):
    def __init__(self, activation_fn):
        super(SpongeNet, self).__init__()

        # setting 2: 3 layers
        self.fc1 = nn.Linear(28*28, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 10)
        self.dropout = nn.Dropout(0.2)

        self.activation_fn = activation_fn

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input, (batch_size, 1, 28, 28) -> (batch_size, 784), -1 causes the batch size to be inferred automatically

        x = self.activation_fn(self.fc1(x))  # First hidden layer with activation
        # Add dropout
        x = self.dropout(x)
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

        # Add regularization term to second layer (L1 or L2)
        # L1 regularization
        # l1_lambda = 1e-5
        # loss += l1_lambda * torch.norm(model.fc2.weight, p=1)

        # L2 regularization
        # l2_lambda = 1e-5
        # loss += l2_lambda * torch.norm(model.fc2.weight, p=2)

        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss

@torch.no_grad()
def evaluate(model, test_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)  # Get the index of the max log-probability
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    avg_loss = total_loss / len(test_loader)
    return accuracy, avg_loss

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_and_evaluate_model():
    # Load the model from path and evaluate it on the test set
    model = SpongeNet(activation_fn=nn.ReLU())
    model.load_state_dict(torch.load(SAVE_MODEL_PATH, weights_only=True))

    _, test_loader = download_mnist()
    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    accuracy, loss = evaluate(model, test_loader, criterion, device)
    print(f"Loaded model accuracy: {accuracy:.4f}, loss: {loss:.4f}")

def plot_training(train_losses, val_accuracies, val_losses):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(18, 5))

    # Plot training loss
    plt.subplot(1, 3, 1)
    sns.lineplot(x=epochs, y=train_losses, marker='o', label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot validation loss
    plt.subplot(1, 3, 2)
    sns.lineplot(x=epochs, y=val_losses, marker='o', label='Validation Loss', color='orange')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot validation accuracy
    plt.subplot(1, 3, 3)
    sns.lineplot(x=epochs, y=val_accuracies, marker='o', label='Validation Accuracy', color='green')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    plt.savefig(SAVE_FIG_PATH)

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
    val_losses = []

    # Use tqdm to show a progress bar
    for epoch in tqdm.tqdm(range(num_epochs), desc="Training"):
        train_loss = train(model, train_loader, criterion, sgd_optimizer, device)
        test_accuracy, test_loss = evaluate(model, test_loader, criterion, device)
        train_losses.append(train_loss)
        val_accuracies.append(test_accuracy)
        val_losses.append(test_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Save the model
    save_model_path = SAVE_MODEL_PATH
    save_model(model, save_model_path)
    print(f"Model saved to {save_model_path}")

    # Plot training loss and validation accuracy
    plot_training(train_losses, val_accuracies, val_losses)


if __name__ == "__main__":
    main()
    load_and_evaluate_model()
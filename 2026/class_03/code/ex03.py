"""
This exercise will extend on the model created during Exercise 1. You must now build a
convolutional neural network that recognises numbers in pictures from the MNIST dataset.
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

SAVE_MODEL_PATH = "output/sponge_net.pth"
SAVE_FIG_PATH = 'output/sponge_loss_plot.png'

def download_mnist():
    # Download the MNIST dataset, sizes of the images are 28x28 pixels, and there are 10 classes (digits 0-9)
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root=SHARED_DATA_FOLDER, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=SHARED_DATA_FOLDER, train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    # 1.a. Output the dimensions of train and test dataset and the size of the images
    print(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")
    print(f"Image size: {train_dataset.data[0].shape}")

    # 1.b. Plot the digit distribution using a bar histogram with the function plt.bar()
    train_labels = train_dataset.targets.numpy()
    test_labels = test_dataset.targets.numpy()
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.countplot(x=train_labels)
    plt.title('Digit Distribution in Train Dataset')
    plt.xlabel('Digit')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    sns.countplot(x=test_labels)
    plt.title('Digit Distribution in Test Dataset')
    plt.xlabel('Digit')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('output/digit_distribution.png')
    plt.show()

    return train_loader, test_loader

"""
    1. Dataset preprocessing:
        a. Output the dimensions of train and test dataset and the size of the images. You
        can get access to the data by calling .data of the MNIST dataset (ie.
        train_dataset.data).
        b. Plot the digit distribution using a bar histogram with the function
        plt.bar().
    2. Train a CNN:
        a. Design an architecture that contains the following:
            i. Convolutional Layer
            ii. MaxPool2D Layer
            iii. Fully Connected Layer
            iv. Dropout Layer
            v. ReLU Activation Function
        b. Achieve at least 95% accuracy in the test dataset by tweaking the architecture
        and save the model.
    3. Check Results
        a. Plot the first convolutional layer filters before and after training the model.
        b. Feed the first convolutional layer an image and plot its activations.
        c. Plot some incorrectly labelled data points. Does it make sense why the
        network was not able to classify those data points?
"""

class SpongeNet(nn.Module):
    def __init__(self, activation_fn):
        super(SpongeNet, self).__init__()
        # initial height and width of the image is 28x28
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 13 * 13, 128)  # After conv and maxpool, the image size is reduced to 13x13

        self.dropout = nn.Dropout(p=0.2)

        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.conv1(x)  # Convolutional layer
        x = self.activation_fn(x)  # Activation function
        x = self.maxpool1(x)  # MaxPool2D layer

        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        x = self.fc1(x)  # Fully connected layer
        x = self.activation_fn(x)  # Activation function
        x = self.dropout(x)  # Dropout layer
        logits = x  # Output layer (no activation, as we'll use CrossEntropyLoss, otherwise we would need to use softmax here)

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

    plot_conv1_activations_v2(model, test_loader, device,
                           save_path='output/conv1_activations_v2.png')

    accuracy, loss = evaluate(model, test_loader, criterion, device)
    print(f"Loaded model accuracy: {accuracy:.4f}, loss: {loss:.4f}")

def load_and_continue_training(num_epochs=3):
    """Load a saved model and continue training from where it left off."""
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Load the model
    model = SpongeNet(activation_fn=nn.ReLU())
    model.load_state_dict(torch.load(SAVE_MODEL_PATH, weights_only=True))
    model.to(device)

    train_loader, test_loader = download_mnist()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Evaluate before continuing training
    accuracy, loss = evaluate(model, test_loader, criterion, device)
    print(f"Before continued training — accuracy: {accuracy:.4f}, loss: {loss:.4f}")

    train_losses, val_accuracies, val_losses = [], [], []

    for epoch in tqdm.tqdm(range(num_epochs), desc="Continued Training"):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_accuracy, test_loss = evaluate(model, test_loader, criterion, device)
        train_losses.append(train_loss)
        val_accuracies.append(test_accuracy)
        val_losses.append(test_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Overwrite the saved model with the updated weights
    save_model(model, f"{SAVE_MODEL_PATH}_continued")
    print(f"Model saved to {SAVE_MODEL_PATH}_continued")

    plot_training(train_losses, val_accuracies, val_losses)

    plot_stuff(model, train_losses, val_losses)

def plot_conv1_filters(model, title, save_path):
    """Plot the 32 filters of the first convolutional layer (conv1).
    Each filter has shape (1, 3, 3) — 1 input channel, 3x3 kernel.
    """
    # weights shape: (out_channels=32, in_channels=1, kH=3, kW=3)
    filters = model.conv1.weight.data.cpu().clone()  # (32, 1, 3, 3)

    # Normalise each filter independently to [0, 1] for display
    # This is needed because the filters can have different ranges of values, and we want to visualise them properly.
    n_filters = filters.shape[0]
    ncols = 8
    nrows = (n_filters + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5))
    fig.suptitle(title, fontsize=14)

    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        ax.axis('off')
        if i < n_filters:
            f = filters[i, 0]  # (3, 3)
            # Normalise to [0, 1]
            f_min, f_max = f.min(), f.max()
            if f_max - f_min > 1e-8:
                f = (f - f_min) / (f_max - f_min)
            ax.imshow(f.numpy(), cmap='gray', interpolation='nearest')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_conv1_activations(model, test_loader, device, save_path):
    """Feed one image through conv1 + ReLU and plot its 32 activation maps."""
    model.eval()

    # Grab the first image from the test loader
    images, labels = next(iter(test_loader))
    image = images[0:1].to(device)  # shape (1, 1, 28, 28)

    with torch.no_grad():
        activations = torch.relu(model.conv1(image))  # (1, 32, 26, 26)

    activations = activations.squeeze(0).cpu()  # (32, 26, 26)
    n_maps = activations.shape[0]
    ncols = 8
    nrows = (n_maps + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows + 1, ncols, figsize=(ncols * 1.5, (nrows + 1) * 1.5))
    fig.suptitle(f'Conv1 Activations (label: {labels[0].item()})', fontsize=14)

    # First row: show the original input image
    for j in range(ncols):
        axes[0, j].axis('off')
    axes[0, ncols // 2 - 1].imshow(images[0, 0].numpy(), cmap='gray')
    axes[0, ncols // 2 - 1].set_title('Input', fontsize=9)
    axes[0, ncols // 2 - 1].axis('on')
    axes[0, ncols // 2 - 1].set_xticks([])
    axes[0, ncols // 2 - 1].set_yticks([])

    # Remaining rows: activation maps
    for i in range(nrows * ncols):
        ax = axes[1 + i // ncols, i % ncols]
        ax.axis('off')
        if i < n_maps:
            act = activations[i].numpy()
            ax.imshow(act, cmap='viridis')
            ax.set_title(f'F{i}', fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_conv1_activations_v2(model, test_loader, device, save_path):
    model.eval()

    img_batch = next(iter(test_loader))[0]  # shape (batch_size, 1, 28, 28)
    single_image = img_batch[0:1].to(device)  # shape (1, 1, 28, 28) — take only the first image
    conv1_output = model.conv1(single_image)  # shape (1, 32, 26, 26)
    layer_visualization = conv1_output.squeeze(0).cpu().data  # shape (32, 26, 26)

    for i, feature_map in enumerate(layer_visualization):  # each feature_map is (26, 26)
        plt.subplot(4, 8, i + 1)
        plt.imshow(feature_map.numpy(), cmap='gray')
        plt.axis('off')

    plt.savefig(save_path)


def plot_misclassified(model, test_loader, device, save_path, n_samples=25):
    """Collect misclassified images and plot them with true vs predicted labels."""
    model.eval()
    wrong_images = []
    wrong_true = []
    wrong_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            mask = predicted != labels  # Boolean mask of wrong predictions
            wrong_images.append(images[mask].cpu())
            wrong_true.append(labels[mask].cpu())
            wrong_pred.append(predicted[mask].cpu())

            if sum(len(w) for w in wrong_images) >= n_samples:
                break

    wrong_images = torch.cat(wrong_images)[:n_samples]
    wrong_true = torch.cat(wrong_true)[:n_samples]
    wrong_pred = torch.cat(wrong_pred)[:n_samples]

    ncols = 5 # Show 5 misclassified samples per row
    nrows = (n_samples + ncols - 1) // ncols # Calculate number of rows needed

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2.2))
    fig.suptitle('Misclassified Samples', fontsize=14)

    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        ax.axis('off')
        if i < n_samples:
            ax.imshow(wrong_images[i, 0].numpy(), cmap='gray')
            ax.set_title(f'True: {wrong_true[i].item()}\nPred: {wrong_pred[i].item()}',
                         fontsize=8, color='red')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


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

def plot_stuff(model, test_loader, device):
    # 3.a — Plot conv1 filters AFTER training
    plot_conv1_filters(model, title='Conv1 Filters - After Training',
                       save_path='output/conv1_filters_after.png')

    # 3.b — Feed one image through conv1 and plot its activations
    plot_conv1_activations(model, test_loader, device,
                           save_path='output/conv1_activations.png')

    # 3.c — Plot misclassified samples
    plot_misclassified(model, test_loader, device,
                       save_path='output/misclassified.png')

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

    # 3.a — Plot conv1 filters BEFORE training
    plot_conv1_filters(model, title='Conv1 Filters - Before Training',
                       save_path='output/conv1_filters_before.png')

    # Train the model
    num_epochs = 3
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

    plot_stuff(model, test_loader, device)



if __name__ == "__main__":
    # main()
    # load_and_evaluate_model()
    load_and_continue_training(num_epochs=2)

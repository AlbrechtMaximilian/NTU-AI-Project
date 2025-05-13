import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import Subset
import json
import os
os.environ['KMP_WARNINGS'] = '0'

dataset_path = "Khan Dataset Mel"

#resize + ToTensor
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # can adjust size
    transforms.ToTensor()
])
# loading the dataset using ImageFolder
full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Splitting the dataset into training and validation sets (80/20)
# Using shuffled split
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

indices = torch.randperm(len(full_dataset)).tolist()
train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)

# creating DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#####
# Set up pretrained model
#####

# Load and use a pretrained ResNet18 CNN (pretrained image CNN)
resnet18 = models.resnet18(weights='DEFAULT')

# We want to predict 5 classes, so we modify the output layer to 5
resnet18.fc = nn.Linear(512, 5)  # 5 heart disease classes

# Because we have limited data, we freeze lower layers
#   and only train final classifier:
for param in resnet18.parameters():
    param.requires_grad = False

# Unfreeze the final layer
for param in resnet18.fc.parameters():
    param.requires_grad = True
# We will later unfreeze earlier layers for full fine-tuning

#####
# Train the CNN on our data
#####

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet18 = resnet18.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = optim.Adam(resnet18.fc.parameters(), lr=0.001)  # Only optimizing the unfrozen final layer

# Track the best validation accuracy
best_val_acc = 0.0

# Store accuracy curves and predictions
train_accuracies = []
val_accuracies = []
all_preds = []
all_probs = []
all_targets = []

best_val_acc = 0.0

# Training loop
num_epochs = 10

for epoch in range(num_epochs):

    # Unfreeze layer4 after 5 epochs for fine-tuning
    if epoch == 5:
        for param in resnet18.layer4.parameters():
            param.requires_grad = True
        print("🔓 Unfroze layer4 for fine-tuning")

    resnet18.train()  # Set to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # zero gradients
        outputs = resnet18(images)  # forward pass
        loss = criterion(outputs, labels)  # compute loss
        loss.backward()  # backpropagation
        optimizer.step()  # update weights

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, "
          f"Train Accuracy: {train_acc:.4f}")

    #####
    # Validation loop
    #####
    resnet18.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = resnet18(images)
            # Compute and store probabilities and stats
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)

            all_probs.extend(probs.cpu())
            all_preds.extend(predicted.cpu())
            all_targets.extend(labels.cpu())

            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    print(f"Validation Accuracy: {val_acc:.4f}")

    # Save accuracies
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    # Save best model based on validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(resnet18.state_dict(), "best_model.pth")
        print("✅ Best model saved at epoch", epoch + 1)

    # Save performance-related outputs for later analysis
    torch.save(torch.stack(all_probs),
               "val_probs.pt")  # predicted probabilities (for AUC)
    torch.save(torch.tensor(all_preds),
               "val_preds.pt")  # predicted class labels
    torch.save(torch.tensor(all_targets), "val_targets.pt")  # true class labels

    # Save training/validation accuracy curves
    with open("training_metrics.json", "w") as f:
        json.dump({
            "train_accuracy": train_accuracies,
            "val_accuracy": val_accuracies
        }, f)

    print("📁 Evaluation data saved.")

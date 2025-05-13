import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

dataset_path = '/content/drive/MyDrive/AI_Python_final_project/Khan Dataset'#will change the path

#resize + ToTensor
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # can adjust size
    transforms.ToTensor()
])
# loading the dataset using ImageFolder
full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

#spliting the dataset into training and validation sets (80/20)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
# creating DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
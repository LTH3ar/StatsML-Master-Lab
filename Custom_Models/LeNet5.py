import torch.nn as nn
import torch
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Feature extractor
        self.feature = nn.Sequential(
            # Convolutional layers

            # ============================================================================== #
            # First conv layer
            # input: 1 x 28 x 28 --> padding = 2 --> 1 x 32 x 32 --> 6 x 28 x 28
            # nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            # activation function
            nn.Tanh(),
            # pooling layer 14 x 14
            nn.AvgPool2d(kernel_size=2, stride=2),
            # ============================================================================== #

            # ============================================================================== #
            # Second conv layer
            # input: 6 x 14 x 14 --> 16 x 10 x 10
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            # activation function
            nn.Tanh(),
            # pooling layer 5 x 5
            nn.AvgPool2d(kernel_size=2, stride=2),
            # ============================================================================== #
        )

        # Classifier
        self.classifier = nn.Sequential(
            # Fully connected layers

            # ============================================================================== #
            # First fc layer
            # input: 16 x 5 x 5 = 400 --> 120
            # flatten
            nn.Flatten(),
            # fc layer
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            # activation function
            nn.Sigmoid(),  # sigmoid
            # ============================================================================== #

            # ============================================================================== #
            # Second fc layer
            nn.Linear(in_features=120, out_features=84),

            # activation function
            nn.Sigmoid(),  # sigmoid
            # ============================================================================== #

            # ============================================================================== #
            # Third fc layer
            nn.Linear(in_features=84, out_features=num_classes),
            # ============================================================================== #
            nn.Softmax(dim=1)
        )

    # Forward function
    def forward(self, x):
        return self.classifier(self.feature(x))
    
    def preprocessing_dataset(self, dataset_name, batch_size, num_workers):
        # if MNIST: 
        """
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),  # 50% chance of horizontal flip
        transforms.RandomVerticalFlip(p=0.5),    # 50% chance of vertical flip
        transforms.Normalize((0.1307,), (0.3081,))  # mean and std for MNIST
        """
        # if FashionMNIST:
        """
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),  # 50% chance of horizontal flip
        transforms.RandomVerticalFlip(p=0.5),    # 50% chance of vertical flip
        transforms.Normalize((0.2860,), (0.3530,))  # mean and std for FashionMNIST
        """
        # if CIFAR10: # have color so different Normalization but still keep the same color
        """
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),  # 50% chance of horizontal flip
        transforms.RandomVerticalFlip(p=0.5),    # 50% chance of vertical flip
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) 
        """
        
        # main
        if dataset_name == "MNIST":
            transform = transforms.Compose([
                transforms.Pad(2),
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            # split train dataset to train and validation dataset
            train_size = int(0.8 * len(datasets.MNIST(root='./data', train=True, download=True, transform=transform)))
            val_size = len(datasets.MNIST(root='./data', train=True, download=True, transform=transform)) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(datasets.MNIST(root='./data', train=True, download=True, transform=transform), [train_size, val_size])
            test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        elif dataset_name == "FashionMNIST":
            transform = transforms.Compose([
                transforms.Pad(2),
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.Normalize((0.2860,), (0.3530,))
            ])
            train_size = int(0.8 * len(datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)))
            val_size = len(datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform), [train_size, val_size])
            test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        elif dataset_name == "CIFAR10":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            train_size = int(0.8 * len(datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)))
            val_size = len(datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(datasets.CIFAR10(root='./data', train=True, download=True, transform=transform), [train_size, val_size])
            test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        else:
            raise ValueError("Unsupported dataset. Choose from 'MNIST', 'FashionMNIST', or 'CIFAR10'.")
        
        # Create DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, val_loader, test_loader
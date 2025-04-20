import torch.nn as nn
import torch
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # ============================================================================== #
            # 1st conv layer
            # input: 3x224x224 (upscaled from 1x28x28)
            # output: 96x55x55
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0, ),
            # activation function: ReLU
            nn.ReLU(),
            # max pooling layer with kernel size 3 and stride 2
            # output: 96x27x27
            nn.MaxPool2d(kernel_size=3, stride=2),
            # ============================================================================== #
            
            # ============================================================================== #
            # 2nd conv layer
            # input: 96x27x27
            # output: 256x27x27
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            # activation function: ReLU
            nn.ReLU(),
            # max pooling layer with kernel size 3 and stride 2
            # output: 256x13x13
            nn.MaxPool2d(kernel_size=3, stride=2),
            # ============================================================================== #
            
            # ============================================================================== #
            # 3rd conv layer
            # input: 256x13x13
            # output: 384x13x13
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            # activation function: ReLU
            nn.ReLU(),
            # ============================================================================== #
            
            # ============================================================================== #
            # 4th conv layer
            # input: 384x13x13
            # output: 384x13x13
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            # activation function: ReLU
            nn.ReLU(),
            # ============================================================================== #
            
            # ============================================================================== #
            # 5th conv layer
            # input: 384x13x13
            # output: 256x13x13
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            # activation function: ReLU
            nn.ReLU(),
            # max pooling layer with kernel size 3 and stride 2
            # output: 256x6x6
            nn.MaxPool2d(kernel_size=3, stride=2)
            # ============================================================================== #
        )

        self.classifier = nn.Sequential(
            # flatten
            nn.Flatten(), # 256*5*5 = 6400
            # ============================================================================== #
            # 1st fc layer Dense: 4096 fully connected neurons
            nn.Dropout(p=0.5), # dropout layer with p=0.5
            nn.Linear(in_features=256 * 6 * 6, out_features=4096), # 256*5*5
            nn.ReLU(),
            # ============================================================================== #
            
            # ============================================================================== #
            # 2nd fc layer Dense: 4096 fully connected neurons
            nn.Dropout(p=0.5), # dropout layer with p=0.5
            nn.Linear(in_features=4096, out_features=4096), # 4096
            nn.ReLU(),
            # ============================================================================== #
            
            # ============================================================================== #
            # 3rd fc layer Dense: 10 fully connected neurons
            nn.Linear(in_features=4096, out_features=num_classes) # 4096
            # ============================================================================== #

        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x
    
    def preprocessing_dataset(self, dataset_name, batch_size, num_workers):
        # if MNIST: 
        """
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),  # 50% chance of horizontal flip
        transforms.RandomVerticalFlip(p=0.5),    # 50% chance of vertical flip
        transforms.Normalize((0.1307,), (0.3081,))  # mean and std for MNIST
        # add 3 channels
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))
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
                transforms.Resize((227, 227)),  # Resize to 224x224 for AlexNet
                transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
                transforms.ToTensor(),
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomVerticalFlip(p=0.5),
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
                transforms.Resize((227, 227)),  # Resize to 224x224 for AlexNet
                transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
                transforms.ToTensor(),
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomVerticalFlip(p=0.5),
                transforms.Normalize((0.2860,), (0.3530,))
            ])
            train_size = int(0.8 * len(datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)))
            val_size = len(datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform), [train_size, val_size])
            test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        elif dataset_name == "CIFAR10":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((227, 227)),  # Resize to 224x224 for AlexNet
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomVerticalFlip(p=0.5),
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
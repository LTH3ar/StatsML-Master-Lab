# Importing Libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import transforms
from torchmetrics import Accuracy
from torchinfo import summary
import numpy as np
import os
import datetime

# 2. Setting Device

# Setting Device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(device)

# 3. Preparing Input Data

# Preparing Input Data
# prepare the dataset MNIST(1x28x28) -> (3x224x224) for LeNet
# Upscale the grayscale images to RGB size
param_transform = transforms.Compose([
    transforms.Pad(2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307], std=[0.3081])  # Normalize to [-1, 1] range
])

# Download the dataset
train_val_dataset = datasets.MNIST(root='./dataset', train=True, transform=param_transform, download=True)

# Dataset summary
print('train_val_dataset length:', len(train_val_dataset))
print('train_val_dataset shape:', train_val_dataset[0][0].shape)

# Split the dataset into train and validation
train_size = int(0.8 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [train_size, val_size])

# Dataset summary
print('train_dataset length:', len(train_dataset))
print('val_dataset length:', len(val_dataset))

# Create dataloaders
if torch.cuda.is_available():
    BATCH_SIZE = 128
elif torch.backends.mps.is_available():
    BATCH_SIZE = 128
else:
    BATCH_SIZE = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# dataloaders summary
print('train_loader length:', len(train_loader))
print('val_loader length:', len(val_loader))

# 4. Defining Model

# Define the model LeNet specific for the transformed MNIST
class LeNet5(nn.Module):
    def __init__(self):
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
            nn.Linear(in_features=84, out_features=10),
            # ============================================================================== #
            nn.Softmax(dim=1)
        )

    # Forward function
    def forward(self, x):
        return self.classifier(self.feature(x))


# %%
# Create model
model = LeNet5().to(device)
print(model)

# Model summary
# Detailed layer-wise summary
#summary(model, input_size=(1, 3, 224, 224), verbose=2, device=device)
summary(model, input_size=(1, 1, 32, 32), verbose=2, device=device)
# %%
# Optimizer and loss function
LEARNING_RATE = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_function = nn.CrossEntropyLoss().to(device)
accuracy = Accuracy(task='multiclass', num_classes=10).to(device)
# %% md
# 5. Training
# %%
# Training
# Log training process to TensorBoard
date_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
log_dir = os.path.join('train_logs', date_time)
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

# Training
# Training Parameters
NUM_EPOCHS = 40
NUM_BATCHES = len(train_loader)
NUM_BATCHES_VAL = len(val_loader)
print('NUM_BATCHES:', NUM_BATCHES)
print('NUM_BATCHES_VAL:', NUM_BATCHES_VAL)
if not os.path.exists('models'):
    os.mkdir('models')

# Training Loop

# Calculate function for metrics
from torchmetrics import Precision, Recall, F1Score

# Initialize metrics
precision_train = Precision(task='multiclass', num_classes=10).to(device)
recall_train = Recall(task='multiclass', num_classes=10).to(device)
f1_train = F1Score(task='multiclass', num_classes=10).to(device)

precision_val = Precision(task='multiclass', num_classes=10).to(device)
recall_val = Recall(task='multiclass', num_classes=10).to(device)
f1_val = F1Score(task='multiclass', num_classes=10).to(device)

# Training loop
best_val_acc = 0.0

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * data.size(0)
        _, predicted = torch.max(output, 1)
        train_total += target.size(0)
        train_correct += (predicted == target).sum().item()

        precision_train.update(output, target)
        recall_train.update(output, target)
        f1_train.update(output, target)

    avg_train_loss = train_loss / train_total
    avg_train_acc = 100.0 * train_correct / train_total
    avg_train_precision = precision_train.compute().item()
    avg_train_recall = recall_train.compute().item()
    avg_train_f1 = f1_train.compute().item()

    precision_train.reset()
    recall_train.reset()
    f1_train.reset()

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_function(output, target)

            val_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output, 1)
            val_total += target.size(0)
            val_correct += (predicted == target).sum().item()

            precision_val.update(output, target)
            recall_val.update(output, target)
            f1_val.update(output, target)

    avg_val_loss = val_loss / val_total
    avg_val_acc = 100.0 * val_correct / val_total
    avg_val_precision = precision_val.compute().item()
    avg_val_recall = recall_val.compute().item()
    avg_val_f1 = f1_val.compute().item()

    precision_val.reset()
    recall_val.reset()
    f1_val.reset()

    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.2f}%, '
          f'Train Precision: {avg_train_precision:.4f}, Train Recall: {avg_train_recall:.4f}, Train F1: {avg_train_f1:.4f}, '
          f'Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.2f}%, Val Precision: {avg_val_precision:.4f}, '
          f'Val Recall: {avg_val_recall:.4f}, Val F1: {avg_val_f1:.4f}')

    writer.add_scalar('Loss/Train', avg_train_loss, epoch)
    writer.add_scalar('Accuracy/Train', avg_train_acc, epoch)
    writer.add_scalar('Precision/Train', avg_train_precision, epoch)
    writer.add_scalar('Recall/Train', avg_train_recall, epoch)
    writer.add_scalar('F1-Score/Train', avg_train_f1, epoch)
    writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
    writer.add_scalar('Accuracy/Validation', avg_val_acc, epoch)
    writer.add_scalar('Precision/Validation', avg_val_precision, epoch)
    writer.add_scalar('Recall/Validation', avg_val_recall, epoch)
    writer.add_scalar('F1-Score/Validation', avg_val_f1, epoch)

    if avg_val_acc > best_val_acc:
        best_val_acc = avg_val_acc
        MODEL_NAME = f'LeNet-5_best_{date_time}.pth'
        torch.save(model.state_dict(), os.path.join('models', MODEL_NAME))
        print(f'Saved best model to {MODEL_NAME}')

if torch.cuda.is_available():
    torch.cuda.empty_cache()
elif torch.backends.mps.is_available():
    torch.mps.empty_cache()
else:
    torch.empty_cache()

writer.flush()
writer.close()

print('Training complete. Released all variables.')
import Custom_Utils.main as Custom_Utils
from Custom_Models import LeNet5, AlexNet, ResNet18
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import transforms
from torchmetrics import Accuracy
from torchinfo import summary
# import prebuilt models
import torchvision
import numpy as np
import os
import datetime
import sys

# setup device and batch size
device = Custom_Utils.set_device()
batch_size = Custom_Utils.set_batch_size()
print(f"Using device: {device}, Batch size: {batch_size}")

# Lenet 5 (sys argv 1 for model name, arg 2 for dataset name)
if len(sys.argv) < 3:
    print("Usage: python main.py <model_name> <dataset_name>")
    sys.exit(1)
model_name = sys.argv[1]
dataset_name = sys.argv[2]
print(f"Model: {model_name}, Dataset: {dataset_name}")
# Check if the dataset name is valid
if dataset_name not in ["MNIST", "FashionMNIST", "CIFAR10"]:
    print(f"Dataset {dataset_name} not found.")
    sys.exit(1)

# Check if the model name is valid
if model_name not in ["LeNet-5", "AlexNet", "ResNet18"]:
    print(f"Model {model_name} not found.")
    sys.exit(1)

# models & datasets
if model_name == "LeNet-5":
    model = LeNet5.LeNet5(num_classes=10)
elif model_name == "AlexNet":
    model = AlexNet.AlexNet(num_classes=10)
elif model_name == "ResNet18":
    model = ResNet18.ResNet18(num_classes=10, pretrained=False)

# dataloader
train_dataloader, val_dataloader, test_dataloader = model.preprocessing_dataset(dataset_name, batch_size, num_workers=4)

# print model summary
if model_name == "LeNet-5":
    summary(model, input_size=(batch_size, 1, 32, 32), device=device)
elif model_name == "AlexNet":
    summary(model, input_size=(batch_size, 3, 227, 227), device=device)
elif model_name == "ResNet18":
    summary(model, input_size=(batch_size, 3, 227, 227), device=device)

# move to device
model = model.to(device)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# loss function
loss_function = nn.CrossEntropyLoss()
# metrics
task = "multiclass"
num_classes = 10
precision, recall, f1_score = Custom_Utils.set_metrics(task, num_classes)

# setup logging
folder_name = sys.argv[2]
task_type = "classification"
writer, log_dir = Custom_Utils.setup_logging(task_type, model_name, folder_name)

# training loop
num_epochs = sys.argv[3] if len(sys.argv) > 3 else 10
train_metrics = (precision, recall, f1_score)
val_metrics = (precision, recall, f1_score)

# move all to device
precision = precision.to(device)
recall = recall.to(device)
f1_score = f1_score.to(device)
loss_function = loss_function.to(device)

best_model_path = Custom_Utils.training_loop(
    model,
    model_name=model_name, 
    num_epochs=num_epochs, 
    train_loader=train_dataloader, 
    val_loader=val_dataloader, 
    optimizer=optimizer, 
    loss_function=loss_function, 
    device=device, 
    writer=writer,
    train_metrics=train_metrics,
    val_metrics=val_metrics,
    log_dir=log_dir
)

# test
if model_name == "LeNet-5":
    init_model = LeNet5.LeNet5(num_classes=10)
elif model_name == "AlexNet":
    init_model = AlexNet.AlexNet(num_classes=10)
elif model_name == "ResNet18":
    init_model = ResNet18.ResNet18(num_classes=10, pretrained=False)

test_precision, test_recall, test_f1_score = Custom_Utils.set_metrics(task, num_classes)
test_precision = test_precision.to(device)
test_recall = test_recall.to(device)
test_f1_score = test_f1_score.to(device)
model = Custom_Utils.file2model(init_model, best_model_path).to(device)
Custom_Utils.test_model(model, test_dataloader, loss_function, (test_precision, test_recall, test_f1_score), device, log_dir)
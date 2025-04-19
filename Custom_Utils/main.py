import torch

def set_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
    

def set_batch_size():
    if torch.cuda.is_available():
        return 128
    elif torch.backends.mps.is_available():
        return 128
    else:
        return 64
    
from torchmetrics import Accuracy, Precision, Recall, F1Score
def set_metrics(task, num_classes):
    precision = Precision(task=task, num_classes=num_classes)
    recall = Recall(task=task, num_classes=num_classes)
    f1_score = F1Score(task=task, num_classes=num_classes)
    return precision, recall, f1_score

import os
import datetime
from torch.utils.tensorboard import SummaryWriter
def setup_logging(task_type, model_name, folder_name):
    date_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    log_dir = os.path.join(folder_name, f"{task_type}_{model_name}", date_time)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    return writer, log_dir

def training_loop(
        model, 
        model_name, 
        num_epochs, 
        train_loader, 
        val_loader, 
        optimizer, 
        loss_function, 
        device, 
        writer, 
        train_metrics, 
        val_metrics,
        log_dir):
    
    best_val_loss = float('inf')
    train_loss_lst = []
    val_loss_lst = []
    train_accuracy_lst = []
    val_accuracy_lst = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_loss /= total_train
        train_accuracy = train_correct / total_train
        train_loss_lst.append(train_loss)
        train_accuracy_lst.append(train_accuracy)

        for metric in train_metrics:
            metric(outputs, labels)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                total_val += labels.size(0)

                for metric in val_metrics:
                    metric(outputs, labels)

        val_loss /= total_val
        val_accuracy = val_correct / total_val
        val_loss_lst.append(val_loss)
        val_accuracy_lst.append(val_accuracy)

        # Log metrics to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)

        for metric in train_metrics:
            writer.add_scalar(f'Train/{metric.__class__.__name__}', metric.compute(), epoch)
            metric.reset()

        for metric in val_metrics:
            writer.add_scalar(f'Validation/{metric.__class__.__name__}', metric.compute(), epoch)
            metric.reset()

        # Save best model weights
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # format: {model_name}_{epoch}_{val_loss:.4f}.pth
            torch.save(model.state_dict(), os.path.join(log_dir, f'best_model_{model_name}.pth'))

        print(f"Epoch {epoch}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Save training and validation metrics
    torch.save({
        'train_loss': train_loss_lst,
        'val_loss': val_loss_lst,
        'train_accuracy': train_accuracy_lst,
        'val_accuracy': val_accuracy_lst
    }, os.path.join(log_dir, 'metrics.pth'))

    # Plot and save loss and accuracy curves
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_lst, label='Train Loss')
    plt.plot(val_loss_lst, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'loss_curve.png'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracy_lst, label='Train Accuracy')
    plt.plot(val_accuracy_lst, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'accuracy_curve.png'))
    plt.close()

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
    num_epochs = int(num_epochs)
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
            torch.save(model.state_dict(), os.path.join(log_dir, f'{model_name}_{epoch+1}_{val_loss:.4f}.pth'))
            print(f"Model saved at epoch {epoch+1} with validation loss: {val_loss:.4f}")
            best_model_path = os.path.join(log_dir, f'{model_name}_{epoch+1}_{val_loss:.4f}.pth')

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Save training and validation metrics
    torch.save({
        'train_loss': train_loss_lst,
        'val_loss': val_loss_lst,
        'train_accuracy': train_accuracy_lst,
        'val_accuracy': val_accuracy_lst
    }, os.path.join(log_dir, 'metrics.pth'))
    with open(os.path.join(log_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Train Loss: {train_loss_lst}\n")
        f.write(f"Validation Loss: {val_loss_lst}\n")
        f.write(f"Train Accuracy: {train_accuracy_lst}\n")
        f.write(f"Validation Accuracy: {val_accuracy_lst}\n")

    # Plot and save loss and accuracy curves, turn on grid
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_lst, label='Train Loss')
    plt.plot(val_loss_lst, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(log_dir, 'loss_curve.png'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracy_lst, label='Train Accuracy')
    plt.plot(val_accuracy_lst, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(log_dir, 'accuracy_curve.png'))
    plt.close()

    # return the path to the best model
    return best_model_path

import sys
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
def file2model(model, weight_path):
    if os.path.isfile(weight_path):
        print(f"Loading model weights from {weight_path}")
        model.load_state_dict(torch.load(weight_path))
    else:
        print(f"Model weights file {weight_path} not found.")
        sys.exit(1)
    return model

def test_model(model, test_loader, loss_function, metrics, device, log_dir):
    model.eval()
    precision, recall, f1_score = metrics
    test_loss = 0.0
    test_correct = 0
    total_test = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
            total_test += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            precision(outputs, labels)
            recall(outputs, labels)
            f1_score(outputs, labels)

    test_loss /= total_test
    test_accuracy = test_correct / total_test

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {precision.compute():.4f}, Recall: {recall.compute():.4f}, F1 Score: {f1_score.compute():.4f}")

    # Confusion Matrix
    import matplotlib.pyplot as plt

    cm = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:")
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(log_dir, 'confusion_matrix.png'))
    plt.show()

    # save all metrics with confusions matrix in a text file
    with open(os.path.join(log_dir, 'test_metrics.txt'), 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Precision: {precision.compute():.4f}\n")
        f.write(f"Recall: {recall.compute():.4f}\n")
        f.write(f"F1 Score: {f1_score.compute():.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\n")

    # Reset metrics
    precision.reset()
    recall.reset()
    f1_score.reset()
import numpy as np
import torch
import time
import copy
import torch.nn as nn
from test import test
from torchmetrics import Accuracy, F1Score


def train(model: torch.nn.Module, criterion,
          train_loader, val_loader, test_loader, optimizer: torch.optim.Optimizer,
          device: torch.device, max_epoch: int, disp_freq):  # para: val_set

    avg_train_loss, train_acc, train_f1 = [], [], []
    avg_val_loss, avg_val_acc, avg_val_f1 = [], [], []

    min_val_loss = 1e9  # Guaranteed to update for the first time
    last_min_ind = -1
    early_stopping_epoch = 300  # early stop
    best_val_acc = 0
    best_val_f1 = 0

    # Training process
    # tran and update model only on train, but keep track of best_model based on valid performance?
    for epoch in range(max_epoch):
        batch_train_loss, batch_train_acc, batch_train_f1 = train_one_epoch(model, criterion, train_loader, optimizer, device, max_epoch, disp_freq, epoch)
        batch_val_loss, batch_val_acc, batch_val_f1 = validate(model, criterion, val_loader, device)

        avg_train_loss.append(np.mean(batch_train_loss))
        train_acc.append(batch_train_acc)
        train_f1.append(batch_train_f1)
        
        avg_val_loss.append(np.mean(batch_val_loss))
        avg_val_acc.append(batch_val_acc)
        avg_val_f1.append(batch_val_f1)

        fp = open('output.log', 'a+')
        print(f'Epoch [{epoch}]\t Average training loss {avg_train_loss[-1]:.4f}\t Average validation loss {avg_val_loss[-1]:.4f}\t Average validation accuracy {avg_val_acc[-1]:.4f}\t Average validation f1 {avg_val_f1[-1]:.4f}', file=fp)
        fp.close()
        print(f'Epoch [{epoch}]\t Average training loss {avg_train_loss[-1]:.4f}\t Average validation loss {avg_val_loss[-1]:.4f}\t Average validation accuracy {avg_val_acc[-1]:.4f}\t Average validation f1 {avg_val_f1[-1]:.4f}')
        early_stop_epoch = 0
        running_loss = avg_val_loss[-1]

        # early stop
        if running_loss < min_val_loss:
            last_min_ind = epoch
            min_val_loss = running_loss  # Check whether the val loss becomes smaller after each epoch
            best_model = copy.deepcopy(model)
            best_val_acc = avg_val_acc[-1]
            best_val_f1 = avg_val_f1[-1]
            print("Best ckpt(Loss)", min_val_loss)
        elif epoch - last_min_ind >= early_stopping_epoch:
            early_stop_epoch = epoch
            break

    return model, best_model, avg_train_loss, train_acc, train_f1, avg_val_loss, avg_val_acc, avg_val_f1, early_stop_epoch


def train_one_epoch(model: torch.nn.Module, criterion,
                    train_loader, optimizer: torch.optim.Optimizer,
                    device: torch.device, max_epoch: int, disp_freq, epoch):
    model.train(True)
    batch_train_loss = []
    max_train_iteration = len(train_loader)

    accuracy = Accuracy(task='binary').to(device)
    f1 = F1Score(task='binary', num_classes=2).to(device)

    for iteration, (inputs, labels, _) in enumerate(train_loader):
        start_time = time.time()
        inputs = inputs.to(device)
        #inputs[inputs > 0.1] = 1
        labels = labels.to(device)
        output = model(inputs)

        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_train_loss.append(loss.detach().item())

        _, predicted = torch.max(output, 1)
        print(f"train output: {output}")
        print(f"train label pred: {predicted}")
        print(f"train true label: {labels}")

        accuracy.update(predicted, labels)
        f1.update(predicted, labels)
        end_time = time.time()
        batch_time = end_time - start_time

        if iteration % disp_freq == 0:
            batch_acc = accuracy.compute().item()
            batch_f1 = f1.compute().item()
            batch_avg_loss = np.mean(batch_train_loss)
            fp = open('output.log', 'a+')
            print(f"Epoch [{epoch}][{max_epoch}]\t Batch [{iteration}][{max_train_iteration}]\t Training Loss {batch_avg_loss:.4f}\t Accuracy {batch_acc:.4f}\t F1 {batch_f1:.4f}\t Time(Iter) {batch_time:.4f}", file=fp)
            fp.close()
            print(f"Epoch [{epoch}][{max_epoch}]\t Batch [{iteration}][{max_train_iteration}]\t Training Loss {batch_avg_loss:.4f}\t Accuracy {batch_acc:.4f}\t F1 {batch_f1:.4f}\t Time(Iter) {batch_time:.4f}")
    batch_acc = accuracy.compute().item()
    batch_f1 = f1.compute().item()
    return batch_train_loss, batch_acc, batch_f1


def validate(model, criterion, val_loader, device: torch.device):
    batch_val_loss = []
    model.eval()
    accuracy = Accuracy(task='binary').to(device)
    f1 = F1Score(task='binary', num_classes=2).to(device)
    with torch.no_grad():
        for iteration, (inputs, labels, _) in enumerate(val_loader):
            # Get validating data and label
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Only forward pass
            probs = model(inputs)
            loss = criterion(probs, labels)
            _, predicted = torch.max(probs, 1)
            accuracy.update(predicted, labels)
            f1.update(predicted, labels)
            # Record loss
            batch_val_loss.append(loss.detach().item())

    final_acc = accuracy.compute().item()
    final_f1 = f1.compute().item()
    return batch_val_loss, final_acc, final_f1

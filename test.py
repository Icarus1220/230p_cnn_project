import numpy as np
import torch
import time
import copy
import torch.nn as nn
from torchmetrics import Accuracy, F1Score
import pandas as pd

def test(model, criterion, test_loader, device: torch.device):
    fp = open('output.log', 'a+')
    print('Testing...', file=fp)
    fp.close()
    model.eval()
    accuracy = Accuracy(task='binary').to(device)
    f1 = F1Score(task='binary', num_classes=2).to(device)
    results_list = []
    with torch.no_grad():
        if isinstance(criterion, nn.CrossEntropyLoss):
            total_loss = 0.0
            results_list = []

            for iteration, (inputs, labels, paths) in enumerate(test_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Only forward pass
                probs = model(inputs)
                loss = criterion(probs, labels)
                total_loss += loss.item()

                prob_cls_1 = probs[:, 1]

                _, predicted = torch.max(probs, 1)

                accuracy.update(predicted, labels)
                f1.update(predicted, labels)

                for i in range(len(labels)):
                    results_list.append({
                        'label': labels[i].item(),
                        'predicted': predicted[i].item(), 
                        'prob_cls_1': prob_cls_1[i].cpu().numpy(),
                    })

            final_acc = accuracy.compute().item()
            final_f1 = f1.compute().item()
            average_loss = total_loss / len(test_loader)
            
            fp = open('inference.log', 'a+')
            print(f"The test accuracy is {final_acc:.4f}; F1 score is {final_f1:.4f}; Average Loss: {average_loss}.\n", file=fp)
            fp.close()
            print(f"The test accuracy is {final_acc:.4f}; F1 score is {final_f1:.4f}; Average Loss: {average_loss}.\n")
        else:
            # 原代码是分类，numeric output+MSE逻辑要重新写
            batch_test_acc = 0
            sum = 0
            output_list, path_list = [], []
            imgs = []
            for iteration, (inputs, labels, paths) in enumerate(test_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Only forward pass
                output = model.forward(inputs)
                # imgs.append([img1.cpu().numpy(), img2.cpu().numpy(), img3.cpu().numpy()])
                loss = criterion.forward(output, labels) # 这个应该就是loss了？TODO
                # prob = logit[:, 1].detach().cpu().numpy()
                output_list.append(output)
                path_list.extend(list(paths))
                # _, predicted = torch.max(logit, 1)
                # Number of correct predictions
                # batch_test_acc += (predicted == labels).sum()

                # batch_test_acc += (predicted == labels).sum()
    
    result_df = pd.DataFrame(results_list)
    return result_df
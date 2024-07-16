import torch.nn as nn
import torch
import torch.nn.functional as F

class SoftF1Loss(nn.Module):
    def __init__(self):
        super(SoftF1Loss, self).__init__()

    def forward(self, probs, labels):
        # Apply sigmoid to get probabilities
        #print(f"probs: {probs}")
        #print(f"labels: {labels}")
        #labels_one_hot = F.one_hot(labels, num_classes=probs.shape[1]).float()
        #print(f"labels after onehot: {labels_one_hot}")
        # Flatten tensors to 1D
        probs = probs[:,-1]
        #labels_one_hot = labels_one_hot.view(-1)

        
        # Calculate true positives, false positives, and false negatives
        tp = torch.sum(probs * labels)
        fp = torch.sum(probs * (1 - labels))
        fn = torch.sum((1 - probs) * labels)
        
        # Calculate Soft F1 score
        soft_f1 = 2 * tp / (2 * tp + fp + fn + 1e-7)  # Add a small value to avoid division by zero
        return 1 - soft_f1  # We return 1 - F1 score to minimize the loss

def loss(args):
    loss_lower = args.loss.lower()
    
    if loss_lower == 'ce':
        loss = nn.CrossEntropyLoss()
    elif loss_lower == 'mse':
        loss = nn.MSELoss()
    elif loss_lower == 'l1':
        loss = nn.L1Loss()
    elif loss_lower == 'f1':
        loss = SoftF1Loss()
    else:
        assert False and "Invalid optimizer"
    fp=open('output.log','a+')
    print(f'loss is {loss}', file=fp)
    fp.close()
    return loss
    
    

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import os
import time
import torch
from collections import Counter
import matplotlib.pyplot as plt

# from torchvision import transforms, datasets
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

from network import CNN_5, CNN_20, CNN_60, ResNet
from loss import loss
from optimizer import initialize_optimizer
from train_and_valid import train
from test import test
from plot import plot_loss_and_acc

from sklearn.model_selection import train_test_split
# from vit import VisionTransformer
# filter the warnings
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*")


def get_args_from_keyboard():
    parser = argparse.ArgumentParser('CNN Training', add_help=False)
    parser.add_argument('--trainset', default='train', type=str,
                        help='chootrse dataset (default: train)')
    parser.add_argument('--testset', default='test', type=str,
                        help='choose dataset (default: test)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size of the dataset default(128)')
    parser.add_argument('--epoch', type=int, default=10,
                        help='epochs of training process default(10)')
    parser.add_argument('--use_pretrained', type=int, default=0,
                        help='label whether to run new training or load trained model, default 0 (run training)')

    # Optimization parameters
    # At least need to specify task and loss (for BB-classifer-CrossEnropy or return-predictor-MSE)
    parser.add_argument('--task', type=int, default=0,
                        help='define what is the target label, 0 for BB, 1 for return, defualt is BB')
    parser.add_argument('--loss', type=str, default='ce',
                        help='define loss function (default: CrossEntropy)')
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adam"')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight decay (default: 0.01)')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate (default: 3e-4)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='choose running device (default: cpu)')
    parser.add_argument('--exp', type=str, default='debug',
                        help='choose using mode, (default: experiment mode)')
    parser.add_argument('--model', type=str, default='ResNet',
                        help='choosing the model from cnn_5, cnn_20, cnn_60 and ResNet (default: ResNet)')
    parser.add_argument('--infer', type=int, default=0,
                        help='if infer mode or not default(0)')
    parser.add_argument('--small_set', type=int, default=0,
                        help='using the small dataset or not default(0)')
    return parser

    
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, task=0, transform=None): # 0 for BB, 1 for return, default is BB
        self.root_dir = root_dir
        self.transform = transform
        self.task = int(task)
        self.images = [f for f in os.listdir(root_dir) if f.endswith('png') and f.split('_')[3+self.task] != 'nan']
        self.cached_images = {}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        if img_name in self.cached_images:
            image = self.cached_images[img_name]
        else:
            img_path = os.path.join(self.root_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            self.cached_images[img_name] = image

        info = img_name.split('_')
        label_dict = {
            "sell": 0,
            "buy": 1
        }

        labels = label_dict[info[3+self.task]]

        if self.transform:
            image = self.transform(image)

        return image, labels, img_name


def main(args):
    transform_train = transforms.Compose([
        # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        # transforms.Resize(img_resize),
        # transforms.CenterCrop(img_resize),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # Only perform deterministic operations on image augmentation on the test set
    transform_test = transforms.Compose([
        # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        # transforms.Resize(img_resize),
        # transforms.CenterCrop(img_resize),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Set up the training device
    device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #torch.device(args.device)
    # Choose model, should be consistent with input picture size
    model_dict = {
        'cnn_5': CNN_5(),
        'cnn_20': CNN_20(),
        'cnn_60': CNN_60(),
        'ResNet': ResNet()
    }
    model = model_dict[args.model]
    model = model.to(device)

    fp = open('output.log', 'a+')
    print(f"using {device} device", file=fp)
    print(f"trainset:{args.trainset}", file=fp)
    print(f"testset:{args.testset}", file=fp)
    print(f"model:{args.model}", file=fp)
    print(f"using {device} device")
    print(f"model:{args.model}")
    fp.close()

    # train test is predefined to make test_set time stamp is AFTER train and valid
    train_valid_set = CustomImageDataset(root_dir=os.path.join(current_dir, args.trainset), task=int(args.task), transform=transform_train)
    test_set = CustomImageDataset(root_dir=os.path.join(current_dir, args.testset), task=int(args.task), transform=transform_test)
    
    train_size = int(0.89 * len(train_valid_set))
    valid_size = len(train_valid_set) - train_size
    torch.manual_seed(123)
    train_set, valid_set = random_split(train_valid_set, [train_size, valid_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=2)

    criterion = loss(args)
    optimizer = initialize_optimizer(args, model)

    if args.use_pretrained == 1:
        model.load_state_dict(torch.load('my_model.pth'))

    model_trained, best_model, train_los, train_acc, train_f1, val_los, val_acc, val_f1, early_stop_epoch = train(
            model=model,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=valid_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            device=device,
            max_epoch=args.epoch,
            disp_freq=100
    )
    
    test_result_df = test(
        model=best_model,
        criterion=criterion,
        test_loader=test_loader,
        device=device
    )

    test_result_df.to_csv(os.path.join(current_dir, 'test_output.csv'))

    # Saving loss and accuracy
    fp = open('output.log', 'a+')
    print(f'Drawing...', file=fp)
    print(f'Drawing...')

    # Plotting loss and accuracy
    suffix = args.trainset + '_' + args.model + '_lr' + str(args.lr) + '_' + str(
        args.opt) + '_wd' + str(args.weight_decay) + '_epoch' + str(args.epoch) + '.png'
    path = ['loss_' + suffix, 'acc_' + suffix]
    prefix = args.trainset + '_' + args.model + '_lr' + str(args.lr) + '_' + str(
        args.opt) + '_wd' + str(args.weight_decay) + '_epoch' + str(args.epoch)
    if args.use_pretrained != 1:
        torch.save(best_model.state_dict(), prefix + '_model.pth')
        plot_loss_and_acc({'model_train': [train_los, train_acc]}, {'model_val':  [val_los, val_acc]}, path)
        print("Draw Done", file=fp)
        print("Draw Done")
    else:
        torch.save(best_model.state_dict(), prefix + '_updated_model.pth')
        plot_loss_and_acc({'model_train': [train_los, train_acc]}, {'model_val':  [val_los, val_acc]}, path)
        print("Draw Done", file=fp)
        print("Draw Done")
    fp.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Problem solver', parents=[get_args_from_keyboard()])
    args = parser.parse_args()
    main(args)
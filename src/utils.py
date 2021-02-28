from global_vars import DEVICE, OUTPUT_DIR, LR

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim
import numpy as np
import torch

from typing import Tuple
import pickle
import time
import copy
import os


# modified function from
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
def train(
        model: nn.Module,
        dataloaders: dict,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        num_epochs: int = 25,
        threshold: float = 0.5,
) -> Tuple[nn.Module, list]:
    since = time.time()
    val_acc_history = list()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 15)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for sample in dataloaders[phase]:
                inputs = sample['image'].to(DEVICE)
                labels = sample['label'].to(DEVICE)

                # reset the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs).to(DEVICE)
                    loss = criterion(outputs, labels).to(DEVICE)
                    preds = (outputs >= threshold).float()
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc)
            )

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        torch.save(best_model_wts, os.path.join(OUTPUT_DIR, 'resnet50.pth'))
        update_learn_rate(epoch, LR, optimizer)
        # save accuracy history
        open_file = open(
            os.path.join(OUTPUT_DIR, 'val_acc_history.pkl'),
            'wb'
        )
        pickle.dump(val_acc_history, open_file)
        open_file.close()
        print()

    # Print final statistics
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60)
    )
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model parameters
    model.load_state_dict(best_model_wts)

    return model, val_acc_history


def update_learn_rate(epoch: int, lr: float, optimizer: optim.Optimizer) -> None:
    if epoch <= 5:
        pass
    elif epoch <= 10:
        lr *= 1e-1
    else:
        lr *= 1e-2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def compute_statistics(dataset_path: str) -> Tuple[list, list]:
    dataset = datasets.ImageFolder(
        root=dataset_path,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=len(dataset),
        shuffle=False
    )
    data = next(iter(dataloader))[0]
    mean = np.mean(data.numpy(), axis=(0, 2, 3))
    std = np.std(data.numpy(), axis=(0, 2, 3))
    return mean, std


def set_parameter_requires_grad(model, feature_extracting) -> None:
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

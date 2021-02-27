from global_vars import DEVICE, EPOCHS, LR, BATCH_SIZE, THRESHOLD, SEED
from dataset import CustomDataset
from utils import set_parameter_requires_grad, train

from torch.utils.data import random_split, DataLoader
from torchvision import transforms, models
from torch import nn, optim
import pandas as pd
import numpy as np
import torch


# for reproducibility
torch.manual_seed(SEED)


# filter labels
df1 = pd.read_csv('../input/georges.csv', header=None, squeeze=True)
df1 = df1.apply(lambda x: x.split('/')[-1])
df1.drop_duplicates(inplace=True)
df1 = df1.to_frame()
df1.columns = ['name']
df1['label'] = np.ones(df1.shape[0], dtype=int)

df2 = pd.read_csv('../input/non_georges.csv', header=None, squeeze=True)
df2 = df2.apply(lambda x: x.split('/')[-1])
df2.drop_duplicates(inplace=True)
df2 = df2.to_frame()
df2.columns = ['name']
df2['label'] = np.zeros(df2.shape[0], dtype=int)

df = pd.concat([df1, df2], ignore_index=True)
df.drop_duplicates(subset='name', inplace=True, ignore_index=True)
print('Label fractions: \n{}'.format(df.label.value_counts(normalize=True)))

# cleanup
del df1, df2

# Create dataset
torch.manual_seed(11)
transform = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

images_path = '../input/images'
dataset = CustomDataset(images_path, df, transform)

train_length = int(len(dataset) * 0.75)
valid_length = len(dataset) - train_length

dataset_train, dataset_valid = random_split(dataset, [train_length, valid_length])
train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset_valid, batch_size=BATCH_SIZE, shuffle=True)
dataloaders = dict(train=train_loader, val=valid_loader)

# load model
feature_extract = True
model = models.resnet50(pretrained=True)
set_parameter_requires_grad(model, feature_extract)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)
model.fc = nn.Sequential(
    model.fc,
    nn.Sigmoid()
)

print('Device:', DEVICE)
model = model.to(DEVICE)

# load params
model.load_state_dict(torch.load('../params/resnet50_old.pth'))

params_to_update = model.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("\t", name)
print()

criterion = nn.BCELoss()
optimizer = optim.Adam(params=params_to_update, lr=LR)

# train
model, hist = train(
    model=model,
    dataloaders=dataloaders,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=EPOCHS,
    threshold=THRESHOLD
)

# ======================
# Import Libs
# ======================

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
# from copy import deepcopy
import matplotlib.pyplot as plt
import time
import datetime

import config
from data_loader import RS21BD, get_training_augmentation, get_val_augmentation
from model import NestedUNet


model_dir = config.model_dir
graph_dir = config.graph_dir

x_train_dir = config.x_train_dir
y_train_dir = config.y_train_dir
x_val_dir = config.x_val_dir
y_val_dir = config.y_val_dir

batch_size = config.batch_size
img_size = config.img_size
epochs = config.epochs
in_ch = config.in_ch
out_ch = config.out_ch
model_file = config.model_file

device = config.device

CLASSES = config.CLASSES

model_date = datetime.datetime.now().strftime('%m%d-%H%M')
model_dir = os.path.join(model_dir, model_date)
if os.path.isdir(model_dir):
    while True:
        yn = input("Overwrite %s? [y/n]: " % model_dir)
        if yn == 'y':
            break
        elif yn == 'n':
            print("Aborting..")
            exit(1)
os.mkdir(model_dir)

# ======================
# Function for training a model
# ======================

# modified from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

def train_model(model, criterion, optimizer, loader, n_epochs=25):
    since = time.time()

    # best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0

    train_accs = []
    val_accs = []

    for epoch in range(1, n_epochs + 1):
        print('\nEpoch {}/{}'.format(epoch, n_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for x, y, f in loader[phase]:

                img_in = x.to(device).float()
                img_out = y.to(device).float()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model.forward(img_in)
                    loss = 0
                    for output in outputs:
                        loss += criterion(output, img_out)
                    loss /= len(outputs)
                    pred = (output[-1] > 0.5).long()

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * img_in.size(0)
                running_corrects += torch.sum(pred == img_out)

            epoch_loss = running_loss / len(loader[phase].dataset)

            if phase == 'train':
                epoch_acc = running_corrects.double() / (len(loader[phase].dataset) * img_size * img_size)
                train_accs.append(epoch_acc.cpu())
            else:
                epoch_acc = running_corrects.double() / (len(loader[phase].dataset) * img_size * img_size)
                val_accs.append(epoch_acc.cpu())

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                # best_model_wts = copy.deepcopy(model.state_dict())
                # save model
                torch.save(model, os.path.join(model_dir, model_file))
                print('Model saved!')


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Epoch {}/{}'.format(best_epoch, n_epochs))
    print('Best val Acc: {:4f}'.format(best_acc))

    plt.plot(np.array(train_accs), 'r', label="train")
    plt.plot(np.array(val_accs), 'b', label="val")
    plt.grid()
    plt.legend()
    # plt.show()
    plt.savefig(graph_dir + 'accuracy.png')

    # load best model weights
    # model.load_state_dict(best_model_wts)
    # return model

# ======================
# Dataset
# ======================

# CLASSES =  ["building"]

train_dataset = RS21BD(
    x_train_dir,
    y_train_dir,
    augmentation=get_training_augmentation(),
    classes=CLASSES,
    )

val_dataset = RS21BD(
    x_val_dir,
    y_val_dir,
    augmentation=get_val_augmentation(),
    classes=CLASSES,
    )

dataloaders = {'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1),
              'val': DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)}

# ======================
# Model set up & traning
# ======================
criterion = torch.nn.BCEWithLogitsLoss()
model = NestedUNet(in_ch, out_ch).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
train_model(model, criterion, optimizer, dataloaders, epochs)


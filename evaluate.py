
import os
import torch
import torch.nn as nn
import pathlib
import cv2
from tqdm import tqdm
import argparse

import config
from data_loader import RS21BD, get_val_augmentation
from loss import IoULoss
import crf

parser = argparse.ArgumentParser()
parser.add_argument('-y', '--yes', default=False, action='store_true', help='overwrite result or not')
parser.add_argument('-c', '--crf', default=False, action='store_true', help='post-process (fully connected CRF) or not')
parser.add_argument('-d', '--denoising', default=False, action='store_true', help='post-process (fully connected crf and denoising) or not')
args = parser.parse_args()

data_dir = config.data_dir
model_dir = config.model_dir
pred_dir = config.pred_dir

model_date = config.model_date

x_val_dir = config.x_val_dir
y_val_dir = config.y_val_dir

model_file = config.model_file

device = config.device

CLASSES = config.CLASSES


model_dir = os.path.join(model_dir, model_date)
pred_dir = os.path.join(pred_dir, model_date)
if not(args.yes) and os.path.isdir(pred_dir):
    while True:
        yn = input("Overwrite %s? [y/n]: " % pred_dir)
        if yn == 'y':
            break
        elif yn == 'n':
            print("Aborting..")
            exit(1)
os.makedirs(pred_dir, exist_ok=True)

val_dataset = RS21BD(
    x_val_dir,
    y_val_dir,
    augmentation=get_val_augmentation(),
    classes=CLASSES,
)

# load best saved checkpoint
# criterion = torch.nn.BCEWithLogitsLoss()
criterion = IoULoss()
model = torch.load(os.path.join(model_dir, model_file))

# ======================
# Visualize results for val set
# ======================

model.eval()

post_processor = crf.DenseCRF(
    iter_max=3,    # 10
    pos_xy_std=3,   # 3
    pos_w=5,        # 3
    bi_xy_std=16,  # 121, 140
    bi_rgb_std=3,   # 5, 5
    bi_w=9,         # 4, 5

    # iter_max=3,    # 10
    # pos_xy_std=3,   # 3
    # pos_w=5,        # 3
    # bi_xy_std=16,  # 121, 140
    # bi_rgb_std=3,   # 5, 5
    # bi_w=9,         # 4, 5

    # iter_max=3,    # 10
    # pos_xy_std=5,   # 3
    # pos_w=5,        # 3
    # bi_xy_std=16,  # 121, 140
    # bi_rgb_std=3,   # 5, 5
    # bi_w=9,         # 4, 5

    # iter_max=3,    # 10
    # pos_xy_std=5,   # 3
    # pos_w=5,        # 3
    # bi_xy_std=16,  # 121, 140
    # bi_rgb_std=5,   # 5, 5
    # bi_w=7,         # 4, 5

    # iter_max=3,    # 10
    # pos_xy_std=3,   # 3
    # pos_w=3,        # 3
    # bi_xy_std=25,  # 121, 140
    # bi_rgb_std=5,   # 5, 5
    # bi_w=5,         # 4, 5
)

def filtering(input, color=0, area=5):
    seen = (input == color)
    for i in tqdm(range(input.shape[0])):
        for j in range(input.shape[1]):
            if seen[i][j]: continue
            cells = set({(i, j)})
            que = set({(i, j)})
            while len(que) > 0:
                ii, jj = que.pop()
                for iii, jjj in set({(ii-1, jj), (ii, jj-1), (ii+1, jj), (ii, jj+1)}):
                    if 0 <= iii < input.shape[0] and 0 <= jjj < input.shape[1] and not(seen[iii][jjj]):
                        seen[iii][jjj] = True
                        if len(cells) <= area: cells.add((iii, jjj))
                        que.add((iii, jjj))

            if len(cells) <= area:
                while len(cells) > 0:
                    ii, jj = cells.pop()
                    input[ii][jj] = color

    return input


losses = 0
max_loss = 0
max_path = 0
min_loss = 1000
min_path = 0
sig = nn.Sigmoid()
th = 0.4

def post_process(input, prediction):
    if args.filtering:
        prediction = post_processor(input.transpose(1, 2, 0).astype('uint8'), sig(prediction).detach().cpu().numpy())
        prediction = torch.from_numpy(prediction).to(device)
        return filtering(filtering((prediction > th), color=0), color=1).long()
    elif args.postprocess:
        prediction = post_processor(input.transpose(1, 2, 0).astype('uint8'), sig(prediction).detach().cpu().numpy())
        prediction = torch.from_numpy(prediction).to(device)
        return (prediction > th).long()
    else:
        return (prediction > 0).long()


for n in range(len(val_dataset)):
    input, output, rgb_path = val_dataset[n]
    x_tensor = torch.from_numpy(input).to(device).unsqueeze(0)

    _, _, _, prediction = model(x_tensor)
    prediction = post_process(input, prediction[0])

    loss = criterion(prediction, torch.from_numpy(output).to(device))
    print('Loss for {}: {:.4f}'.format(os.path.basename(rgb_path), loss))
    losses += loss
    if loss > max_loss:
        max_loss = loss
        max_path = rgb_path
    elif loss < min_loss:
        min_loss = loss
        min_path = rgb_path

    prediction = prediction.squeeze().cpu().numpy().round()

    # prediction = (prediction > 0).long().squeeze().cpu().numpy().round()

    output = output.squeeze()
    rgb = input.transpose(1, 2, 0)

    # plt.figure(figsize=(12,12))
    # plt.subplot(1,3,1)
    # plt.imshow(rgb[:,:,[2,1,0]]/255)
    # plt.subplot(1,3,2)
    # plt.imshow(prediction)
    # plt.subplot(1,3,3)
    # plt.imshow(output)
    # plt.show()

    rgb_fname = pathlib.Path(rgb_path).stem
    prediction = (prediction*255).astype('uint8')
    # prediction = cv2.resize(prediction, (512, 512), interpolation=cv2.INTER_CUBIC)
    # save prediction with the original image size
    cv2.imwrite(os.path.join(pred_dir, rgb_fname + '.png'), prediction)

print('Ave Loss: {:.4f}'.format(losses / len(val_dataset)))
print('Max Loss: {}, {:.4f}'.format(os.path.basename(max_path), max_loss))
print('Min Loss: {}, {:.4f}'.format(os.path.basename(min_path), min_loss))

# ======================
# Visualize results for test set
# ======================
from glob import glob

x_test_dir = os.path.join(data_dir, 'test_img')
test_files = glob(x_test_dir+'/*.png')

model.eval()

for f in test_files:
    image = cv2.imread(f, cv2.IMREAD_COLOR)
    input = image.transpose(2, 0, 1).astype('float32')
    x_tensor = torch.from_numpy(input).to(device).unsqueeze(0)

    _, _, _, prediction = model(x_tensor)
    prediction = post_process(input, prediction[0])

    prediction = prediction.squeeze().cpu().numpy().round()

    rgb = input.transpose(1, 2, 0)

    # plt.figure(figsize=(12,12))
    # plt.subplot(1,2,1)
    # plt.imshow(rgb[:,:,[2,1,0]]/255)
    # plt.subplot(1,2,2)
    # plt.imshow(prediction)
    # plt.show()

    rgb_fname = pathlib.Path(f).stem
    prediction = (prediction*255).astype('uint8')
    prediction = cv2.resize(prediction, (512, 512), interpolation=cv2.INTER_CUBIC)
    # save prediction with the original image size
    cv2.imwrite(os.path.join(pred_dir, rgb_fname + '.png'), prediction)

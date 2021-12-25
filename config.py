import os
import torch

# ======================
# Path
# ======================

data_dir = 'data'
model_dir = 'model/'
graph_dir = 'graph/'
pred_dir = 'result/'

# model_date = 'sample'
# model_date = '1223-2329-72'
# model_date = '1223-2329-mask'
# model_date = '1225-1322'
# model_date = '1225-1336'
model_date = '1225-1425'

x_train_dir = os.path.join(data_dir, 'train_img')
y_train_dir = os.path.join(data_dir, 'train_label')

x_val_dir = os.path.join(data_dir, 'val_img')
y_val_dir = os.path.join(data_dir, 'val_label')

# ======================
# Config
# ======================

batch_size = 8
epochs = 100
# img_size = 256
img_size = 512
in_ch = 3 # number of input channels (RGB)
out_ch = 1 # number of output channels (1 for binary classification)
model_name = 'unet++'
model_file = "%s_E%d_B%d.pth"%(model_name, epochs, batch_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CLASSES =  ["building"]

import os
import numpy as np
import sklearn.metrics as skm
import argparse
from datetime import datetime

from tsai.all import *
from tsai.data.transforms import TSRandomResizedCrop, TSRandomCropPad

from fastai.callback.all import *

from utils.loss.FocalLoss import FocalLoss


timestamp = datetime.now().strftime("%Y|%m|%d|%H|%M")
parser = argparse.ArgumentParser(description='Train TST.')
parser.add_argument('--exp_name', default="stage7_focalloss_lrmax1e-3", type=str, help='unique name of this exp & used for saving.')
parser.add_argument('--root_data', default="data/dataset_TimeSeries_SlideWindow_cls_onlyEpi_v3", type=str, help='root of dataset for training')
parser.add_argument('--log_folder', default="exp_logs", type=str, help='folder for saving logs and args info.')
parser.add_argument('--load_model_name', default="stage65_focalloss_lrmax5e-3", type=str, help='pre-trained or resumed model\'s name.')

parser.add_argument('--lr_max', default=1e-3, type=float, help='learning rate')
parser.add_argument('--bs', default=1024 * 8, type=int, help='batch size')
parser.add_argument('--dropout', default=0.3, type=float, help='p dropout')
parser.add_argument('--n_epoch', default=60, type=int, help='epoch number')

parser.add_argument('--data_length', default=40, type=int, help='data length of the dataset.')

args = parser.parse_args()
print(args)

os.makedirs(args.log_folder, exist_ok=True)
os.makedirs(os.path.join(args.log_folder, args.exp_name), exist_ok=True)
EXP_NAME = args.exp_name
with open(os.path.join(args.log_folder, args.exp_name, "args_info.txt"), "w") as file:
    file.write(f"Time: {timestamp}\n")
    file.write(f"Exp name: {EXP_NAME}\n")
    file.write("Args:\n")
    for arg in vars(args):
        file.write(f"{arg}: {getattr(args, arg)}\n")


N_EPOCH = args.n_epoch
BS = args.bs
DROP_OUT = args.dropout
LR_MAX = args.lr_max
DATA_LENGTH = args.data_length
root_dataset = args.root_data

# load training data
X_train = np.load(os.path.join(root_dataset, "train_data.npy"))
y_train = np.load(os.path.join(root_dataset, "train_label.npy"))
X_val = np.load(os.path.join(root_dataset, "val_data.npy"))
y_val = np.load(os.path.join(root_dataset, "val_label.npy"))
X, y, splits = combine_split_data(
    [X_train.astype(float), X_val.astype(float)],
    [y_train.astype(float), y_val.astype(float)]
)

# Data augmentation
aug = [
    TSMagScale(p=0.2, magnitude=0.2),
    TSMagWarp(p=0.5, magnitude=0.1),
    TSTimeWarp(p=0.5, magnitude=0.1),
    TSMagMulNoise(magnitude=0.05),
    # TSBlur(magnitude=0.1),
    # TSSmooth(magnitude=0.1),
    # TSInputDropout(magnitude=0.1)
    TSRandomResizedCrop(size=DATA_LENGTH, scale=(0.05, 0.95)),
    # TSRandomCropPad(magnitude=0.2),
    # TSIdentity
]
# batch_tfms = [TSStandardize(), TSRandomResizedCrop(size=DATA_LENGTH, scale=(0.05, 0.95)), TSRandomCropPad(magnitude=0.2), *aug]
# batch_tfms = [TSStandardize(), TSRandomResizedCrop(size=DATA_LENGTH, )]
batch_tfms = [TSStandardize(), *aug]

tfms = [None, [Categorize()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
dls = TSDataLoaders.from_dsets(
    dsets.train,
    dsets.valid,
    bs=[BS, BS * 2],
    batch_tfms=batch_tfms,
    num_workers=10
)

model = TST(dls.vars, dls.c, dls.len, dropout=DROP_OUT)
learn = Learner(
    dls,
    model,
    metrics=[accuracy, BalancedAccuracy(), F1Score()],
    cbs=[CSVLogger(fname=os.path.join(args.log_folder, args.exp_name, 'log.csv'))]  # Add CSVLogger here
)
learn.loss_func = FocalLoss(alpha=0.25, gamma=2.0)

learn.load(args.load_model_name)
learn.fit_one_cycle(N_EPOCH, lr_max=LR_MAX)
learn.save(EXP_NAME)
learn.save_all(path='export', dls_fname='dls', model_fname='model', learner_fname='learner')
learn.show_results()

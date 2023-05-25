import os
import numpy as np
from tsai.all import *
import sklearn.metrics as skm
from utils.loss.FocalLoss import FocalLoss

BS = 512
DROP_OUT = .3

root_dataset = "data/dataset_TimeSeries_SlideWindow_cls_onlyEpi_v2"
# shape: n_data, n_channels, data_length
X_train = np.load(os.path.join(root_dataset, "train_data.npy"))
y_train = np.load(os.path.join(root_dataset, "train_label.npy"))
X_val = np.load(os.path.join(root_dataset, "val_data.npy"))
y_val = np.load(os.path.join(root_dataset, "val_label.npy"))
#
# X_val = np.load("data/dataset_TimeSeries_SlideWindow_cls_onlyEpi_v2/val_data.npy")
# y_val = np.load("data/dataset_TimeSeries_SlideWindow_cls_onlyEpi_v2/val_label.npy")
# X_test = np.load("data/dataset_TimeSeries_SlideWindow_cls_onlyEpi_v2/test_data.npy")
# y_test = np.load("data/dataset_TimeSeries_SlideWindow_cls_onlyEpi_v2/test_label.npy")

X, y, splits = combine_split_data(
    [X_train.astype(float), X_val.astype(float)],
    [y_train.astype(float), y_val.astype(float)]
)

tfms = [None, [Categorize()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)

dls = TSDataLoaders.from_dsets(
    dsets.train,
    dsets.valid,
    bs=[BS, BS * 1],
    batch_tfms=[TSStandardize()],
    num_workers=10
)

# model = InceptionTime(dls.vars, dls.c)
model = TST(dls.vars, dls.c, dls.len, dropout=DROP_OUT)
learn = Learner(
    dls,
    model,
    metrics=[accuracy, BalancedAccuracy(), F1Score()]
)
learn.loss_func = FocalLoss(alpha=0.25, gamma=2.0)

# learn.save('stage0')
# learn.save('stage1')
# learn.load('stage2')
learn.load('stage4')
learn.fit_one_cycle(25, lr_max=6.30957365501672e-04)
learn.save('stage5_focalloss_lrmax6e-4')
learn.save_all(path='export', dls_fname='dls', model_fname='model', learner_fname='learner')
learn.show_results()


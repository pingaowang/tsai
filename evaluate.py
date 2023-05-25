import os
import numpy as np
from fastai.learner import Learner, load_learner
from tsai.all import *

# shape: n_data, n_channels, data_length
# X_train = np.load("data/dataset_TimeSeries_SlideWindow_cls_onlyEpi_v2/train_data.npy")
# y_train = np.load("data/dataset_TimeSeries_SlideWindow_cls_onlyEpi_v2/train_label.npy")
# X_val = np.load("data/dataset_TimeSeries_SlideWindow_cls_onlyEpi_v2/val_data.npy")
# y_val = np.load("data/dataset_TimeSeries_SlideWindow_cls_onlyEpi_v2/val_label.npy")
X_test = np.load("data/dataset_TimeSeries_SlideWindow_cls_onlyEpi_v2/test_data.npy").astype(float)
y_test = np.load("data/dataset_TimeSeries_SlideWindow_cls_onlyEpi_v2/test_label.npy").astype(float)

np.save("X_for_init_dls.npy", X_test[:50])
np.save("y_for_init_dls.npy", y_test[:50])

# X, y, splits = combine_split_data(
#     [X_train.astype(float), X_val.astype(float)],
#     [y_train.astype(float), y_val.astype(float)]
# )

tfms = [None, [Categorize()]]
dsets = TSDatasets(X_test[:50], y_test[:50], tfms=tfms, splits=None, inplace=True)
dls = TSDataLoaders.from_dsets(
    dsets.train,
    dsets.valid,
    bs=[16, 32],
    batch_tfms=[TSStandardize()],
    num_workers=0
)

model = TST(2, 2, 40)
learn = Learner(
    dls,
    model,
    metrics=[accuracy, BalancedAccuracy(), F1Score()]
)

# learn = load_learner("export/model.pth")
learn.load("stage2")

test_probas, test_targets, test_preds = learn.get_X_preds(
    X=X_test,
    y=y_test,
    with_decoded=True,
    bs=16
)

print(
    "test_probas: \n {} \n test_targets: \n {} \n test_preds: \n {} \n".format(
        test_probas, test_targets, test_preds
    )
)
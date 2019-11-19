# Copyright (C) 2019, Francois Berenger
# Yamanishi laboratory,
# Department of Bioscience and Bioinformatics,
# Faculty of Computer Science and Systems Engineering,
# Kyushu Institute of Technology,
# 680-4 Kawazu, Iizuka, Fukuoka, 820-8502, Japan.

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tempfile
import numpy as np
import deepchem as dc

def tmp_filename():
    fn = tempfile.NamedTemporaryFile()
    return fn.name

metric = dc.metrics.Metric(
    dc.metrics.roc_auc_score, np.mean, mode="classification")

def train_early_stop(max_epochs, model, train, valid, transformers):
    restore = False # force 1st model to be trained from scratch
    best_auc = 0.0
    best_epoch = -1
    opt_miss = 0
    epoch2checkpoint = {}
    for i in range(max_epochs):
        model.fit(train, nb_epoch=1, restore=restore, checkpoint_interval=0)
        # save model to file
        model.save_checkpoint(max_checkpoints_to_keep=10)
        checkpoints = model.get_checkpoints()
        nb_checkpoints = len(checkpoints)
        last_checkpoint = checkpoints[nb_checkpoints - 1]
        restore = True # next fits will continue training the same model
        e1 = model.evaluate(train, [metric], transformers)
        e2 = model.evaluate(valid, [metric], transformers)
        tAUC = e1["mean-roc_auc_score"]
        vAUC = e2["mean-roc_auc_score"]
        epoch2checkpoint[i] = last_checkpoint
        print("step: %d traAUC: %.3f valAUC: %.3f" % (i, tAUC, vAUC))
        if vAUC > best_auc:
            best_auc = vAUC
            best_epoch = i
            opt_miss = 0
        else:
            opt_miss += 1
        if opt_miss == 5:
            break # model doesn't improve anymore
    # retrieve model that gave best vAUC
    model.restore(checkpoint=epoch2checkpoint[best_epoch])
    return model

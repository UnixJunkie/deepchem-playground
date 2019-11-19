#!/usr/bin/env python3

# Copyright (C) 2019, Francois Berenger
# Yamanishi laboratory,
# Department of Bioscience and Bioinformatics,
# Faculty of Computer Science and Systems Engineering,
# Kyushu Institute of Technology,
# 680-4 Kawazu, Iizuka, Fukuoka, 820-8502, Japan.

# inspired by DeepChem's "Graph Convolutions For Tox21" tutorial
# cf.
# https://deepchem.io/docs/notebooks/graph_convolutional_networks_for_tox21.html

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import deepchem as dc
import sys
import utils
from deepchem.models.tensorgraph.models.graph_models import GraphConvModel

# Load Tox21 dataset
tox21_tasks, tox21_datasets, transformers = \
    dc.molnet.load_tox21(featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = tox21_datasets

if __name__ == '__main__':
    # CLI parsing
    parser = argparse.ArgumentParser(
        description = "train/test a GraphConv model on Tox21")
    parser.add_argument("-n", metavar = "max_epochs", type = int,
                        default = 100, dest = "max_epochs",
                        help = "max training epochs (default=100)")
    # # show help in case user has no clue of what to do
    # if len(sys.argv) == 1:
    #     parser.print_help(sys.stderr)
    #     sys.exit(1)
    args = parser.parse_args()
    max_epochs = args.max_epochs
    model = GraphConvModel(
        len(tox21_tasks), batch_size=50, mode='classification')
    # Set nb_epoch higher for better results.
    model = utils.train_early_stop(
        max_epochs, model, train_dataset, valid_dataset, transformers)
    print("Evaluating model")
    train_scores = model.evaluate(train_dataset, [utils.metric], transformers)
    valid_scores = model.evaluate(valid_dataset, [utils.metric], transformers)
    test_scores = model.evaluate(test_dataset, [utils.metric], transformers)
    print("traAUC: %.3f" % train_scores["mean-roc_auc_score"])
    print("valAUC: %.3f" % valid_scores["mean-roc_auc_score"])
    print("tesAUC: %.3f" % test_scores["mean-roc_auc_score"])

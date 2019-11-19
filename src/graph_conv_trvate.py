#!/usr/bin/env python3

# Copyright (C) 2019, Francois Berenger
# Yamanishi laboratory,
# Department of Bioscience and Bioinformatics,
# Faculty of Computer Science and Systems Engineering,
# Kyushu Institute of Technology,
# 680-4 Kawazu, Iizuka, Fukuoka, 820-8502, Japan.

# inspired by DeepChem's "Graph Convolutions For Tox21" tutorial

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import random
import sys
import utils
import numpy as np
import deepchem as dc
from deepchem.models.tensorgraph.models.graph_models import GraphConvModel

def task_name(l):
    return l.split(',')[0]

if __name__ == '__main__':
    # CLI parsing
    parser = argparse.ArgumentParser(
        description = "train/test a GraphConv model on given dataset")
    parser.add_argument("-tr", metavar = "train_csv", dest = "train_csv")
    parser.add_argument("-va", metavar = "valid_csv", dest = "valid_csv")
    parser.add_argument("-te", metavar = "test_csv", dest = "test_csv")
    parser.add_argument("-d", metavar = "dropout_rate", dest = "dropout_rate",
                        type = float, default = 0.0)
    # show help in case user has no clue of what to do
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    train_csv = args.train_csv
    valid_csv = args.valid_csv
    test_csv = args.test_csv
    dropout_r = args.dropout_rate
    # training set header
    train_lines = open(train_csv, 'r').readlines()
    train_header_line = train_lines[0]
    target = task_name(train_header_line)
    # validation set header
    valid_lines = open(valid_csv, 'r').readlines()
    valid_header_line = valid_lines[0]
    valid_target = task_name(valid_header_line)
    # test set header
    test_lines = open(test_csv, 'r').readlines()
    test_header_line = test_lines[0]
    test_target = task_name(test_header_line)
    assert(target == valid_target and
           target == test_target)
    # load CSV files
    graph_conv = dc.feat.ConvMolFeaturizer()
    loader = dc.data.CSVLoader(
        tasks = [target], smiles_field = "smiles", featurizer = graph_conv)
    raw_train_set = loader.featurize(train_csv, shard_size=8192)
    raw_valid_set = loader.featurize(valid_csv, shard_size=8192)
    raw_test_set = loader.featurize(test_csv, shard_size=8192)
    transformer = dc.trans.BalancingTransformer(transform_w=True,
                                                dataset=raw_train_set)
    transformers = [transformer]
    train_dataset = transformer.transform(raw_train_set)
    valid_dataset = transformer.transform(raw_valid_set)
    test_dataset = transformer.transform(raw_test_set)
    model = GraphConvModel(1, batch_size=50, mode='classification',
                           dropout=dropout_r)
    max_epochs = 100
    model = utils.train_early_stop(
        max_epochs, model, train_dataset, valid_dataset, transformers)
    print("Evaluating model")
    train_scores = model.evaluate(train_dataset, [utils.metric], transformers)
    valid_scores = model.evaluate(valid_dataset, [utils.metric], transformers)
    test_scores = model.evaluate(test_dataset, [utils.metric], transformers)
    print("traAUC: %.3f" % train_scores["mean-roc_auc_score"])
    print("valAUC: %.3f" % valid_scores["mean-roc_auc_score"])
    print("tesAUC: %.3f" % test_scores["mean-roc_auc_score"])

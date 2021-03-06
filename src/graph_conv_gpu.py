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
import tensorflow as tf
from tensorflow.python.client import device_lib 
import numpy as np
import deepchem as dc
from deepchem.models.tensorgraph.models.graph_models import GraphConvModel

def task_name(l):
    return l.split(',')[0]

# header line must have been removed before
def train_test_split(p, l):
    n = len(l)
    random.shuffle(l)
    m = round(n * p)
    train = l[0:m]
    test = l[m:]
    assert(n == len(train) + len(test))
    return (train, test)

def train_valid_test_split(p, l):
    train, rest = train_test_split(p, l)
    valid, test = train_test_split(0.5, rest)
    assert(len(l) == len(train) + len(valid) + len(test))
    return (train, valid, test)

def write_to_file(fn, header_line, data_lines):
    out = open(fn, 'w')
    out.write(header_line)
    for l in data_lines:
        out.write(l)
    out.close()

if __name__ == '__main__':
    # print GPU info
    print(device_lib.list_local_devices())
    # CLI parsing
    parser = argparse.ArgumentParser(
        description = "train/test a GraphConv model on given dataset")
    parser.add_argument("-i", metavar = "input_csv", dest = "input_csv")
    # show help in case user has no clue of what to do
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    input_csv = args.input_csv
    # split input file into train/valid/test
    all_lines = open(input_csv, 'r').readlines()
    header_line = all_lines[0]
    target = task_name(header_line)
    data_lines = all_lines[1:]
    (train, valid, test) = train_valid_test_split(0.8, data_lines)
    train_fn = utils.tmp_filename()
    valid_fn = utils.tmp_filename()
    test_fn = utils.tmp_filename()
    write_to_file(train_fn, header_line, train)
    write_to_file(valid_fn, header_line, valid)
    write_to_file(test_fn, header_line, test)
    print("train: " + train_fn)
    print("valid: " + valid_fn)
    print("test: " + test_fn)
    # load CSV files
    graph_conv = dc.feat.ConvMolFeaturizer()
    loader = dc.data.CSVLoader(
        tasks = [target], smiles_field = "smiles", featurizer = graph_conv)
    raw_train_set = loader.featurize(train_fn, shard_size=8192)
    raw_valid_set = loader.featurize(valid_fn, shard_size=8192)
    raw_test_set = loader.featurize(test_fn, shard_size=8192)
    transformer = dc.trans.BalancingTransformer(transform_w=True,
                                                dataset=raw_train_set)
    transformers = [transformer]
    train_dataset = transformer.transform(raw_train_set)
    valid_dataset = transformer.transform(raw_valid_set)
    test_dataset = transformer.transform(raw_test_set)
    # force using the GPU; run 'watch nvidia-smi -a --display=utilization'
    # in another terminal to check at run-time
    use_GPU = tf.ConfigProto(device_count = {'GPU': 1})
    model = GraphConvModel(1, batch_size=50, mode='classification',
                           config=use_GPU)
    max_epochs = 100
    # model = utils.train_early_stop(
    #     max_epochs, model, train_dataset, valid_dataset, transformers)
    # WARNING: nb_epoch is fixed to 8
    model.fit(train_dataset, nb_epoch=8, restore=False, checkpoint_interval=0)
    print("Evaluating model")
    train_scores = model.evaluate(train_dataset, [utils.metric], transformers)
    valid_scores = model.evaluate(valid_dataset, [utils.metric], transformers)
    test_scores = model.evaluate(test_dataset, [utils.metric], transformers)
    print("traAUC: %.3f" % train_scores["mean-roc_auc_score"])
    print("valAUC: %.3f" % valid_scores["mean-roc_auc_score"])
    print("tesAUC: %.3f" % test_scores["mean-roc_auc_score"])

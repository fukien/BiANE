#! ~/conda3/bin/python3
# -*- coding: utf-8 -*-

import argparse
import ipdb
import os
import pickle
import random
import sys

import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import tensorflow as tf

import jointmodel
import utils


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def link_prediction(sess, model, u_list, i_list, y_list):
    test_embs = []
    test_labels = []

    for user_node, item_node in zip(u_list, i_list):
        test_embs.append(np.multiply(model_user_embeddings[user_node], model_item_embeddings[item_node]))
    test_labels = y_list

    test_predict = classifier.predict(test_embs)
    test_prob = classifier.predict_proba(test_embs)[:,1]
    fpr, tpr, thresholds = metrics.roc_curve(test_labels, test_prob)
    auc_pr = metrics.average_precision_score(test_labels, test_prob)
    auc_roc = metrics.auc(fpr, tpr)

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for idx, label in enumerate(test_labels):
        if label == 1:
            if test_predict[idx] == 1:
                tp += 1
            else:
                fn += 1
        else:
            if test_predict[idx] == 1:
                fp += 1
            else:
                tn += 1
    if tp + fp:
        precision = tp/(tp+fp)
    else:
        precision = 0.0
    if tp + fn:
        recall = tp/(tp+fn)
    else:
        recall = 0.0
    if tp + tn:
        acc = (tp + tn)/(tp + fp + tn + fn)
    else:
        acc = 0.0
    if precision + recall > 1e-6:
        f1 = 2*precision*recall/(precision+recall)
    else:
        f1 = 0.0
    return (f1, acc, auc_roc, auc_pr, precision, recall, tp, fp, tn, fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Link Prediction', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='ami',
        # default='mvl'
        )
    parser.add_argument('--attr_dim_0_u', type=int, default=128,
        # default=23
        )
    parser.add_argument('--attr_dim_0_v', type=int, default=128,
        # default=18
        )
    parser.add_argument('--attr_dim_1', type=int, default=64,
        # default=32
        )
    parser.add_argument('--attr_dim_2', type=int, default=16,
        # default=64
        )
    parser.add_argument('--attr_dim_3', type=int, default=16)
    parser.add_argument('--struc_dim_0', type=int, default=128)
    parser.add_argument('--struc_dim_1', type=int, default=64,
        # default=96
        )
    parser.add_argument('--struc_dim_2', type=int, default=16,
        # default=64
        )
    parser.add_argument('--struc_dim_3', type=int, default=16)
    args = parser.parse_args()

    if args.dataset == 'ami':
        dim_list = [
            [args.attr_dim_0_u, args.attr_dim_1, args.attr_dim_2],
            [args.struc_dim_0, args.struc_dim_1, args.struc_dim_2],
            [args.attr_dim_0_v, args.attr_dim_1, args.attr_dim_2],
            [args.struc_dim_0, args.struc_dim_1, args.struc_dim_2]
        ]
        with open('{}_best_model.pkl'.format(args.dataset), 'rb') as f:
            params = pickle.load(f)
        model = jointmodel.JointModelAmi(dim_list, params)
    elif args.dataset == 'mvl':
        dim_list = [
            [args.attr_dim_0_u, args.attr_dim_1, args.attr_dim_2, args.attr_dim_3],
            [args.struc_dim_0, args.struc_dim_1, args.struc_dim_2, args.struc_dim_3],
            [args.attr_dim_0_v, args.attr_dim_1, args.attr_dim_2, args.attr_dim_3],
            [args.struc_dim_0, args.struc_dim_1, args.struc_dim_2, args.struc_dim_3]
        ]
        with open('{}_best_model.pkl'.format(args.dataset), 'rb') as f:
            params = pickle.load(f)
        model = jointmodel.JointModelMvl(dim_list, params)
    else:
        sys.exit(1)

    with open(os.path.join('../data', args.dataset, 'user_attr.pkl'), 'rb') as f:
        user_attr_raw = pickle.load(f)
    with open(os.path.join('../data', args.dataset, 'item_attr.pkl'), 'rb') as f:
        item_attr_raw = pickle.load(f)

    import ipdb
    user_list = [i for i in range(user_attr_raw.shape[0])]
    item_list = [i for i in range(item_attr_raw.shape[0])]

    struc_raw = {}
    with open(os.path.join('../data', args.dataset, 'emb_{}.txt'.format(args.dataset)), 'r', encoding='utf-8') as f:
        line = f.readline()
        line = f.readline()
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip().split()
            node_id = int(line[0])
            emb = np.array([float(v) for v in line[1:]])
            struc_raw[node_id] = emb
    user_struc_raw = []
    for user_id in range(len(user_list)):
        user_struc_raw.append(struc_raw[user_id])
    item_struc_raw = []
    for item_id in range(len(user_list), len(user_list)+len(item_list)):
        item_struc_raw.append(struc_raw[item_id])
    user_struc_raw = np.array(user_struc_raw)
    item_struc_raw = np.array(item_struc_raw)
    del struc_raw

    with open(os.path.join('../data', args.dataset, 'train.tsv'), 'r', encoding='utf-8') as f:
        u_list_train = []
        i_list_train = []
        y_list_train = []
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip().split('\t')
            u_list_train.append(int(line[0]))
            i_list_train.append(int(line[1]))
            y_list_train.append(int(line[2]))

    with open(os.path.join('../data', args.dataset, 'test.tsv'), 'r', encoding='utf-8') as f:
        u_list_test = []
        i_list_test = []
        y_list_test = []
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip().split('\t')
            u_list_test.append(int(line[0]))
            i_list_test.append(int(line[1]))
            y_list_test.append(int(line[2]))


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    feed_dict={
        model.user_attr.input: user_attr_raw[user_list,:],
        model.user_struc.input: user_struc_raw[user_list,:],
        model.item_attr.input: item_attr_raw[item_list,:],
        model.item_struc.input: item_struc_raw[item_list,:],
    }

    model_user_embeddings, model_item_embeddings = sess.run([model.user_emb, model.item_emb], feed_dict=feed_dict)

    train_embs = []
    train_labels = []
    for user_node, item_node in zip(u_list_train, i_list_train):
        train_embs.append(np.multiply(model_user_embeddings[user_node], model_item_embeddings[item_node]))
    train_labels = y_list_train
    classifier = LogisticRegression(
        solver='liblinear',
        n_jobs=-1,
        verbose=1,
        tol=1e-7
    )
    classifier.fit(train_embs, train_labels)

    f1, acc, auc_roc, auc_pr, precision, recall, tp, fp, tn, fn = link_prediction(sess, model, u_list_test, i_list_test, y_list_test)
    print('{}|{}|{}|{}|{}|{}|{}|{}|{}|{}'.format(str(utils.round_up_4(float(f1))), 
        str(utils.round_up_4(float(acc))), 
        str(utils.round_up_4(float(auc_roc))), 
        str(utils.round_up_4(float(auc_pr))), 
        str(utils.round_up_4(float(precision))), 
        str(utils.round_up_4(float(recall))), tp, fp, tn, fn))
    print('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(f1, acc, auc_roc, auc_pr, precision, recall, tp, fp, tn, fn), utils.date_time_format())
    sess.close()
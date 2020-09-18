#! ~/conda3/bin/python3
# -*- coding: utf-8 -*-

import argparse
import copy
import ipdb
import logging
import math
import os
import pickle
import random
import sys
import time

import nmslib
import numpy as np
import tensorflow as tf

import jointmodel
import utils

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def read_node_id(filepath):
    n2i = {}
    i2n = {}
    with open(filepath, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip().split('\t')
            node_name = line[0]
            node_id = int(line[1])
            n2i[node_name] = node_id
            i2n[node_id] = node_name
    return n2i, i2n


def train(args):
    def degrees_normalization(node_degree):
        norm = sum([math.pow(node_degree[i], args.power) for i in range(len(node_degree))])
        sampling_table = np.zeros(int(args.table_size), dtype=np.uint32)
        p = 0
        i = 0
        for j in range(len(node_degree)):
            p += float(math.pow(node_degree[j], args.power)) / norm
            while i < args.table_size and float(i) / args.table_size < p:
                sampling_table[i] = j
                i += 1
        return sampling_table

    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('{}.log'.format(args.dataset), mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    user_n2i, user_i2n = read_node_id(os.path.join('../data', args.dataset, 'user_id.tsv'))
    item_n2i, item_i2n = read_node_id(os.path.join('../data', args.dataset, 'item_id.tsv'))
    adj_user_n2i, adj_user_i2n = read_node_id(os.path.join('../data', args.dataset, 'adjlist_user_id.tsv'))
    adj_item_n2i, adj_item_i2n = read_node_id(os.path.join('../data', args.dataset, 'adjlist_item_id.tsv'))
    user_list = [i for i in range(len(user_n2i))]
    item_list = [i for i in range(len(item_n2i))]
    with open(os.path.join('../data', args.dataset, 'user_attr.pkl'), 'rb') as f:
        user_attr_raw = pickle.load(f)
    with open(os.path.join('../data', args.dataset, 'item_attr.pkl'), 'rb') as f:
        item_attr_raw = pickle.load(f)
    struc_raw = {}
    with open(os.path.join('../data', args.dataset, 'emb_{}.txt'.format(args.dataset)), 'r', encoding='utf-8') as f:
        line = f.readline() # header line 1 (skip this header line): node_num, emb_dim
        line = f.readline() # header line 2 (skip this header line): </s>(invalid token) embedding
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip().split()
            node_id = int(line[0]) # adjlist.txt node id
            emb = np.array([float(v) for v in line[1:]])
            struc_raw[node_id] = emb
    user_struc_raw = []
    for user_id in range(len(user_n2i)):
        user_struc_raw.append(struc_raw[user_id])
    item_struc_raw = []
    for item_id in range(len(user_n2i), len(user_n2i)+len(item_n2i)):
        item_struc_raw.append(struc_raw[item_id])
    user_struc_raw = np.array(user_struc_raw)
    item_struc_raw = np.array(item_struc_raw)
    del struc_raw
    with open(os.path.join('../data', args.dataset, 'adjlist.txt'), 'r', encoding='utf-8') as f:
        user_adjlist_dict = {}
        item_adjlist_dict = {}
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip().split()
            if int(line[0]) < len(user_n2i):
                user_id = int(line[0])
                user_adjlist_dict[user_id] = set([item_n2i[adj_item_i2n[int(i)]] for i in line[1:]])
            else:
                item_id = item_n2i[adj_item_i2n[int(line[0])]]
                item_adjlist_dict[item_id] = set([int(i) for i in line[1:]])
    with open(os.path.join('../data', args.dataset, 'train.csv'), 'r', encoding='utf-8') as f:
        user_pos_train = {}
        item_pos_train = {}
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip().split(',')
            user_pos_train.setdefault(int(line[0]), set()).add(int(line[1]))
            item_pos_train.setdefault(int(line[1]), set()).add(int(line[0]))
    with open(os.path.join('../data', args.dataset, 'valid.tsv'), 'r', encoding='utf-8') as f:
        u_list_valid = []
        i_list_valid = []
        y_list_valid = []
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip().split('\t')
            u_list_valid.append(int(line[0]))
            i_list_valid.append(int(line[1]))
            y_list_valid.append(float(line[2]))
    logger.info('DATA LOADING DONE')

    user_user_cache = []
    item_item_cache = []
    user_i_deg = []
    item_u_deg = []
    user_u_deg = []
    item_i_deg = []
    for user_id in user_list:
        user_i_deg.append(len(user_adjlist_dict[user_id]))
        tmp_user_cache = set()
        for item_id in user_adjlist_dict[user_id]:
            tmp_user_cache = tmp_user_cache | item_adjlist_dict[item_id]
        user_user_cache.append(tmp_user_cache)
        user_u_deg.append(len(tmp_user_cache))
    for item_id in item_list:
        item_u_deg.append(len(item_adjlist_dict[item_id]))
        tmp_item_cache = set()
        for user_id in item_adjlist_dict[item_id]:
            tmp_item_cache = tmp_item_cache | user_adjlist_dict[user_id]
        item_item_cache.append(tmp_item_cache)
        item_i_deg.append(len(item_item_cache[item_id]))
    user_i_deg = degrees_normalization(np.array(user_i_deg))
    item_u_deg = degrees_normalization(np.array(item_u_deg))
    user_u_deg = degrees_normalization(np.array(user_u_deg))
    item_i_deg = degrees_normalization(np.array(item_i_deg))
    del user_adjlist_dict
    del item_adjlist_dict
    logger.info('INTRA-PARTITION NETWORK SYNTHESIZED')

    if args.dataset == 'ami':
        dim_list = [
            [args.attr_dim_0_u, args.attr_dim_1, args.attr_dim_2],
            [args.struc_dim_0, args.struc_dim_1, args.struc_dim_2],
            [args.attr_dim_0_v, args.attr_dim_1, args.attr_dim_2],
            [args.struc_dim_0, args.struc_dim_1, args.struc_dim_2]
        ]
        model = jointmodel.JointModelAmi(dim_list, None)
    elif args.dataset == 'mvl':
        dim_list = [
            [args.attr_dim_0_u, args.attr_dim_1, args.attr_dim_2, args.attr_dim_3],
            [args.struc_dim_0, args.struc_dim_1, args.struc_dim_2, args.struc_dim_3],
            [args.attr_dim_0_v, args.attr_dim_1, args.attr_dim_2, args.attr_dim_3],
            [args.struc_dim_0, args.struc_dim_1, args.struc_dim_2, args.struc_dim_3]
        ]
        model = jointmodel.JointModelMvl(dim_list, None)
    else:
        sys.exit(1)

    ua_recon_loss = tf.reduce_mean([
        args.lambda_0*model.user_attr.recon_loss,
        args.lambda_11*model.user_attr.en_l2_loss,
        args.lambda_11*model.user_attr.de_l2_loss
    ])

    ua_recon_update = tf.train.AdamOptimizer(learning_rate=args.intra_lr).minimize(ua_recon_loss,
        var_list=model.user_attr.vars_en+model.user_attr.vars_de)

    us_recon_loss = tf.reduce_mean([
        args.lambda_1*model.user_struc.recon_loss,
        args.lambda_11*model.user_struc.en_l2_loss,
        args.lambda_11*model.user_struc.de_l2_loss,
    ])

    us_recon_update = tf.train.AdamOptimizer(learning_rate=args.intra_lr).minimize(us_recon_loss,
        var_list=model.user_struc.vars_en+model.user_struc.vars_de)

    ia_recon_loss = tf.reduce_mean([
        args.lambda_2*model.item_attr.recon_loss,
        args.lambda_11*model.item_attr.en_l2_loss,
        args.lambda_11*model.item_attr.de_l2_loss
    ])

    ia_recon_update = tf.train.AdamOptimizer(learning_rate=args.intra_lr).minimize(ia_recon_loss,
        var_list=model.item_attr.vars_en+model.item_attr.vars_de)

    is_recon_loss = tf.reduce_mean([
        args.lambda_3*model.item_struc.recon_loss,
        args.lambda_11*model.item_struc.en_l2_loss,
        args.lambda_11*model.item_struc.de_l2_loss
    ])

    is_recon_update = tf.train.AdamOptimizer(learning_rate=args.intra_lr).minimize(is_recon_loss,
        var_list=model.item_struc.vars_en+model.item_struc.vars_de)

    ua_local_loss = tf.reduce_mean([
        args.lambda_4*model.user_attr.local_loss,
        args.lambda_11*model.user_attr.en_l2_loss,
    ])

    ua_local_update = tf.train.AdamOptimizer(learning_rate=args.intra_lr).minimize(ua_local_loss,
        var_list=model.user_attr.vars_en)

    us_local_loss = tf.reduce_mean([
        args.lambda_5*model.user_struc.local_loss,
        args.lambda_11*model.user_struc.en_l2_loss
    ])

    us_local_update = tf.train.AdamOptimizer(learning_rate=args.intra_lr).minimize(us_local_loss,
        var_list=model.user_struc.vars_en)

    uas_loss = tf.reduce_mean([
        args.lambda_6*model.uas_loss,
        args.lambda_11*model.user_attr.en_l2_loss,
        args.lambda_11*model.user_attr.dr_l2_loss,
        args.lambda_11*model.user_struc.en_l2_loss,
        args.lambda_11*model.user_struc.dr_l2_loss
    ])

    uas_update = tf.train.AdamOptimizer(learning_rate=args.intra_lr).minimize(uas_loss,
        var_list=model.user_attr.vars_en+model.user_attr.vars_dr+\
        model.user_struc.vars_en+model.user_struc.vars_dr)

    ia_local_loss = tf.reduce_mean([
        args.lambda_7*model.item_attr.local_loss,
        args.lambda_11*model.item_attr.en_l2_loss,
    ])

    ia_local_update = tf.train.AdamOptimizer(learning_rate=args.intra_lr).minimize(ia_local_loss,
        var_list=model.item_attr.vars_en)

    is_local_loss = tf.reduce_mean([
        args.lambda_8*model.item_struc.local_loss,
        args.lambda_11*model.item_struc.en_l2_loss
    ])

    is_local_update = tf.train.AdamOptimizer(learning_rate=args.intra_lr).minimize(is_local_loss,
        var_list=model.item_struc.vars_en)

    ias_loss = tf.reduce_mean([
        args.lambda_9*model.ias_loss,
        args.lambda_11*model.item_attr.en_l2_loss,
        args.lambda_11*model.item_attr.dr_l2_loss,
        args.lambda_11*model.item_struc.en_l2_loss,
        args.lambda_11*model.item_struc.dr_l2_loss
    ])

    ias_update = tf.train.AdamOptimizer(learning_rate=args.intra_lr).minimize(ias_loss,
        var_list=model.item_attr.vars_en+model.item_attr.vars_dr+\
        model.item_struc.vars_en+model.item_struc.vars_dr)

    global_loss = tf.reduce_mean([
        args.lambda_10*model.pred_loss,
        args.lambda_11*model.user_attr.en_l2_loss,
        args.lambda_11*model.user_struc.en_l2_loss,
        args.lambda_11*model.item_attr.en_l2_loss,
        args.lambda_11*model.item_struc.en_l2_loss
    ])

    global_update = tf.train.AdamOptimizer(learning_rate=args.lr).minimize(global_loss,
        var_list=model.user_attr.vars_en+\
        model.user_struc.vars_en+\
        model.item_attr.vars_en+\
        model.item_struc.vars_en)

    # INTER-PARTITION PROXIMITY
    def generate_candidate_train():
        u_list = []
        i_list = []
        y_list = []
        for user in user_list:
            pos_set = user_pos_train[user]
            pos_num = len(pos_set)
            candidate_set = set(item_u_deg[np.random.randint(args.table_size, size=3*pos_num)])
            neg_list = list(candidate_set - pos_set)
            if len(neg_list) < pos_num:
                continue
            u_list.extend([user]*2*pos_num)
            i_list.extend(list(pos_set))
            i_list.extend(neg_list[:pos_num])
            y_list.extend([1]*pos_num)
            y_list.extend([0]*pos_num)
        for item in item_list:
            pos_set = item_pos_train[item]
            pos_num = len(pos_set)
            candidate_set = set(user_i_deg[np.random.randint(args.table_size, size=3*pos_num)])
            neg_list = list(candidate_set - pos_set)
            if len(neg_list) < pos_num:
                continue
            u_list.extend(list(pos_set))
            u_list.extend(neg_list[:pos_num])
            i_list.extend([item]*2*pos_num)
            y_list.extend([1]*pos_num)
            y_list.extend([0]*pos_num)
        rand_seed = random.randint(0,10000000)
        random.seed(rand_seed)
        random.shuffle(u_list)
        random.seed(rand_seed)
        random.shuffle(i_list)
        random.seed(rand_seed)
        random.shuffle(y_list)
        return u_list, i_list, y_list
    # FIRST-ORDER PROXIMITY
    def generate_candidate_local_train(cache, deg_table):
        c_list = []
        n_list = []
        y_list = []
        for idx, pos_set in enumerate(cache):
            if len(pos_set) == 1:
                continue
            pos_num = len(pos_set)
            candidate_set = set(deg_table[np.random.randint(args.table_size, size=3*pos_num)])
            neg_list = list(candidate_set - pos_set)
            pos_set.remove(idx)
            pos_num = pos_num - 1
            if len(neg_list) < pos_num:
                continue
            c_list.extend([idx]*2*pos_num)
            n_list.extend(list(pos_set))
            n_list.extend(neg_list[:pos_num])
            y_list.extend([1]*pos_num)
            y_list.extend([0]*pos_num)
        rand_seed = random.randint(0,10000000)
        random.seed(rand_seed)
        random.shuffle(c_list)
        random.seed(rand_seed)
        random.shuffle(n_list)
        random.seed(rand_seed)
        random.shuffle(y_list)
        return c_list, n_list, y_list
    # RECONSTRUCTION LOSS
    def train_recon(sess, update, loss, idx_list, raw, model):
        recon_loss = 0.0

        total_batch = len(idx_list)//args.batch_size
        for batch_idx in range(total_batch+1):
            batch_idx_list = utils.get_batch_data([idx_list], batch_idx, args.batch_size)[0]
            if len(batch_idx_list) == 0:
                continue
            _, tmp_loss = sess.run([update, loss], feed_dict={
                model.input: raw[batch_idx_list, :]
            })
            recon_loss += tmp_loss
        recon_loss = recon_loss/total_batch
        return recon_loss
    # FIRST-ORDER PROXIMITY
    def train_local(sess, update, loss, c_list, n_list, y_list, raw, model):
        local_loss = 0.0

        total_batch = len(c_list)//args.batch_size
        for batch_idx in range(total_batch+1):
            batch_c_list, batch_n_list, batch_y_list = utils.get_batch_data([c_list, n_list, y_list],
                batch_idx, args.batch_size)
            if len(batch_c_list) == 0:
                continue
            _, tmp_loss = sess.run([update, loss], feed_dict={
                    model.m_input: raw[batch_c_list, :],
                    model.n_input: raw[batch_n_list, :],
                    model.y_handle: batch_y_list
                })
            local_loss += tmp_loss
        local_loss = local_loss/total_batch
        return local_loss
    # ATTR-STRUC CORRELATION
    def train_attr_struc(sess, data_type):
        attr_struc_loss = 0.0
        a_list = []
        s_list = []
        y_list = []
        if data_type == 'user':
            update = uas_update
            loss = uas_loss
            cache = user_user_cache
            deg_table = user_u_deg
            attr_raw = user_attr_raw
            struc_raw = user_struc_raw
            attr_model = model.user_attr
            struc_model = model.user_struc
            adr = model.uadr
            sdr = model.usdr
            y_handle = model.uy_handle
        else: # item data type
            update = ias_update
            loss = ias_loss
            cache = item_item_cache
            deg_table = item_i_deg
            attr_raw = item_attr_raw
            struc_raw = item_struc_raw
            attr_model = model.item_attr
            struc_model = model.item_struc
            adr = model.iadr
            sdr = model.isdr
            y_handle = model.iy_handle
        attr_dr2, struc_dr2 = sess.run([adr, sdr],feed_dict={
            attr_model.input: attr_raw,
            struc_model.input: struc_raw
        })
        struc_index = nmslib.init(method='hnsw', space='negdotprod')
        struc_index.addDataPointBatch(struc_dr2)
        struc_index.createIndex({'post': 2}, print_progress=False)
        attr_struc_neighbours = struc_index.knnQueryBatch(attr_dr2, k=args.samp_num, num_threads=4)
        for node_id, struc_neigh in enumerate(attr_struc_neighbours):
            # pos_set = cache[node_id] | set(struc_neigh[0])
            pos_set = set(struc_neigh[0]) | set([node_id])
            pos_num = len(pos_set)
            candidate_set = set(deg_table[np.random.randint(args.table_size, size=3*pos_num)])
            neg_list = list(candidate_set - pos_set)
            if len(neg_list) < pos_num:
                continue
            a_list.extend([node_id]*2*pos_num)
            s_list.extend(list(pos_set))
            s_list.extend(neg_list[:pos_num])
            y_list.extend([1]*pos_num)
            y_list.extend([0]*pos_num)
        attr_index = nmslib.init(method='hnsw', space='negdotprod')
        attr_index.addDataPointBatch(attr_dr2)
        attr_index.createIndex({'post': 2}, print_progress=False)
        struc_attr_neighbours = attr_index.knnQueryBatch(struc_dr2, k=args.samp_num, num_threads=4)
        for node_id, attr_neigh in enumerate(struc_attr_neighbours):
            # pos_set = cache[node_id] | set(attr_neigh[0])
            pos_set = set(attr_neigh[0]) | set([node_id])
            pos_num = len(pos_set)
            candidate_set = set(deg_table[np.random.randint(args.table_size, size=3*pos_num)])
            neg_list = list(candidate_set - pos_set)
            if len(neg_list) < pos_num:
                continue
            s_list.extend([node_id]*2*pos_num)
            a_list.extend(list(pos_set))
            a_list.extend(neg_list[:pos_num])
            y_list.extend([1]*pos_num)
            y_list.extend([0]*pos_num)
        rand_seed = random.randint(0,10000000)
        random.seed(rand_seed)
        random.shuffle(a_list)
        random.seed(rand_seed)
        random.shuffle(s_list)
        random.seed(rand_seed)
        random.shuffle(y_list)
        total_batch = len(a_list) // args.batch_size
        for batch_idx in range(total_batch+1):
            batch_a_list, batch_s_list, batch_y_list = utils.get_batch_data([a_list, s_list, y_list],
                batch_idx, args.batch_size)
            if len(batch_a_list) == 0:
                continue
            _, tmp_loss = sess.run([update, loss], feed_dict={
                    attr_model.input: attr_raw[batch_a_list, :],
                    struc_model.input: struc_raw[batch_s_list, :],
                    y_handle: batch_y_list
                })
            attr_struc_loss += tmp_loss
        attr_struc_loss = attr_struc_loss/total_batch
        return attr_struc_loss

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    logger.info('INITIALIZATION DONE')

    best_valid_loss = np.finfo(float).max
    for epoch in range(1, args.epochs+1):
        # RECONSTRUCTION LOSS
        epoch_ua_recon_loss = train_recon(sess, ua_recon_update, ua_recon_loss, 
            user_list, user_attr_raw, model.user_attr)
        epoch_us_recon_loss = train_recon(sess, us_recon_update, us_recon_loss,
            user_list, user_struc_raw, model.user_struc)
        epoch_ia_recon_loss = train_recon(sess, ia_recon_update, ia_recon_loss,
            item_list, item_attr_raw, model.item_attr)
        epoch_is_recon_loss = train_recon(sess, is_recon_update, is_recon_loss,
            item_list, item_struc_raw, model.item_struc)
        #  FIRST-ORDER PROXIMITY
        user_c_list, user_n_list, user_y_list = generate_candidate_local_train(copy.deepcopy(user_user_cache),
            user_u_deg)
        epoch_ua_local_loss = train_local(sess, ua_local_update, ua_local_loss,
            user_c_list, user_n_list, user_y_list, user_attr_raw, model.user_attr)
        epoch_us_local_loss = train_local(sess, us_local_update, us_local_loss,
            user_c_list, user_n_list, user_y_list, user_struc_raw, model.user_struc)
        item_c_list, item_n_list, item_y_list = generate_candidate_local_train(copy.deepcopy(item_item_cache),
            item_i_deg)
        epoch_ia_local_loss = train_local(sess, ia_local_update, ia_local_loss,
            item_c_list, item_n_list, item_y_list, item_attr_raw, model.item_attr)
        epoch_is_local_loss = train_local(sess, is_local_update, is_local_loss,
            item_c_list, item_n_list, item_y_list, item_struc_raw, model.item_struc)
        del user_c_list, user_n_list, user_y_list, item_c_list, item_n_list, item_y_list
        # ATTR-STRUC CORRELATION
        epoch_uas_loss = train_attr_struc(sess, 'user')
        epoch_ias_loss = train_attr_struc(sess, 'item')
        # INTER-PARTITION PROXIMITY
        train_u_list, train_i_list, train_y_list = generate_candidate_train()
        total_batch = len(train_u_list)//args.batch_size
        train_epoch_loss = 0.0
        for batch_idx in range(total_batch + 1):
            batch_u_list, batch_i_list, batch_y_list = utils.get_batch_data([train_u_list, train_i_list, train_y_list],
                batch_idx, args.batch_size)
            if len(batch_u_list) == 0:
                continue
            _, tmp_loss = sess.run([global_update, global_loss], 
                feed_dict={
                    model.user_attr.input: user_attr_raw[batch_u_list,:],
                    model.user_struc.input: user_struc_raw[batch_u_list,:],
                    model.item_attr.input: item_attr_raw[batch_i_list,:],
                    model.item_struc.input: item_struc_raw[batch_i_list,:],
                    model.y_handle: batch_y_list
                })
            train_epoch_loss += tmp_loss
        train_epoch_loss = train_epoch_loss/total_batch

        valid_epoch_loss = 0.0
        total_batch = len(u_list_valid)//args.batch_size
        for batch_idx in range(total_batch + 1):
            batch_u_list, batch_i_list, batch_y_list = utils.get_batch_data([u_list_valid, i_list_valid, y_list_valid],
                batch_idx, args.batch_size)
            if len(batch_u_list) == 0:
                continue
            tmp_loss = sess.run(global_loss, 
                feed_dict={
                    model.user_attr.input: user_attr_raw[batch_u_list,:],
                    model.user_struc.input: user_struc_raw[batch_u_list,:],
                    model.item_attr.input: item_attr_raw[batch_i_list,:],
                    model.item_struc.input: item_struc_raw[batch_i_list,:],
                    model.y_handle: batch_y_list
                })
            valid_epoch_loss += tmp_loss
        if valid_epoch_loss < best_valid_loss:
            best_valid_loss = valid_epoch_loss
            model.save_model(sess, '{}_best_model.pkl'.format(args.dataset))
            logger.info('SAVED BEST MODEL AT EPOCH {}'.format(epoch))

    model.save_model(sess, '{}_final_model.pkl'.format(args.dataset))
    sess.close()
    logging.shutdown()


def main():
    parser = argparse.ArgumentParser('BiANE', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='ami',
        # default='mvl'
        )
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=7.5e-5)
    parser.add_argument('--intra_lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--table_size', type=float, default=1e8)
    parser.add_argument('--power', type=float, default=0.75)
    parser.add_argument('--samp_num', type=int, default=5)
    parser.add_argument('--lambda_0', type=float, default=1)
    parser.add_argument('--lambda_1', type=float, default=1)
    parser.add_argument('--lambda_2', type=float, default=1)
    parser.add_argument('--lambda_3', type=float, default=1)
    parser.add_argument('--lambda_4', type=float, default=1)
    parser.add_argument('--lambda_5', type=float, default=1)
    parser.add_argument('--lambda_6', type=float, default=1,
        # default=10
        )
    parser.add_argument('--lambda_7', type=float, default=1)
    parser.add_argument('--lambda_8', type=float, default=1)
    parser.add_argument('--lambda_9', type=float, default=1,
        # default=10
        )
    parser.add_argument('--lambda_10', type=float, default=1)
    parser.add_argument('--lambda_11', type=float, default=1e-4)
    # parser.add_argument('--layer_num', type=int, default=1,
    #     # default=2
    #     )
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
    train(args)

if __name__ == '__main__':
    sys.exit(main())
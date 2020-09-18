#! ~/conda3/bin/python3
# -*- coding: utf-8 -*-

import pickle
import random

# import numpy as np
import tensorflow as tf

import autoencoder

class JointModelAmi(object):
    """docstring for JointModel"""
    def __init__(self, dim_list, params):
        super(JointModelAmi, self).__init__()
        cur_seed = random.getrandbits(32)
        initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed)

        self.uy_handle = tf.placeholder(name='user_y_handle', dtype=tf.float32)
        self.iy_handle = tf.placeholder(name='item_y_handle', dtype=tf.float32)
        self.y_handle = tf.placeholder(name='y_handle', dtype=tf.float32)

        if params == None:
            with tf.variable_scope('user_attr'):
                self.user_attr = autoencoder.AutoEncoderAmi(dim_list[0], None)
            with tf.variable_scope('user_struc'):
                self.user_struc = autoencoder.AutoEncoderAmi(dim_list[1], None)
            with tf.variable_scope('item_attr'):
                self.item_attr = autoencoder.AutoEncoderAmi(dim_list[2], None)
            with tf.variable_scope('item_struc'):
                self.item_struc = autoencoder.AutoEncoderAmi(dim_list[3], None)

        else:
            with tf.variable_scope('user_attr'):
                self.user_attr = autoencoder.AutoEncoderAmi(dim_list[0], params[0])
            with tf.variable_scope('user_struc'):
                self.user_struc = autoencoder.AutoEncoderAmi(dim_list[1], params[1])
            with tf.variable_scope('item_attr'):
                self.item_attr = autoencoder.AutoEncoderAmi(dim_list[2], params[2])
            with tf.variable_scope('item_struc'):
                self.item_struc = autoencoder.AutoEncoderAmi(dim_list[3], params[3])

        self.params = []
        self.params.append(self.user_attr.params)
        self.params.append(self.user_struc.params)
        self.params.append(self.item_attr.params)
        self.params.append(self.item_struc.params)

        self.user_attr_emb = self.user_attr.en1
        self.user_struc_emb = self.user_struc.en1
        self.uadr = self.user_attr.dr
        self.usdr = self.user_struc.dr

        self.item_attr_emb = self.item_attr.en1
        self.item_struc_emb = self.item_struc.en1
        self.iadr = self.item_attr.dr
        self.isdr = self.item_struc.dr

        self.user_emb = tf.concat(values=[self.user_attr_emb, self.user_struc_emb], axis=1)
        self.item_emb = tf.concat(values=[self.item_attr_emb, self.item_struc_emb], axis=1)

        self.uas_sim = tf.reduce_sum(tf.multiply(self.uadr, self.usdr), axis=1)
        self.ias_sim = tf.reduce_sum(tf.multiply(self.iadr, self.isdr), axis=1)

        self.pred_score = tf.reduce_sum(tf.multiply(self.user_emb, self.item_emb), axis=1)
        self.pred_logits = tf.sigmoid(self.pred_score)

        self.uas_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.uy_handle, logits=self.uas_sim))
        self.ias_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.iy_handle, logits=self.ias_sim))

        self.pred_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_handle, logits=self.pred_score))

    def save_model(self, sess, filepath):
        params = sess.run(self.params)
        with open(filepath, 'wb') as f:
            pickle.dump(params, f, protocol=4)


class JointModelMvl(object):
    """docstring for JointModel"""
    def __init__(self, dim_list, params):
        super(JointModelMvl, self).__init__()
        cur_seed = random.getrandbits(32)
        initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed)

        self.uy_handle = tf.placeholder(name='user_y_handle', dtype=tf.float32)
        self.iy_handle = tf.placeholder(name='item_y_handle', dtype=tf.float32)
        self.y_handle = tf.placeholder(name='y_handle', dtype=tf.float32)

        if params == None:
            with tf.variable_scope('user_attr'):
                self.user_attr = autoencoder.AutoEncoderMvl(dim_list[0], None)
            with tf.variable_scope('user_struc'):
                self.user_struc = autoencoder.AutoEncoderMvl(dim_list[1], None)
            with tf.variable_scope('item_attr'):
                self.item_attr = autoencoder.AutoEncoderMvl(dim_list[2], None)
            with tf.variable_scope('item_struc'):
                self.item_struc = autoencoder.AutoEncoderMvl(dim_list[3], None)

        else:
            with tf.variable_scope('user_attr'):
                self.user_attr = autoencoder.AutoEncoderMvl(dim_list[0], params[0])
            with tf.variable_scope('user_struc'):
                self.user_struc = autoencoder.AutoEncoderMvl(dim_list[1], params[1])
            with tf.variable_scope('item_attr'):
                self.item_attr = autoencoder.AutoEncoderMvl(dim_list[2], params[2])
            with tf.variable_scope('item_struc'):
                self.item_struc = autoencoder.AutoEncoderMvl(dim_list[3], params[3])

        self.params = []
        self.params.append(self.user_attr.params)
        self.params.append(self.user_struc.params)
        self.params.append(self.item_attr.params)
        self.params.append(self.item_struc.params)

        self.user_attr_emb = self.user_attr.en2
        self.user_struc_emb = self.user_struc.en2
        self.uadr = self.user_attr.dr
        self.usdr = self.user_struc.dr

        self.item_attr_emb = self.item_attr.en2
        self.item_struc_emb = self.item_struc.en2
        self.iadr = self.item_attr.dr
        self.isdr = self.item_struc.dr

        self.user_emb = tf.concat(values=[self.user_attr_emb, self.user_struc_emb], axis=1)
        self.item_emb = tf.concat(values=[self.item_attr_emb, self.item_struc_emb], axis=1)

        self.uas_sim = tf.reduce_sum(tf.multiply(self.uadr, self.usdr), axis=1)
        self.ias_sim = tf.reduce_sum(tf.multiply(self.iadr, self.isdr), axis=1)

        self.pred_score = tf.reduce_sum(tf.multiply(self.user_emb, self.item_emb), axis=1)
        self.pred_logits = tf.sigmoid(self.pred_score)

        self.uas_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.uy_handle, logits=self.uas_sim))
        self.ias_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.iy_handle, logits=self.ias_sim))

        self.pred_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_handle, logits=self.pred_score))

    def save_model(self, sess, filepath):
        params = sess.run(self.params)
        with open(filepath, 'wb') as f:
            pickle.dump(params, f, protocol=4)
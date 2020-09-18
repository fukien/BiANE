#! ~/conda3/bin/python3
# -*- coding: utf-8 -*-

import random

import numpy as np
import tensorflow as tf

class AutoEncoderAmi(object):
    """docstring for AutoEncoder"""
    def __init__(self, dim, params):
        super(AutoEncoderAmi, self).__init__()
        cur_seed = random.getrandbits(32)
        initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed)

        with tf.variable_scope('auto_encoder'):
            with tf.name_scope('input'):
                self.input = tf.placeholder(name='input', shape=[None, dim[0]], dtype=tf.float32)
                self.m_input = tf.placeholder(name='m_input', shape=[None, dim[0]], dtype=tf.float32)
                self.n_input = tf.placeholder(name='m_input', shape=[None, dim[0]], dtype=tf.float32)
                self.y_handle = tf.placeholder(name='label', dtype=tf.float32)

            if params == None:
                with tf.name_scope('encoder'):
                    self.w1 = tf.get_variable('w1', shape=[dim[0], dim[1]], initializer=initializer)
                    self.b1 = tf.get_variable('b1', shape=[dim[1]], initializer=tf.constant_initializer(0.0))
                    self.en1 = tf.nn.softsign(tf.nn.xw_plus_b(x=self.input, weights=self.w1, biases=self.b1), name='en1')
                    self.m_en1 = tf.nn.softsign(tf.nn.xw_plus_b(x=self.m_input, weights=self.w1, biases=self.b1), name='m_en1')
                    self.n_en1 = tf.nn.softsign(tf.nn.xw_plus_b(x=self.n_input, weights=self.w1, biases=self.b1), name='n_en1')
                with tf.name_scope('decoder'):
                    self.w2 = tf.get_variable('w2', shape=[dim[1], dim[0]], initializer=initializer)
                    self.b2 = tf.get_variable('b2', shape=[dim[0]], initializer=tf.constant_initializer(0.0))
                    self.de1 = tf.nn.softsign(tf.nn.xw_plus_b(x=self.en1, weights=self.w2, biases=self.b2), name='de1')
                with tf.name_scope('transforming_kernel'):
                    self.dr_w = tf.get_variable('rd_w', shape=[dim[1], dim[2]], initializer=initializer)
                    self.dr_b = tf.get_variable('rd_b', shape=[dim[2]], initializer=tf.constant_initializer(0.0))
                    self.dr = tf.nn.softsign(tf.nn.xw_plus_b(x=self.en1, weights=self.dr_w, biases=self.dr_b), name='dr')

            else:
                with tf.name_scope('encoder'):
                    self.w1 = tf.Variable(initial_value=params[0], name='w1')
                    self.b1 = tf.Variable(initial_value=params[1], name='b1')
                    self.en1 = tf.nn.softsign(tf.nn.xw_plus_b(x=self.input, weights=self.w1, biases=self.b1), name='en1')
                    self.m_en1 = tf.nn.softsign(tf.nn.xw_plus_b(x=self.m_input, weights=self.w1, biases=self.b1), name='m_en1')
                    self.n_en1 = tf.nn.softsign(tf.nn.xw_plus_b(x=self.n_input, weights=self.w1, biases=self.b1), name='n_en1')
                with tf.name_scope('decoder'):
                    self.w2 = tf.Variable(initial_value=params[2], name='w3')
                    self.b2 = tf.Variable(initial_value=params[3], name='b3')
                    self.de1 = tf.nn.softsign(tf.nn.xw_plus_b(x=self.en1, weights=self.w2, biases=self.b2), name='de1')

                with tf.name_scope('transforming_kernel'):
                    self.dr_w = tf.Variable(initial_value=params[4], name='rd_w')
                    self.dr_b = tf.Variable(initial_value=params[5], name='rd_b')
                    self.dr = tf.nn.softsign(tf.nn.xw_plus_b(x=self.en1, weights=self.dr_w, biases=self.dr_b), name='dr')

            self.vars_en = [self.w1, self.b1]
            self.vars_de = [self.w2, self.b2]
            self.vars_dr = [self.dr_w, self.dr_b]
            self.params = [self.w1, self.b1, self.w2, self.b2, self.dr_w, self.dr_b]

            with tf.name_scope('reconstruction'):
                self.recon_loss = tf.reduce_mean(tf.square(self.de1 - self.input))
            with tf.name_scope('local_loss'):
                self.local_sim = tf.reduce_sum(tf.multiply(self.m_en1, self.n_en1), axis=1)
                self.local_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_handle, logits=self.local_sim))
            with tf.name_scope('reguralization'):
                self.en_l2_loss = sum([tf.nn.l2_loss(p) for p in self.vars_en])
                self.de_l2_loss = sum([tf.nn.l2_loss(p) for p in self.vars_de])
                self.dr_l2_loss = sum([tf.nn.l2_loss(p) for p in self.vars_dr])


class AutoEncoderMvl(object):
    """docstring for AutoEncoder"""
    def __init__(self, dim, params):
        super(AutoEncoderMvl, self).__init__()
        cur_seed = random.getrandbits(32)
        initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed)

        with tf.variable_scope('auto_encoder'):
            with tf.name_scope('input'):
                self.input = tf.placeholder(name='input', shape=[None, dim[0]], dtype=tf.float32)
                self.m_input = tf.placeholder(name='m_input', shape=[None, dim[0]], dtype=tf.float32)
                self.n_input = tf.placeholder(name='m_input', shape=[None, dim[0]], dtype=tf.float32)
                self.y_handle = tf.placeholder(name='label', dtype=tf.float32)

            if params == None:
                with tf.name_scope('encoder'):
                    self.w1 = tf.get_variable('w1', shape=[dim[0], dim[1]], initializer=initializer)
                    self.b1 = tf.get_variable('b1', shape=[dim[1]], initializer=tf.constant_initializer(0.0))
                    self.en1 = tf.nn.softsign(tf.nn.xw_plus_b(x=self.input, weights=self.w1, biases=self.b1), name='en1')
                    self.m_en1 = tf.nn.softsign(tf.nn.xw_plus_b(x=self.m_input, weights=self.w1, biases=self.b1), name='m_en1')
                    self.n_en1 = tf.nn.softsign(tf.nn.xw_plus_b(x=self.n_input, weights=self.w1, biases=self.b1), name='n_en1')
                    self.w2 = tf.get_variable('w2', shape=[dim[1], dim[2]], initializer=initializer)
                    self.b2 = tf.get_variable('b2', shape=[dim[2]], initializer=tf.constant_initializer(0.0))
                    self.en2 = tf.nn.xw_plus_b(x=self.en1, weights=self.w2, biases=self.b2, name='en2')
                    self.m_en2 = tf.nn.softsign(tf.nn.xw_plus_b(x=self.m_en1, weights=self.w2, biases=self.b2), name='m_en2')
                    self.n_en2 = tf.nn.softsign(tf.nn.xw_plus_b(x=self.n_en1, weights=self.w2, biases=self.b2), name='n_en2')

                with tf.name_scope('decoder'):
                    self.w3 = tf.get_variable('w3', shape=[dim[2], dim[1]], initializer=initializer)
                    self.b3 = tf.get_variable('b3', shape=[dim[1]], initializer=tf.constant_initializer(0.0))
                    self.de1 = tf.nn.softsign(tf.nn.xw_plus_b(x=self.en2, weights=self.w3, biases=self.b3), name='de1')
                    self.w4 = tf.get_variable('w4', shape=[dim[1], dim[0]], initializer=initializer)
                    self.b4 = tf.get_variable('b4', shape=[dim[0]], initializer=tf.constant_initializer(0.0))
                    self.de2 = tf.nn.xw_plus_b(x=self.de1, weights=self.w4, biases=self.b4, name='de2')

                with tf.name_scope('transforming_kernel'):
                    self.dr_w = tf.get_variable('rd_w', shape=[dim[2], dim[3]], initializer=initializer)
                    self.dr_b = tf.get_variable('rd_b', shape=[dim[3]], initializer=tf.constant_initializer(0.0))
                    self.dr = tf.nn.softsign(tf.nn.xw_plus_b(x=self.en2, weights=self.dr_w, biases=self.dr_b), name='dr')

            else:
                with tf.name_scope('encoder'):
                    self.w1 = tf.Variable(initial_value=params[0], name='w1')
                    self.b1 = tf.Variable(initial_value=params[1], name='b1')
                    self.en1 = tf.nn.softsign(tf.nn.xw_plus_b(x=self.input, weights=self.w1, biases=self.b1), name='en1')
                    self.m_en1 = tf.nn.softsign(tf.nn.xw_plus_b(x=self.m_input, weights=self.w1, biases=self.b1), name='m_en1')
                    self.n_en1 = tf.nn.softsign(tf.nn.xw_plus_b(x=self.n_input, weights=self.w1, biases=self.b1), name='n_en1')
                    self.w2 = tf.Variable(initial_value=params[2], name='w2')
                    self.b2 = tf.Variable(initial_value=params[3], name='b2')
                    self.en2 = tf.nn.xw_plus_b(x=self.en1, weights=self.w2, biases=self.b2, name='en2')
                    self.m_en2 = tf.nn.softsign(tf.nn.xw_plus_b(x=self.m_en1, weights=self.w2, biases=self.b2), name='m_en2')
                    self.n_en2 = tf.nn.softsign(tf.nn.xw_plus_b(x=self.n_en1, weights=self.w2, biases=self.b2), name='n_en2')

                with tf.name_scope('decoder'):
                    self.w3 = tf.Variable(initial_value=params[4], name='w3')
                    self.b3 = tf.Variable(initial_value=params[5], name='b3')
                    self.de1 = tf.nn.softsign(tf.nn.xw_plus_b(x=self.en2, weights=self.w3, biases=self.b3), name='de1')
                    self.w4 = tf.Variable(initial_value=params[6], name='w4')
                    self.b4 = tf.Variable(initial_value=params[7], name='b4')
                    self.de2 = tf.nn.xw_plus_b(x=self.de1, weights=self.w4, biases=self.b4, name='de2')

                with tf.name_scope('transforming_kernel'):
                    self.dr_w = tf.Variable(initial_value=params[8], name='rd_w')
                    self.dr_b = tf.Variable(initial_value=params[9], name='rd_b')
                    self.dr = tf.nn.softsign(tf.nn.xw_plus_b(x=self.en2, weights=self.dr_w, biases=self.dr_b), name='dr')

            self.vars_en = [self.w1, self.b1, self.w2, self.b2]
            self.vars_de = [self.w3, self.b3, self.w4, self.b4]
            self.vars_dr = [self.dr_w, self.dr_b]
            self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4, self.dr_w, self.dr_b]

            with tf.name_scope('reconstruction'):
                self.recon_loss = tf.reduce_mean(tf.square(self.de2 - self.input))
            with tf.name_scope('local_loss'):
                self.local_sim = tf.reduce_sum(tf.multiply(self.m_en2, self.n_en2), axis=1)
                self.local_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_handle, logits=self.local_sim))
            with tf.name_scope('reguralization'):
                self.en_l2_loss = sum([tf.nn.l2_loss(p) for p in self.vars_en])
                self.de_l2_loss = sum([tf.nn.l2_loss(p) for p in self.vars_de])
                self.dr_l2_loss = sum([tf.nn.l2_loss(p) for p in self.vars_dr])
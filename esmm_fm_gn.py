#coding:utf-8
"""train the model"""

import argparse
import configparser
import logging
import reprlib
import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.feature_column.feature_column import _LazyBuilder
from tensorflow.python.ops import embedding_ops, control_flow_ops, state_ops
from tensorflow.python.framework import ops
from tensorflow.python.estimator.canned import dnn, linear, optimizers
from tensorflow.python.estimator.canned import head as head_v1
from tensorflow_estimator.python.estimator.head import binary_class_head, multi_head
from tensorflow_estimator.python.estimator.model_fn import _TPUEstimatorSpec
import six
import math
from collections import OrderedDict
sys.path.append("../esmm/tftools/")

from FCGen import FCGen

L0REG = 0.1
BETA = 0.5
GAMMA = -0.1
ETA = 1.1

class EmbeddingTable:
  """修改自: https://github.com/stasi009/Recommend-Estimators/blob/master/deepfm.py"""

  def __init__(self):
    self._linear_weights = {}
    self._embed_weights = {}

  def __contains__(self, item):
    return item in self._embed_weights

  def add_linear_weights(self, vocab_name, vocab_size):
    """
    :param vocab_name: 一个field拥有两个权重矩阵，一个用于线性连接，另一个用于非线性（二阶或更高阶交叉）连接
    :param vocab_size: 字典总长度
    :param embed_dim: 二阶权重矩阵shape=[vocab_size, order2dim]，映射成的embedding
                      既用于接入DNN的第一屋，也是用于FM二阶交互的隐向量
    :return: None
    """
    linear_weight = tf.get_variable(
      name='{}_linear_weight'.format(vocab_name),
      shape=[vocab_size, 1],
      initializer=tf.glorot_normal_initializer(),
      dtype=tf.float32)

    self._linear_weights[vocab_name] = linear_weight
  
  def add_embed_weights(self, vocab_name, vocab_size, embed_dim, reg):
    """
    :param vocab_name: 一个field拥有两个权重矩阵，一个用于线性连接，另一个用于非线性（二阶或更高阶交叉）连接
    :param vocab_size: 字典总长度
    :param embed_dims: 二阶权重矩阵shape=[vocab_size, embed_dim]，映射成的embedding
                      既用于接入DNN的第一屋，也是用于FM二阶交互的隐向量
    :return: None
    """
    if vocab_name not in self._embed_weights:
      # 二阶（FM）特征的embedding，可共享embedding矩阵
      embed_weight = tf.get_variable(
        name='{}_embed_weight'.format(vocab_name),
        shape=[vocab_size, embed_dim],
        initializer=tf.glorot_normal_initializer(),
        regularizer=tf.contrib.layers.l2_regularizer(reg),
        dtype=tf.float32)

      self._embed_weights[vocab_name] = embed_weight

  def get_linear_weights(self, vocab_name=None):
    """get linear weights"""
    if vocab_name is not None:
      return self._linear_weights[vocab_name]
    else:
      return self._linear_weights

  def get_embed_weights(self, vocab_name=None):
    """get poly weights"""
    if vocab_name is not None:
      return self._embed_weights[vocab_name]
    else:
      return self._embed_weights


def build_deep_layers(hidden, hidden_units, mode, task, reg):
  hidden = build_dnn_layers(hidden, hidden_units, mode, task, reg)
  logits = tf.keras.layers.Dense(units=1,
                              activation=None, use_bias=False,
                              kernel_initializer=tf.glorot_uniform_initializer(),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(reg),
                              name='{}PredictionLayer_{}'.format(task, len(hidden_units)+1))(hidden)
  return logits

def build_dnn_layers(hidden, hidden_units, mode, task, reg):
  for l in range(len(hidden_units)):
    hidden = tf.keras.layers.Dense(units=hidden_units[l],
                             activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=tf.glorot_uniform_initializer(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(reg),
                             name='{}EmbeddingLayer_{}'.format(task, l+1))(hidden)
    hidden = tf.keras.layers.BatchNormalization(epsilon=1e-7, momentum=0.999, trainable=mode == tf.estimator.ModeKeys.TRAIN)(hidden)
  return hidden  

def build_input(features, params):
  cat_columns = params['cat_columns']
  val_columns = params['val_columns']
  column_to_field = params['column_to_field']
  #dnn_columns = params['dnn_columns']
  dimension_config = params['dimension_config']
  reg = params['reg']
  embed_dim = params['embed_dim']
  embedding_table = EmbeddingTable()
  embedding_dict = OrderedDict()
  with tf.variable_scope("fm", reuse=tf.AUTO_REUSE, values=[features]) as scope:
    with tf.device('/cpu:0'):
      for name, col in cat_columns.items():
        field = column_to_field.get(name, name)
        cur_dimension = dimension_config[field] if field in dimension_config else embed_dim
        embedding_table.add_linear_weights(vocab_name=name,
                                           vocab_size=col._num_buckets)
        embedding_table.add_embed_weights(vocab_name=field,
                                          vocab_size=col._num_buckets,
                                          embed_dim=cur_dimension,
                                          reg=reg)
      for name, col in val_columns.items():
        field = column_to_field.get(name, name)
        cur_dimension = dimension_config[field] if field in dimension_config else embed_dim
        embedding_table.add_linear_weights(vocab_name=name,
                                           vocab_size=1)
        embedding_table.add_embed_weights(vocab_name=field,
                                          vocab_size=1,
                                          embed_dim=cur_dimension,
                                          reg=reg)

      builder = _LazyBuilder(features)
      # linear part
      linear_outputs = []
      for name, col in cat_columns.items():
        # get sparse tensor of input feature from feature column
        sp_tensor = col._get_sparse_tensors(builder)
        sp_ids = sp_tensor.id_tensor
        linear_weights = embedding_table.get_linear_weights(name)

        # linear_weights: (vocab_size, 1)
        # sp_ids: (batch_size, max_tokens_per_example)
        # sp_values: (batch_size, max_tokens_per_example)
        linear_output = embedding_ops.safe_embedding_lookup_sparse(
          linear_weights,
          sp_ids,
          None,
          combiner='sum',
          name='{}_linear_output'.format(name))

        linear_outputs.append(linear_output)
      for name, col in val_columns.items():
        dense_tensor = col._get_dense_tensor(builder)
        linear_weights = embedding_table.get_linear_weights(name)
        linear_output = tf.multiply(dense_tensor, linear_weights)
        linear_outputs.append(linear_output)
      # linear_outputs: (batch_szie, nonzero_feature_num)
      linear_outputs = tf.concat(linear_outputs, axis=1)
      # poly part

      for name, col, in cat_columns.items():
        # get sparse tensor of input feature from feature column
        field = column_to_field.get(name, name)
        sp_tensor = col._get_sparse_tensors(builder)
        sp_ids = sp_tensor.id_tensor
        embed_weights = embedding_table.get_embed_weights(field)

        # embeddings: (batch_size, embed_dim)
        # x_i * v_i
        embeddings = embedding_ops.safe_embedding_lookup_sparse(
          embed_weights,
          sp_ids,
          None,
          combiner='sum',
          name='{}_{}_embedding'.format(field, name))
        embedding_dict[field] = embeddings
      for name, col in val_columns.items():
        field = column_to_field.get(name, name)
        dense_tensor = col._get_dense_tensor(builder)
        embed_weights = embedding_table.get_embed_weights(field)
        embeddings = tf.multiply(dense_tensor, embed_weights)
        embedding_dict[field] = embeddings
  with tf.variable_scope("dnn_embed"):
    x = tf.concat(list(embedding_dict.values()), axis=1)
    N = len(embedding_dict)
    T = sum([embedding.get_shape().as_list()[1] for embedding in embedding_dict.values()])
    print("wkfm N:", N, " T:", T)
    indices = []
    for i, embeddings in enumerate(embedding_dict.values()):
      dim = embeddings.get_shape().as_list()[1]
      indices.extend([i] * dim)
    outputs = []
    for field, embeddings in embedding_dict.items():
      di = dimension_config[field] if field in dimension_config else embed_dim
      U = tf.get_variable('{}_wkfm'.format(field), [T, di], initializer=tf.glorot_normal_initializer(), trainable=True)
      wkfm_weights = tf.get_variable('{}_wkfm_weights'.format(field), [N], initializer=tf.ones_initializer, trainable=True)
      weights = tf.gather(wkfm_weights, indices)
      y = tf.matmul(weights * x, U)
      outputs.append(y)
    y = tf.concat(outputs, axis=1)
    y = x * y
    new_inputs = tf.concat([linear_outputs, y], 1)
    shared_weights = tf.get_variable(name="fm_share", dtype=tf.float32,
                                     shape=[new_inputs.get_shape().as_list()[1], 256],
                                     initializer=tf.glorot_normal_initializer(),
                                     regularizer=tf.contrib.layers.l2_regularizer(reg),
                                     trainable=True)
    new_inputs = tf.matmul(new_inputs, shared_weights)
    return new_inputs, shared_weights

total = 34848
def lbtw(loss_list):
  # Loss-Balanced Task Weighting to Reduce Negative Transfer in Multi-Task Learning
  task_num = len(loss_list)
  init_list = [tf.Variable(-1.0, trainable=False) for i in range(task_num)]
  total_tensor = tf.get_variable("total",initializer=tf.constant(total))
  step = tf.mod(tf.train.get_global_step(), total_tensor)
  def assign_init(init, loss):
    with tf.control_dependencies([tf.assign(init, loss)]):
      return tf.identity(init)
  alpha = 0.5
  orig_list = [tf.cond(tf.equal(step, 0), 
                        lambda: assign_init(init_list[i], loss_list[i]),
                        lambda: tf.identity(init_list[i]))
                for i in range(task_num)]
  l_hat_list = [tf.div(loss_list[i], orig_list[i]) for i in range(task_num)]
  l_hat_avg = tf.div(tf.add_n(l_hat_list), task_num)
  inv_rate_list = [tf.div(l_hat_list[i], l_hat_avg) for i in range(task_num)]
  a = tf.constant(alpha)
  w_list = [tf.pow(inv_rate_list[i], a) for i in range(task_num)]
  weight_loss = [tf.multiply(loss_list[i], w_list[i]) for i in range(task_num)]
  return weight_loss
  
def get_weight_loss(loss_list, dynamic, weights_shared):
  if dynamic:
    return grad_norm(loss_list, weights_shared)
  return loss_list, None, None, None

def grad_norm_new(loss_list, weights_shared, loss_pow=2):
  n = len(loss_list)
  def get_norm(grad):
    return tf.reduce_mean(tf.norm(grad, axis=1))
  with tf.variable_scope("grad_weight", reuse=tf.AUTO_REUSE):
    weights = tf.get_variable('grad_norm_weights', shape=[n], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=True)
  weights = tf.nn.softmax(weights)
  grads  = [tf.gradients(loss, weights_shared) for loss in loss_list]
  grads  = [tf.concat(gs, axis = 1) for gs in grads]
  gnorms = [get_norm(g) for g in grads]
  gnorms = tf.stop_gradient(tf.stack(gnorms, axis = 0))
  avgnorm = tf.reduce_sum(gnorms * weights) / n
  wgnorms = gnorms * weights
  gnorm_loss = tf.reduce_sum((wgnorms - avgnorm) ** loss_pow)
  weighted_loss = tf.reduce_sum(tf.stack(loss_list, axis = 0) * tf.stop_gradient(weights))
  for i in range(n):
    tf.summary.scalar('gradnorm/weight_{}'.format(i), weights[i])
  return weighted_loss, None, weights, gnorm_loss


def grad_norm(loss_list, weights_shared):
  alpha = 0.12
  task_num = len(loss_list)
  w_list = [tf.Variable(1.0, name="w_".format(i)) for i in range(task_num)]
  weight_loss = [tf.multiply(loss_list[i], w_list[i]) for i in range(task_num)]
  init_list = [tf.Variable(-1.0, trainable=False) for i in range(task_num)]
  def assign_init(init, loss):
    with tf.control_dependencies([tf.assign(init, loss)]):
      return tf.identity(init)
  orig_list = [tf.cond(
                    tf.equal(init_list[i], -1.0),
                    lambda: assign_init(init_list[i], loss_list[i]),
                    lambda: tf.identity(init_list[i]))
                for i in range(task_num)]
  G_norm_list = []
  for i in range(task_num):
    grads = tf.gradients(weight_loss[i], weights_shared)
    g_norm = tf.norm(tf.concat([tf.add(grad,1e-8) for grad in grads], axis=1), axis=1)
      #g_norm.append(tf.stack([tf.norm(tf.add(grad,1e-8), ord=2) for grad in grads]))
      #g_norm.append(tf.concat([tf.norm(tf.gradients(weight_loss[i], weight)[0],ord=2) for weight in weights], axis=0))
    G_norm_list.append(g_norm)
  G_avg = tf.reduce_mean(G_norm_list)
  l_hat_list = [tf.div(loss_list[i], orig_list[i]) for i in range(task_num)]
  l_hat_avg = tf.div(tf.add_n(l_hat_list), task_num)
  inv_rate_list = [tf.div(l_hat_list[i], l_hat_avg) for i in range(task_num)]
  a = tf.constant(alpha)
  C_list = [tf.multiply(G_avg, tf.pow(inv_rate_list[i], a)) for i in range(task_num)]
  C_list = [tf.stop_gradient(C_list[i]) for i in range(task_num)]
  loss_gradnorm = tf.add_n([tf.reduce_sum(tf.abs(tf.subtract(G_norm_list[i], C_list[i])))
                            for i in range(task_num)])
  with tf.control_dependencies([loss_gradnorm]):
    coef = tf.div(float(task_num), tf.add_n(w_list))
    update_list = [w_list[i].assign(tf.multiply(w_list[i], coef))
                    for i in range(task_num)]
  for i in range(task_num):
    tf.summary.scalar('l_hat/l{}'.format(i), tf.squeeze(tf.reduce_mean(inv_rate_list[i])))
    tf.summary.scalar('lgrad/l{}'.format(i), tf.squeeze(tf.reduce_sum(tf.abs(tf.subtract(G_norm_list[i], C_list[i])))))
    tf.summary.scalar("gradnorm/w_{}".format(i), tf.squeeze(w_list[i]))
  return weight_loss, update_list, w_list, loss_gradnorm

def snr_gate_net(name, input_list, out_dim, out_num, eta, gamma, beta):
  input_dim, in_num = input_list.get_shape().as_list()[1:]
  w_out = tf.get_variable(name='snr_weight_{}'.format(name), 
                          shape=[input_dim, in_num, out_dim, out_num],
                          initializer=tf.glorot_uniform_initializer())
  u = tf.random_uniform([1, in_num, 1, out_num])
  log_alpha = tf.get_variable(name='snr_log_alpha_{}'.format(name),
                              shape=[1, in_num, 1,out_num],
                              initializer=tf.keras.initializers.Zeros())
  s = tf.sigmoid((tf.log(u) - tf.log(1.0 - u) + log_alpha) / beta)
  s = s * (eta - gamma) + gamma
  z = tf.minimum(1.0, tf.maximum(s, 0.0))
  gate = tf.tile(z, [input_dim, 1, out_dim, 1])
  out = tf.tensordot(input_list, tf.multiply(w_out, gate), axes=2)
  l0_reg = tf.reduce_sum(tf.sigmoid(log_alpha - beta * math.log(- gamma / eta)))
  return out, l0_reg

def mixture(exp_stack, gate, agg_type='sum'):
  if agg_type == 'sum':
    if gate is not None:
      exp_stack = tf.matmul(exp_stack, tf.expand_dims(gate, -1))
    exp_out = tf.squeeze(exp_stack, axis=2)
  elif agg_type == 'concat':
    if gate is not None:
      exp_stack = tf.matmul(exp_stack, tf.matrix_diag(gate))
    exp_out = tf.reshape(exp_stack, shape=[-1, exp_stack.shape[1] * exp_stack.shape[2]])
  else:
    raise ValueError('Unsupported agg_type: {}'.format(agg_type))
  return exp_out


def build_mmoe(features, params, scope, task_num=2): 
  inputs = build_input(features, params)
  with tf.variable_scope(scope):
    expert_num = params['expert_num']
    expert_unit = params['expert_unit']
    reg = params['reg']
    experts_weight = tf.get_variable(name="experts_weight", dtype=tf.float32,
                                     shape=(inputs.get_shape()[1], expert_unit, expert_num),
                                     initializer=tf.glorot_normal_initializer(),
                                     regularizer=tf.contrib.layers.l2_regularizer(reg))
    experts_bias = tf.get_variable(name="experts_bias", dtype=tf.float32,
                                   shape=(expert_unit, expert_num),
                                   initializer=tf.glorot_normal_initializer(),
                                   regularizer=tf.contrib.layers.l2_regularizer(reg))
    experts_output = tf.nn.relu(tf.add(tf.tensordot(inputs, experts_weight, axes=1), experts_bias))
                    
    gates_weight = [tf.get_variable(name="gate_weight_task_{}".format(i), dtype=tf.float32,
                                    shape=(inputs.get_shape()[1], expert_num),
                                    initializer=tf.glorot_normal_initializer())
                    for i in range(task_num)]
    gates_bias = [tf.get_variable(name="gate_bias_task_{}".format(i), dtype=tf.float32,
                                  shape=(expert_num,), initializer=tf.glorot_normal_initializer())
                  for i in range(task_num)]
    gates_output = [tf.nn.softmax(tf.add(tf.matmul(inputs, gates_weight[i]), gates_bias[i]))
                    for i in range(task_num)]
    multi_inputs = [tf.reshape(
      tf.reduce_sum(
        tf.multiply(experts_output, tf.expand_dims(gates_output[i], axis=1)),
        axis=2),
      [-1, experts_output.get_shape()[1]])
      for i in range(task_num)]
    return multi_inputs, experts_weight

def esmm_model_fn(features, labels, mode, params):
  wt = tf.feature_column.input_layer(features, params['weight_columns'])
  inputs, weights_shared = build_input(features, params)
  hidden_units = params['hidden_units']
  linear_parent_scope = 'linear'
  dnn_parent_scope = 'dnn'
  is_dynamic = params['dynamic']
  print("is_dynamic:", is_dynamic)
  reg = 1e-4
  dnn_scope = 'dnn' 
  with tf.variable_scope(dnn_scope):
    ctr_logits = build_deep_layers(inputs, hidden_units, mode, 'CTR', reg)
    cvr_logits = build_deep_layers(inputs, hidden_units, mode, 'CVR', reg)
  ctr_preds = tf.nn.sigmoid(ctr_logits)
  cvr_preds = tf.nn.sigmoid(cvr_logits)
  ctcvr_preds = ctr_preds * cvr_preds
  tf.summary.histogram("esmm/ctr_preds", ctr_preds) 
  tf.summary.histogram("esmm/cvr_preds", cvr_preds) 
  tf.summary.histogram("esmm/ctcvr_preds", ctcvr_preds)
  if mode == tf.estimator.ModeKeys.PREDICT:
    #redundant_items = ctr_preds
    predictions = {
      'prob': tf.concat([cvr_preds, ctr_preds], 1)
    }
    export_outputs = {
      tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)  #线上预测需要的
    }
    return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

  else:
    ctr_labels = labels['ctr']
    ctcvr_labels = labels['ctcvr']
    linear_optimizer = tf.train.FtrlOptimizer(0.01, l1_regularization_strength=0.001, l2_regularization_strength=0.001)
    dnn_optimizer = optimizers.get_optimizer_instance('Adam', params['learning_rate'])
    loss_optimizer = optimizers.get_optimizer_instance('Adam', 0.001)
    ctr_loss = tf.losses.log_loss(ctr_labels, ctr_preds, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE, weights=wt)         
    ctcvr_loss = tf.losses.log_loss(ctcvr_labels, ctcvr_preds, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
    #reg_loss = tf.reduce_sum(ops.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    ctr_auc = tf.metrics.auc(labels=ctr_labels, predictions=ctr_preds, weights=wt)
    ctcvr_auc = tf.metrics.auc(labels=ctcvr_labels, predictions=ctcvr_preds)
    mask = tf.map_fn(lambda x:tf.cond(tf.equal(x, 1), lambda: True, lambda: False), tf.squeeze(labels['ctr']), dtype=tf.bool)
    cvr_preds = tf.boolean_mask(cvr_preds, mask)
    cvr_labels = tf.boolean_mask(labels['ctcvr'], mask)
    cvr_auc = tf.metrics.auc(labels=cvr_labels, predictions=cvr_preds)
    cvr_loss = tf.losses.log_loss(cvr_labels, cvr_preds, reduction=tf.losses.Reduction.SUM_OVER_NONZERO_WEIGHTS)
    tf.summary.scalar("cvr_auc", cvr_auc[1])
    tf.summary.scalar("cvr_loss", cvr_loss)
    tf.summary.scalar('ctr_loss', ctr_loss)
    tf.summary.scalar('ctcvr_loss', ctcvr_loss)
    tf.summary.scalar('ctr_auc', ctr_auc[1])
    tf.summary.scalar('ctcvr_auc', ctcvr_auc[1])
    weight_loss, update_list, w_list, loss_gradnorm = get_weight_loss([ctr_loss, ctcvr_loss], is_dynamic, weights_shared)
    #loss = tf.add_n(weight_loss + [reg_loss])
    loss = tf.add_n(weight_loss)
    #loss = weight_loss
    #w_list = ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES, scope='grad_weight')
    def _train_op_fn(loss):
      train_ops = []
      global_step = tf.train.get_global_step()
      if params['model'] in ('dnn'):
        fm_var_list = ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES, scope='fm')
        dnn_var_list = ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES, scope=dnn_scope) + ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES, scope='dnn_embed')
        train_ops.append(
          dnn_optimizer.minimize(
            loss,
            var_list=dnn_var_list))
        train_ops.append(
          linear_optimizer.minimize(
            loss,
            var_list=fm_var_list))
      if w_list is not None and loss_gradnorm is not None:
        train_ops.append(
          loss_optimizer.minimize(
              loss_gradnorm,
              var_list=w_list))
      if update_list is not None:
        train_ops.append(update_list)
      train_op = control_flow_ops.group(*train_ops)
      with ops.control_dependencies([train_op]):
        return state_ops.assign_add(global_step, 1).op
    hooks = tf.train.LoggingTensorHook({'ctr_loss':ctr_loss, 'ctcvr_loss':ctcvr_loss, 'cvr_loss':cvr_loss}, every_n_iter=10000)
    train_op = _train_op_fn(loss)
    train_op = head_v1._append_update_ops(train_op)
    metrics = {'ctr_auc': ctr_auc, 'ctcvr_auc': ctcvr_auc, 'cvr_auc': cvr_auc}
    #return _TPUEstimatorSpec(mode, loss=loss, train_op=train_op).as_estimator_spec()
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=metrics)

"""train the model"""

import argparse
import configparser
import logging
import reprlib
import sys
import os

import tensorflow as tf
from tensorflow.python.feature_column.feature_column import _LazyBuilder
sys.path.append("../..")

from FCGen import FCGen

def input_fn(record_files, spec, shuffle=True, batch_size=64, epochs=1, columns=None, keys=None, mt=False):
  """General input functions

  Args:
    record_files: (list) A list of tfrecord files
    spec: (dict) feature column parsing specification
    shuffle: (bool) whether to shuffle
    batch_size: (int) batch size

  Returns:
    dataset batch iterator and init op
  """
  files = tf.data.Dataset.from_tensor_slices(record_files)
  #dataset = tf.data.TFRecordDataset(record_files)
  dataset = files.interleave(tf.data.TFRecordDataset, 6)

  if epochs > 1:
    dataset = dataset.repeat(epochs)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=batch_size * 1000)

  def _map(example_proto):
    features = tf.parse_example(example_proto, spec)
    if not mt:
      labels = tf.cast(features.pop('label'), tf.int64)
    else:
      label = tf.cast(features.pop('label'), tf.int64)
      ctcvr_label = tf.cast(features.pop('watch_label'), tf.int64)
      labels = {'ctr': label, 'ctcvr': ctcvr_label}
    if keys and columns:
      builder = _LazyBuilder(features)
      for key in keys:
        tensor = features[key]
        features["{}_str".format(key)] = tf.as_string(tensor)
        features["{}_norm".format(key)] = columns[key]._transform_feature(builder)
    return features, labels

  #dataset = dataset.apply(tf.contrib.data.map_and_batch(
  #    map_func=_map, batch_size=batch_size, num_parallel_batches=56))
  dataset = dataset.batch(batch_size)
  dataset = dataset.map(_map, num_parallel_calls=8)

  #dataset = dataset.prefetch(1)

  #iterator = dataset.make_one_shot_iterator()
  #next_batch = iterator.get_next()

  #features, labels = next_batch[0], next_batch[1]

  return dataset

def input_fn_pattern(file_pattern, spec, shuffle=True, batch_size=64, epochs=1, mt=False):
  """General input functions

  Args:
    record_files: (list) A list of tfrecord files
    spec: (dict) feature column parsing specification
    shuffle: (bool) whether to shuffle
    batch_size: (int) batch size

  Returns:
    dataset batch iterator and init op
  """
  print("input_fn_pattern++++")
  print(file_pattern)
  files = tf.data.Dataset.from_tensor_slices(file_pattern).flat_map(
        lambda p: tf.data.Dataset.list_files(p, shuffle=True))
  #files = tf.data.Dataset.list_files(file_pattern)
  print(files)

  #dataset = tf.data.TFRecordDataset(record_files)
  dataset = files.apply(
        tf.data.experimental.parallel_interleave(
            tf.data.TFRecordDataset,
            cycle_length=15,  # �~P~L�~W�读�~O~V15个�~V~G件
            sloppy=True,  # �~E~A许以�~M确�~Z顺�~O�~N读�~O~V
            buffer_output_elements=2,
            prefetch_input_elements=2))
  '''
  dataset = files.interleave(
        tf.data.TFRecordDataset,
        cycle_length=6, # �~P~L�~W�读�~O~V10个�~V~G件
        num_parallel_calls=tf.data.experimental.AUTOTUNE
  )
  '''
  #dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=10,
  #  block_length=100, num_parallel_calls=10)

  if epochs > 1:
    dataset = dataset.repeat(epochs)
  #if shuffle:
  #  dataset = dataset.shuffle(buffer_size=batch_size * 100)

  def _map(example_proto):
    features = tf.parse_example(example_proto, spec)
    if not mt:
      labels = tf.cast(features.pop('label'), tf.int64)
    else:
      label = tf.cast(features.pop('label'), tf.int64)
      ctcvr_label = tf.cast(features.pop('watch_label'), tf.int64)
      labels = {'ctr': label, 'ctcvr': ctcvr_label}
    return features, labels

  dataset = dataset.batch(batch_size)
  dataset = dataset.map(_map, num_parallel_calls=8)

  #dataset = dataset.prefetch(1)

  #iterator = dataset.make_one_shot_iterator()
  #next_batch = iterator.get_next()

  #features = tf.parse_example(next_batch, features=spec)
  #labels = tf.cast(features.pop('label'), tf.int64)
  #features.pop('wt')

  #features, labels = next_batch[0], next_batch[1]

  return dataset

def input_fn_ugly(record_files, spec, nepoch, shuffle=True, batch_size=64, epochs=1, mt=False):
  """General input functions

  Args:
    record_files: (list) A list of tfrecord files
    spec: (dict) feature column parsing specification
    shuffle: (bool) whether to shuffle
    batch_size: (int) batch size

  Returns:
    dataset batch iterator and init op
  """
  files = tf.data.Dataset.from_tensor_slices(record_files)
  #dataset = tf.data.TFRecordDataset(record_files)
  dataset = files.interleave(tf.data.TFRecordDataset, 6)

  if epochs > 1:
    dataset = dataset.repeat(epochs)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=batch_size * 1000)

  def _map(example_proto):
    features = tf.parse_example(example_proto, spec)
    if not mt:
      labels = tf.cast(features.pop('label'), tf.int64)
    else:
      label = tf.cast(features.pop('label'), tf.int64)
      follow_label = tf.cast(features.pop('watch_label'), tf.int64)
      labels = {'ctr': label, 'ctcvr': follow_label}
    return features, labels

  #dataset = dataset.apply(tf.contrib.data.map_and_batch(
  #    map_func=_map, batch_size=batch_size, num_parallel_batches=56))
  dataset = dataset.batch(batch_size)
  dataset = dataset.map(_map, num_parallel_calls=8)

  dataset = dataset.prefetch(1)

  iterator = dataset.make_one_shot_iterator()
  next_batch = iterator.get_next()

  features, labels = next_batch[0], next_batch[1]

  return features, labels


def model_fn(features, labels, mode, params):
  units = params.get('units', 1)
  columns = params['columns']

  cols_to_vars = {}
  print(features)
  logits = tf.feature_column.linear_model(
                  features=features,
                  feature_columns=columns,
                  units=units,
                  cols_to_vars=cols_to_vars)

  prediction = tf.nn.sigmoid(logits)

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
      'logits': logits
    }

    return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  loss = tf.losses.sigmoid_cross_entropy(labels, logits=logits, reduction=tf.losses.Reduction.SUM)
  print(loss)

  loss = tf.losses.compute_weighted_loss(loss, reduction=tf.losses.Reduction.SUM)
  
  auc = tf.metrics.auc(labels=labels, predictions=prediction)

  metrics = {'auc': auc}
  tf.summary.scalar('auc', auc[0])
  
  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics)

  assert mode == tf.estimator.ModeKeys.TRAIN

  optimizer = tf.train.FtrlOptimizer(
                    learning_rate=params['learning_rate'],
                    l1_regularization_strength=params['l1_reg'],
                    l2_regularization_strength=params['l2_reg'])
  train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

"""Functions for evaluation"""

import logging
import os
import re

import tensorflow as tf

def evaluate_sess(sess, model_spec, config):
  """evaluation
  
  Args:
    sess: (tf.Session) current session
    model_spec: (dict) graph ops returned by model_fn
    config: (configparser) contains hyperparameters
  """
  update_metrics = model_spec['update_metrics']
  eval_metrics = model_spec['metrics']
  global_step = tf.train.get_global_step()

  sess.run(model_spec['iterator_init_op'])  # init datset iterator
  sess.run(model_spec['metrics_init_op'])

  exp = r"""^.*_1"""
  init_ops = [op for op in tf.get_default_graph().get_operations() 
                 if op.name.endswith("table_init") 
                   and re.search(exp, op.name) != None]
  
  sess.run(init_ops)

  # go through all the dataset to accumulate metrics
  while True:
    try:
      sess.run(update_metrics)
    except tf.errors.OutOfRangeError:
      break

  # get values of metrics
  metrics_values = {k: v[0] for k, v in eval_metrics.items()}
  metrics_val = sess.run(metrics_values)
  metrics_string = " ; ".join("{}: {:05.5f}".format(k, v) for k, v in metrics_val.items())
  logging.info("- Eval metrics: " + metrics_string)

  return metrics_val
"""Functions for training"""

import logging
import os

import tensorflow as tf
from Trainer.Evaluation import evaluate_sess


def train_sess(sess, model_spec, config):
  """Train model according to `model_spec` and configurations

  Args:
    sess: (tf.Session) current session
    model_spec: (dict) graph ops returned by model_fn
    config: (configparser) contains hyperparameters
  """

  loss = model_spec['loss']
  train_op = model_spec['train_op']
  update_metrics = model_spec['update_metrics']
  metrics = model_spec['metrics']
  global_step = tf.train.get_global_step()

  sess.run(model_spec['iterator_init_op'])  # init datset iterator
  sess.run(model_spec['metrics_init_op'])

  saver = tf.train.Saver(max_to_keep=2)

  num_steps = 0
  max_step = int(config['train'].get('max_step', -1))
  steps_per_save = int(config['train'].get('steps_per_save', 10000))
  save_dir = config['train'].get('checkpoint', './checkpoint/ckpt')
  # max_step = -1 表示不限制最大迭代次数
  while max_step < 0 or num_steps < max_step:
    try:
      _, _, loss_val, global_step_val = sess.run([train_op, update_metrics, loss, global_step])
      metrics_values = {k: v[0] for k, v in metrics.items()}
      metrics_val = sess.run(metrics_values)
      metrics_string = " ; ".join("{}: {:05.5f}".format(k, v) for k, v in metrics_val.items())
      if (num_steps % 100 == 0):
        logging.info("- step {}, Train metrics: ".format(global_step_val) + metrics_string)

      num_steps += 1
      # save checkpoint
      if num_steps % steps_per_save == 0:
        logging.info("start to saving checkpoint at step: {}".format(num_steps))
        saver.save(sess, save_dir, global_step=global_step_val)
      
      
    except tf.errors.OutOfRangeError:
      break



def train_and_evaluate(train_model_spec, eval_model_spec, config):
  """Train the model and evaluate every epoch

  Args:
    train_model_spec: (dict) graph ops for training
    eval_model_spec: (dict) graph ops for evaluation
    config: (configparser) contains hyperparameters
  """
  restore_from = config['train'].get('restore_from', '')
  last_saver = tf.train.Saver()
  last_saver_path = config['train'].get('checkpoint', './checkpoint/ckpt')

  best_saver = tf.train.Saver(max_to_keep=1)
  best_saver_path = config['train'].get('best_checkpoint', './best_checkpoint/ckpt')

  begin_at_epoch = 0
  num_epochs = int(config['train'].get('epochs', 1))

  with tf.Session() as sess:
    sess.run(train_model_spec['variable_init_op'])

    # Retore from checkpoint if specified
    if restore_from != '':
      logging.info("Restoring from directory: {}".format(restore_from))
      if os.path.isdir(restore_from):
        restore_from = tf.train.latest_checkpoint(restore_from)
        begin_at_epoch = int(restore_from.split('-')[-1])
      last_saver.restore(sess, restore_from)

    best_auc = 0.0
    for epoch in range(begin_at_epoch, begin_at_epoch + num_epochs):
      # one epoch
      logging.info("Epoch {}/{}".format(epoch + 1, begin_at_epoch + num_epochs))
      train_sess(sess, train_model_spec, config)

      # save for every epoch
      last_saver.save(sess, last_saver_path, global_step=epoch+1)

      # Evaluation for every epoch on validation set
      logging.info("Start evaluation after epoch {}".format(epoch))
      metrics = evaluate_sess(sess, eval_model_spec, config)

      eval_auc = metrics['auc']
      if eval_auc >= best_auc:
        best_auc = eval_auc
        logging.info("- Found new best auc: {}, saving in {}".format(best_auc, best_saver_path))
        best_saver.save(sess, best_saver_path, global_step=epoch+1)
        


    

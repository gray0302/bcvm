"""Construct graph for wide and deep model."""

import tensorflow as tf

from tensorflow.python.estimator.canned import dnn


def build_linear_model(inputs, columns, config):
  """Compute logits of the linear part (output distribution)
  Args:
      inputs: (dict) contains the inputs of the graph (features, labels...)
              this can be `tf.placeholder` or outputs of `tf.data`
      columns: (list) contains linear feature columns
      config: (configparser) contains hyperparameters for model building
  Returns:
      logits: (tf.Tensor) output of the model
  """
  features = inputs['features']

  cols_to_vars = {}
  units = int(config['linear_model'].get('units', 1))
  combiner = config['linear_model'].get('combiner', 'sum')
  linear_logits = tf.feature_column.linear_model(
                  features=features,
                  feature_columns=columns,
                  units=units,
                  sparse_combiner=combiner,
                  cols_to_vars=cols_to_vars)

  return linear_logits


def build_dnn_model(mode, inputs, columns, config):
  """Compute logits of the dnn part (output distribution)
  Args:
      inputs: (dict) contains the inputs of the graph (features, labels...)
              this can be `tf.placeholder` or outputs of `tf.data`
      columns: (list) contains dnn feature columns
      config: (configparser) contains hyperparameters for model building
  Returns:
      logits: (tf.Tensor) output of the model
  """
  features = inputs['features']

  # parse configurations
  units = int(config['dnn_model'].get('units', 1))
  dnn_hidden_units = [int(n) for n in config['dnn_model'].get('hiden_units', '512,128,64').split(',')]
  dnn_activation_fn = tf.nn.relu
  if config['dnn_model'].get('activation_fn', None) is not None:
    dnn_activation_fn = eval(config['dnn_model']['activation_fn'])
  dnn_dropout = None
  if config['dnn_model'].get('dropout', None) is not None:
    dnn_dropout = float(config['dnn_model']['dropout'])
  batch_norm = False
  if config['dnn_model'].get('batch_norm', '').lower() == 'true':
    batch_norm = True
  
  # build dnn part
  dnn_logit_fn = dnn._dnn_logit_fn_builder(
      units=units,
      hidden_units=dnn_hidden_units,
      feature_columns=columns,
      activation_fn=dnn_activation_fn,
      dropout=dnn_dropout,
      batch_norm=batch_norm)

  dnn_logits = dnn_logit_fn(features=features, mode=mode) 

  return dnn_logits


def model_fn(mode, inputs, columns, config, reuse=False):
  """Model function defining the graph operations.
  Args:
    mode: (string) can be 'train' or 'eval'
    inputs: (dict) contains the inputs of the graph (features, labels...)
            this can be `tf.placeholder` or outputs of `tf.data`
    columns: (list) feature columns        
    config: (configparser) contains hyperparameters for model building
    reuse: (bool) whether to reuse the weights
  Returns:
    model_spec: (dict) contains the graph operations or nodes needed for training/evaluation
  """

  if mode == 'train':
    mode = tf.estimator.ModeKeys.TRAIN
  
  is_training = (mode == tf.estimator.ModeKeys.TRAIN)
  
  labels = inputs['labels']
  labels = tf.cast(labels, tf.int64)

  # -----------------------------------------------------------
  # MODEL: define the forward ops
  with tf.variable_scope('linear_part', reuse=reuse):
    linear_logits = build_linear_model(inputs, columns['linear'], config)

  with tf.variable_scope('dnn_part', reuse=reuse):
    dnn_logits = build_dnn_model(mode, inputs, columns['dnn'], config)
       
  logits = linear_logits + dnn_logits
  predictions = tf.nn.sigmoid(logits)

  # loss
  loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

  linear_model_config = config['linear_model']
  dnn_model_config = config['dnn_model']
  # train_ops
  if is_training:
    global_step = tf.train.get_or_create_global_step()
    linear_optimizer = tf.train.FtrlOptimizer(
                    learning_rate=float(linear_model_config['learning_rate']),
                    l1_regularization_strength=float(linear_model_config['l1_reg']),
                    l2_regularization_strength=float(linearmodel_config['l2_reg']))

    dnn_optimizer = tf.train.AdamOptimizer(
                    learning_rate=float(dnn_model_config['learning_rate']),
                    beta1=float(dnn_model_config.get('beta1', 0.9)),
                    beta2=float(dnn_model_config.get('beta2', 0.999)),
                    epsilon=float(dnn_model_config.get('epsilon', 1e-8)))

    train_ops = []

    train_ops.append(
      linear_optimizer.minimize(
        loss,
        global_step=global_step,
        var_list=tf.get_collection(
          tf.GraphKeys.TRAINABLE_VARIABLES,
          scope='linear_part')))

    train_ops.append(
      dnn_optimizer.minimize(
        loss,
        global_step=global_step,
        var_list=tf.get_collection(
          tf.GraphKeys.TRAINABLE_VARIABLES,
          scope='dnn_part')))

    train_op = tf.group(*train_ops)

  # -----------------------------------------------------------
  # METRICS AND SUMMARIES
  # Metrics for evaluation using tf.metrics (average over whole dataset)
  with tf.variable_scope("metrics"):
    metrics = {
      'loss': tf.metrics.mean(loss),
      'auc': tf.metrics.auc(labels=labels, predictions=predictions, 
                            num_thresholds=200, summation_method='careful_interpolation')
    }

  # Group the update ops for the tf.metrics
  update_metrics_op = tf.group(*[op for _, op in metrics.values()])

  # Get the op to reset the local variables used in tf.metrics
  metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
  metrics_init_op = tf.variables_initializer(metric_variables)

  # -----------------------------------------------------------
  # MODEL SPECIFICATION
  # Create the model specification and return it
  # It contains nodes or operations in the graph that will be used for training and evaluation
  model_spec = inputs
  model_spec['variable_init_op'] = [tf.global_variables_initializer(),
                                    tf.tables_initializer()]
  model_spec["predictions"] = predictions
  model_spec['loss'] = loss
  model_spec['metrics_init_op'] = metrics_init_op
  model_spec['metrics'] = metrics
  model_spec['update_metrics'] = update_metrics_op

  if is_training:
    model_spec['train_op'] = train_op

  return model_spec
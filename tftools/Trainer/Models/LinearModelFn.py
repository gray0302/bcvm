"""Construct graph for linear model."""

import tensorflow as tf

def build_model(inputs, columns, config):
  """Compute logits of the model (output distribution)
  Args:
      inputs: (dict) contains the inputs of the graph (features, labels...)
              this can be `tf.placeholder` or outputs of `tf.data`
      columns: (list) feature columns
      config: (configparser) contains hyperparameters for model building
  Returns:
      logits: (tf.Tensor) output of the model
  """
  features = inputs['features']

  cols_to_vars = {}
  units = int(config['model'].get('units', 1))
  combiner = config['model'].get('combiner', 'sum')
  logits = tf.feature_column.linear_model(
                  features=features,
                  feature_columns=columns,
                  units=units,
                  sparse_combiner=combiner,
                  cols_to_vars=cols_to_vars)

  return logits


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

  is_training = (mode == 'train')
  labels = inputs['labels']
  #labels = tf.cast(labels, tf.int64)

  # -----------------------------------------------------------
  # MODEL: define the forward ops
  with tf.variable_scope('model', reuse=reuse):
    # Compute the output distribution of the model and the predictions
    logits = build_model(inputs, columns['default'], config)
    predictions = tf.nn.sigmoid(logits)
       
  # loss
  loss = tf.losses.sigmoid_cross_entropy(labels, logits=logits)

  model_config = config['model']
  # train_ops
  if is_training:
    optimizer = tf.train.FtrlOptimizer(
                    learning_rate=float(model_config['learning_rate']),
                    l1_regularization_strength=float(model_config['l1_reg']),
                    l2_regularization_strength=float(model_config['l2_reg']))

    global_step = tf.train.get_or_create_global_step()
    train_op = optimizer.minimize(loss, global_step=global_step)

  # -----------------------------------------------------------
  # METRICS AND SUMMARIES
  # Metrics for evaluation using tf.metrics (average over whole dataset)
  with tf.variable_scope("metrics"):
    metrics = {
      'loss': tf.metrics.mean(loss),
      'auc': tf.metrics.auc(labels=labels, predictions=predictions, 
                            num_thresholds=200)
    }

  # Group the update ops for the tf.metrics
  update_metrics_op = tf.group(*[op for _, op in metrics.values()])

  # Get the op to reset the local variables used in tf.metrics
  metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
  print(metric_variables)
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

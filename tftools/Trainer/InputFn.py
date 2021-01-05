"""Create general input functions for training and evaluation using tf.data apis"""

import tensorflow as tf

def input_fn(record_files, spec, shuffle=True, batch_size=64, epochs=1, mt=False):
  """General input functions

  Args:
    record_files: (list) A list of tfrecord files
    spec: (dict) feature column parsing specification
    shuffle: (bool) whether to shuffle
    batch_size: (int) batch size

  Returns:
    dataset batch iterator and init op
  """
  dataset = tf.data.TFRecordDataset(record_files)
  if shuffle:
    dataset = dataset.shuffle(buffer_size=batch_size)
  dataset = dataset.batch(batch_size)
  dataset = dataset.repeat(epochs)

  dataset = dataset.prefetch(1)

  iterator = dataset.make_initializable_iterator()
  next_batch = iterator.get_next()
  
  features = tf.parse_example(next_batch, features=spec)
  labels = features.pop('label')
  if not mt:
    inputs = {
      'features': features,
      'labels': labels,
      'iterator_init_op': iterator.initializer
    }
  else:
    follow_labels = features.pop('follow_label')
    inputs = {
      'features': features,
      'labels': {'ctr': labels, 'ctcvr': follow_labels},
      'iterator_init_op': iterator.initializer
    }
  inputs['example'] = next_batch
  return inputs

import tensorflow as tf
from collections import defaultdict
import numpy as np
# current column list
cur_columns = {}
dimension_config = {}
# final columns for output
out_columns = {}

# {group1: {k1: col1, k2: col2, ...}, group2: {...}, ...}
group_columns = defaultdict(dict)

def FCGenerator(generator):
  def fn(*args, **kwargs):
    key, col, is_out, group = generator(*args, **kwargs)
    cur_columns[key] = col
    if is_out:
      out_columns[key] = col
      for g in group.split(','):
        group_columns[g][key] = col
    return key, col

  return fn

def SharedFCGenerator(generator):
  def fn(*args, **kwargs):
    key, col, is_out, group = generator(*args, **kwargs)
    cur_columns[key] = col
    if is_out:
      out_columns[key] = col
      if len(group.split(',')) < 2:
        for g in group.split(','):
          group_columns[g][key] = col
      else:
        for i, g in enumerate(group.split(',')):
          group_columns[g][key] = [col[i]]
    return key, col

  return fn


def _GetNumericColumn(key, info_dict):
  length = int(info_dict.get('shape', 1))
  default = float(info_dict.get('default', 0.0))
  return key, [tf.feature_column.numeric_column(
                key, shape=(length,), default_value=default)]


@FCGenerator
def GetNumericColumn(key, info_dict):
  length = int(info_dict.get('shape', 1))
  default = float(info_dict.get('default', 0.0))
  is_out = info_dict.get('is_out', True) not in ("False", "false", "0")
  group = info_dict.get('group', 'default')
  fname = info_dict.get('fname', key)
  dimension = int(info_dict.get('dimension', '0'))
  if dimension > 0:
    dimension_config[key] = dimension
  if 'norm' in group.split(','):
    fmean = float (info_dict.get('mean', '0.0'))
    fstd = float (info_dict.get('std', '1.0'))
    def zscore(col):
      if 'log' in group.split(','):
        col = tf.math.log(1.0 + col)
        #print(key, col)
      return (col - fmean) / fstd
    return key, [tf.feature_column.numeric_column(
                fname, shape=(length,), default_value=default, normalizer_fn=zscore)], is_out, group
  else:
    return key, [tf.feature_column.numeric_column(
                fname, shape=(length,), default_value=default)], is_out, group
                               

@FCGenerator
def GetBucketizedColumn(key, info_dict):
  fname = info_dict.get('fname', key)
  _, num_col = _GetNumericColumn(fname, info_dict)
  print(fname)
  length = int(info_dict.get('shape', 1))
  boundaries = [float(b) for b in info_dict['boundaries'].split(',')]
  is_out = info_dict.get('is_out', True) not in ("False", "false", "0")
  group = info_dict.get('group', 'default')
  dimension = int(info_dict.get('dimension', '0'))
  if dimension > 0:
    dimension_config[key] = dimension

  return key, \
         [tf.feature_column.bucketized_column(num_col[0], boundaries)], is_out, group


@FCGenerator
def GetCatIdentityColumn(key, info_dict):
  def_value = int(info_dict.get("default", 0))
  num_buckets = int(info_dict["num_buckets"])
  is_out = info_dict.get('is_out', True) not in ("False", "false", "0")
  group = info_dict.get('group', 'default')
  fname = info_dict.get('fname', key)
  dimension = int(info_dict.get('dimension', '0'))
  if dimension > 0:
    dimension_config[key] = dimension

  return key, \
         [tf.feature_column.categorical_column_with_identity(
            fname, num_buckets=num_buckets, default_value=def_value)], is_out, group



@FCGenerator
def GetCatHashColumn(key, info_dict):
  num_buckets = int(info_dict["num_buckets"])
  dtype = info_dict.get("dtype", "string")
  is_out = info_dict.get('is_out', True) not in ("False", "false", "0")
  group = info_dict.get('group', 'default')
  fname = info_dict.get('fname', key)
  dimension = int(info_dict.get('dimension', '0'))
  if dimension > 0:
    dimension_config[key] = dimension
  #print("cathash:", key, fname, is_out) 
  if dtype == "int64":
    return key,  \
          [tf.feature_column.categorical_column_with_hash_bucket(
              fname, hash_bucket_size=num_buckets, dtype=tf.int64)], is_out, group
  else:
    return key,  \
          [tf.feature_column.categorical_column_with_hash_bucket(
              fname, hash_bucket_size=num_buckets, dtype=tf.string)], is_out, group
  
@FCGenerator
def GetCatHashColumnSelf(key, info_dict):
  num_buckets = int(info_dict["num_buckets"])
  dtype = info_dict.get("dtype", "string")
  is_out = info_dict.get('is_out', True) not in ("False", "false", "0")
  group = info_dict.get('group', 'default')
  fname = info_dict.get('fname', key)
  seed = info_dict.get('seed',249997)
  dimension = int(info_dict.get('dimension', '0'))
  if dimension > 0:
    dimension_config[key] = dimension
  if dtype == "int64":
    return key,  \
          [tf.feature_column.categorical_column_with_hash_bucket(
              fname, hash_bucket_size=num_buckets, dtype=tf.int64)], is_out, group
  else:
    return key,  \
          [tf.feature_column.categorical_column_with_hash_bucket(
              fname, hash_bucket_size=num_buckets, dtype=tf.string)], is_out, group
 
@FCGenerator
def GetCatVocabColumn(key, info_dict):
  dtype = info_dict.get("dtype", "string")
  vocab = info_dict["vocab"].split(',')
  num_oov_buckets = int(info_dict.get("num_oov_buckets", 10))
  is_out = info_dict.get('is_out', True) not in ("False", "false", "0")
  group = info_dict.get('group', 'default')
  fname = info_dict.get('fname', key)
  dimension = int(info_dict.get('dimension', '0'))
  if dimension > 0:
    dimension_config[key] = dimension

  if dtype == "int64":
    return key, \
           [tf.feature_column.categorical_column_with_vocabulary_list(
              fname, vocabulary_list=[int(w) for w in vocab], num_oov_buckets=num_oov_buckets)], is_out, group
  else:
    return key, \
           [tf.feature_column.categorical_column_with_vocabulary_list(
              fname, vocabulary_list=vocab, num_oov_buckets=num_oov_buckets)], is_out, group

@FCGenerator
def GetCatVocabFileColumn(key, info_dict):
  dtype = info_dict.get("dtype", "string")
  vocab_file = info_dict["vocab_file"]
  num_oov_buckets = int(info_dict.get("num_oov_buckets", 10))
  is_out = info_dict.get('is_out', True) not in ("False", "false", "0")
  group = info_dict.get('group', 'default')
  fname = info_dict.get('fname', key)
  dimension = int(info_dict.get('dimension', '0'))
  if dimension > 0:
    dimension_config[key] = dimension
  if dtype == "int64":
    return key, \
           [tf.feature_column.categorical_column_with_vocabulary_file(
              fname, vocabulary_file=vocab_file, dtype=tf.int64,
                num_oov_buckets=num_oov_buckets)], is_out, group
  else:
    return key, \
           [tf.feature_column.categorical_column_with_vocabulary_file(
              fname, vocabulary_file=vocab_file, dtype=tf.string,
                num_oov_buckets=num_oov_buckets)], is_out, group

@FCGenerator
def GetCrossColumn(key, info_dict):
  #keys = [cur_columns.get(c, c) for c_lst in info_dict["keys"].split(',') for c in c_lst]
  keys = [c for k in info_dict['keys'].split(',') for c in cur_columns.get(k, [k])]
  num_buckets = int(info_dict["num_buckets"])
  is_out = info_dict.get('is_out', True) not in ("False", "false", "0")
  group = info_dict.get('group', 'default')
  dimension = int(info_dict.get('dimension', '0'))
  if dimension > 0:
    dimension_config[key] = dimension
  
  return key, \
         [tf.feature_column.crossed_column(
          keys=keys, hash_bucket_size=num_buckets)], is_out, group

@FCGenerator
def GetIndicatorColumn(key, info_dict):
  cat_col = cur_columns.get(info_dict['cat_col'])[0]
  is_out = info_dict.get('is_out', True) not in ('False', 'false', '0')
  group = info_dict.get('group', 'default')
  dimension = int(info_dict.get('dimension', '0'))
  if dimension > 0:
    dimension_config[key] = dimension
  return key, \
         [tf.feature_column.indicator_column(cat_col)], is_out, group

@FCGenerator
def GetEmbeddingColumn(key, info_dict):
  cat_col = cur_columns.get(info_dict["cat_col"])[0]
  dimension = int(info_dict["dimension"])
  combiner = info_dict.get("combiner", "mean")
  is_out = info_dict.get('is_out', True) not in ("False", "false", "0")
  group = info_dict.get('group', 'default')
  ckpt_to_load_from = info_dict.get('ckpt_to_load_from',None)
  tensor_name_in_ckpt = info_dict.get('tensor_name_in_ckpt',None)
  emb_npz = info_dict.get('emb_npz',None)
  initializer = tf.initializers.uniform_unit_scaling()
  if emb_npz:
    embedding = np.load(emb_npz)
    initializer = tf.constant_initializer(embedding)
  return key, \
         [tf.feature_column.embedding_column(         
            categorical_column=cat_col,
            dimension=dimension,
            combiner=combiner,
            initializer=initializer,
            ckpt_to_load_from=ckpt_to_load_from,
            tensor_name_in_ckpt=tensor_name_in_ckpt)], is_out, group

@FCGenerator
def GetWeightedCatColumn(key, info_dict):
  cat_col = cur_columns.get(info_dict['cat_col'])[0]
  weight_key = info_dict.get('fname', key)
  is_out = info_dict.get('is_out', True) not in ('False', 'false', '0')
  group = info_dict.get('group', 'default')
  dtype = info_dict.get('dtype', 'float')
  if dtype == 'float':
    dtype = tf.float32
  else:
    dtype = tf.int32
  return key, \
         [tf.feature_column.weighted_categorical_column(
            categorical_column=cat_col,
            weight_feature_key=weight_key,
            dtype=dtype)], is_out, group

@SharedFCGenerator
def GetSharedEmbeddingColumn(key, info_dict):
  cat_col = [c for k in info_dict['cat_col'].split(',') for c in cur_columns.get(k)]
  dimension = int(info_dict["dimension"])
  combiner = info_dict.get("combiner", "mean")
  is_out = info_dict.get('is_out', True) not in ("False", "false", "0")
  group = info_dict.get('group', 'default')
  ckpt_to_load_from = info_dict.get('ckpt_to_load_from',None)
  tensor_name_in_ckpt = info_dict.get('tensor_name_in_ckpt',None)

  return key, \
         tf.feature_column.shared_embedding_columns(         
            categorical_columns=cat_col,
            dimension=dimension,
            combiner=combiner,
            initializer=tf.initializers.uniform_unit_scaling(),
            ckpt_to_load_from=ckpt_to_load_from,
            tensor_name_in_ckpt=tensor_name_in_ckpt), is_out, group

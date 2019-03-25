import tensorflow as tf


def random_pooling(feats, output_1d_size, batch_size):
  is_input_tensor = type(feats) is tf.Tensor
  _, H, W, C = tf.convert_to_tensor(feats[0]).shape.as_list()
  
  if is_input_tensor:
    feats = [feats]
  
  # convert all inputs to tensors
  feats = [tf.convert_to_tensor(feats_i) for feats_i in feats]
  
  _, H, W, C = feats[0].shape.as_list()
  feats_sampled_0, indices = random_sampling(feats[0], output_1d_size ** 2, H, W, C, batch_size)
  res = [feats_sampled_0]
  for i in range(1, len(feats)):
    C = feats[i].shape.as_list()[-1]
    feats_sampled_i, _ = random_sampling(feats[i], -1, H, W, C, batch_size, indices)
    res.append(feats_sampled_i)
  r = []
  for feats_sampled in res:
    C = feats_sampled.shape.as_list()[-1]
    f = tf.reshape(feats_sampled, [batch_size, output_1d_size, output_1d_size, C])
    r.append(f)
  
  if is_input_tensor:
    return r[0]
  return r


def random_sampling(tensor_in, n, H, W, C, batch_size, indices=None):
  S = H * W
  tensor_NSC = tf.reshape(tensor_in, [batch_size, S, C])
  all_indices = list(range(S))
  shuffled_indices = tf.random_shuffle(all_indices)
  indices = tf.gather(shuffled_indices, list(range(n)), axis=0) if indices is None else indices
  res = tf.gather(tensor_NSC, indices, axis=1)
  return res, indices

import tensorflow as tf

from utils import contextual_similarity_utills
from utils import training_utils

log = training_utils.get_logger()


def id_mrf_loss_sum_for_layers(y_true_vgg, y_pred_vgg, layers, mrf_config, batch_size):
  id_mrf_losses = []
  for layer in layers:
    l = id_mrf_on_features(y_pred_vgg[layer], y_true_vgg[layer], mrf_config, batch_size)
    id_mrf_losses.append(l)
  id_mrf_loss_total = tf.reduce_sum(id_mrf_losses)
  return id_mrf_loss_total


def id_mrf_on_features(y_pred_vgg, y_true_vgg, config, batch_size):
  if config['crop_quarters'] is True:
    y_pred_vgg = crop_quarters(y_pred_vgg)
    y_true_vgg = crop_quarters(y_true_vgg)
  
  _, fH, fW, fC = y_pred_vgg.shape.as_list()
  if fH * fW <= config['max_sampling_1d_size'] ** 2:
    log.info(' #### Skipping random pooling ...')
  else:
    log.info(' #### pooling %dx%d out of %dx%d' % (
      config['max_sampling_1d_size'], config['max_sampling_1d_size'], fH, fW))
    y_pred_vgg, y_true_vgg = random_pooling([y_pred_vgg, y_true_vgg],
                                            output_1d_size=config['max_sampling_1d_size'],
                                            batch_size=batch_size)
  
  return mrf_loss(y_pred_vgg, y_true_vgg, batch_size, nnsigma=config['nn_stretch_sigma'])


def crop_quarters(feature_tensor):
  N, fH, fW, fC = feature_tensor.shape.as_list()
  quarters_list = []
  quarter_size = [N, round(fH / 2), round(fW / 2), fC]
  quarters_list.append(tf.slice(feature_tensor, [0, 0, 0, 0], quarter_size))
  quarters_list.append(tf.slice(feature_tensor, [0, round(fH / 2), 0, 0], quarter_size))
  quarters_list.append(tf.slice(feature_tensor, [0, 0, round(fW / 2), 0], quarter_size))
  quarters_list.append(tf.slice(feature_tensor, [0, round(fH / 2), round(fW / 2), 0], quarter_size))
  feature_tensor = tf.concat(quarters_list, axis=0)
  return feature_tensor


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
    feats_sampled_i, _ = random_sampling(feats[i], -1, H, W, C, batch_size, indices)
    res.append(feats_sampled_i)
  
  res = [tf.reshape(feats_sampled_i, [batch_size, output_1d_size, output_1d_size, C]) for
         feats_sampled_i in res]
  if is_input_tensor:
    return res[0]
  return res


def mrf_loss(y_pred_vgg, y_true_vgg, batch_size, nnsigma):
  y_pred_vgg = tf.convert_to_tensor(y_pred_vgg, dtype=tf.float32)
  y_true_vgg = tf.convert_to_tensor(y_true_vgg, dtype=tf.float32)
  
  cs = contextual_similarity_utills.calculate_cs(y_true_vgg, y_pred_vgg, batch_size, nnsigma)
  height_width_axis = [1, 2]
  cs_sim_max = tf.reduce_max(cs, axis=height_width_axis)
  contextual_similarities = tf.reduce_mean(cs_sim_max, axis=[1])
  loss = -tf.log(contextual_similarities)
  loss = tf.reduce_mean(loss)
  return loss


def random_sampling(tensor_in, n, H, W, C, batch_size, indices=None):
  S = H * W
  tensor_NSC = tf.reshape(tensor_in, [batch_size, S, C])
  all_indices = list(range(S))
  shuffled_indices = tf.random_shuffle(all_indices)
  indices = tf.gather(shuffled_indices, list(range(n)), axis=0) if indices is None else indices
  res = tf.gather(tensor_NSC, indices, axis=1)
  return res, indices

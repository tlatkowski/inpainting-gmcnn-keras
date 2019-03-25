import tensorflow as tf

from utils import contextual_similarity_utills
from utils import other_utils
from utils import sampling_utils
from utils import training_utils

log = training_utils.get_logger()


def id_mrf_loss_sum_for_layers(y_true_vgg, y_pred_vgg, mask, layers, mrf_config,
                               batch_size):
  id_mrf_losses = []
  for layer in layers:
    mask_resized = other_utils.resize_mask(mask, y_pred_vgg[layer])
    l = id_mrf_on_features(y_pred_vgg[layer], y_true_vgg[layer], mask_resized, mrf_config,
                           batch_size)
    id_mrf_losses.append(l)
  id_mrf_loss_total = tf.reduce_sum(id_mrf_losses)
  return id_mrf_loss_total


def id_mrf_on_features(y_pred_vgg, y_true_vgg, mask_resized, config, batch_size):
  if config['crop_quarters'] is True:
    y_pred_vgg = other_utils.crop_quarters(y_pred_vgg)
    y_true_vgg = other_utils.crop_quarters(y_true_vgg)
  
  _, fH, fW, fC = y_pred_vgg.shape.as_list()
  if fH * fW <= config['max_sampling_1d_size'] ** 2:
    log.info(' #### Skipping random pooling ...')
  else:
    log.info(' #### pooling %dx%d out of %dx%d' % (
      config['max_sampling_1d_size'], config['max_sampling_1d_size'], fH, fW))
    y_pred_vgg, y_true_vgg, mask_resized = sampling_utils.random_pooling(
      [y_pred_vgg, y_true_vgg, mask_resized], output_1d_size=config['max_sampling_1d_size'],
      batch_size=batch_size)
  
  return mrf_loss(y_pred_vgg, y_true_vgg, mask_resized, batch_size,
                  nnsigma=config['nn_stretch_sigma'])


def mrf_loss(y_pred_vgg, y_true_vgg, mask_resized, batch_size, nnsigma):
  y_pred_vgg = tf.convert_to_tensor(y_pred_vgg, dtype=tf.float32)
  y_true_vgg = tf.convert_to_tensor(y_true_vgg, dtype=tf.float32)
  
  cs = contextual_similarity_utills.calculate_cs(y_true_vgg, y_pred_vgg, batch_size, nnsigma)
  cs *= tf.expand_dims(mask_resized[:, :, :, 0], 3)
  height_width_axis = [1, 2]
  cs_sim_max = tf.reduce_max(cs, axis=height_width_axis)
  contextual_similarities = tf.reduce_mean(cs_sim_max, axis=[1])  # TODO norm by num of mask pixels
  loss = -tf.log(contextual_similarities)
  loss = tf.reduce_mean(loss)  # mean over batches
  return loss


def count_mean_in_mask(cs_sim_max, mask):
  mask_one_channel = tf.expand_dims(mask[:, :, :, 0], 3)
  num_mask_pixels = tf.reduce_sum(mask_one_channel, axis=[1, 2])
  sum_maximum = tf.reduce_sum(cs_sim_max, axis=[1], keepdims=True)
  contextual_similarities = sum_maximum / num_mask_pixels
  return contextual_similarities

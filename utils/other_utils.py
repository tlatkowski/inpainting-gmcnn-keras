import tensorflow as tf


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


def resize_mask(mask, vgg_features):
  h = vgg_features.shape.as_list()[1]
  mask_resized = tf.image.resize_nearest_neighbor(mask, size=(h, h))
  return mask_resized

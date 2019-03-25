import tensorflow as tf


def center_by_predicted(y_pred_vgg: tf.Tensor, y_true_vgg: tf.Tensor):
  # assuming both input are of the same size
  
  # calculate stas over [batch, height, width], expecting 1x1xDepth tensor
  axes = [0, 1, 2]
  meanT, varT = tf.nn.moments(y_pred_vgg, axes)
  # we do not divide by std since its causing the histogram
  # for the final cs to be very thin, so the NN weights
  # are not distinctive, giving similar values for all patches.
  # stdT = tf.sqrt(varT, "stdT")
  # correct places with std zero
  # stdT[tf.less(stdT, tf.constant(0.001))] = tf.constant(1)
  y_pred_vgg_centered = y_pred_vgg - meanT
  y_true_vgg_centered = y_true_vgg - meanT
  return y_pred_vgg_centered, y_true_vgg_centered


def l2_normalize_channel_wise(features):
  norms = tf.norm(features, ord='euclidean', axis=3)
  # expanding the norms tensor to support broadcast division
  norms_expanded = tf.expand_dims(norms, axis=3)
  features_norm = tf.divide(features, norms_expanded)
  return features_norm

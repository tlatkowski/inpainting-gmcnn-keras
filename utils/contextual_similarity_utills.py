import tensorflow as tf


def calculate_contextual_similarity(scaled_distances, sigma=float(0.1), b=float(1.0),
                                    axis_for_normalization=3):
  cs_weights_before_normalization = tf.exp((b - scaled_distances) / sigma,
                                           name='weights_before_normalization')
  cs_sim = sum_normalize(cs_weights_before_normalization, axis_for_normalization)
  return cs_sim


def extract_patches(y_pred_vgg):
  # patch decomposition
  patch_size = 1
  patches_as_depth_vectors = tf.extract_image_patches(
    images=y_pred_vgg, ksizes=[1, patch_size, patch_size, 1],
    strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID',
    name='patches_as_depth_vectors')
  
  patches_NHWC = tf.reshape(patches_as_depth_vectors,
                            shape=[-1, patch_size, patch_size,
                                   patches_as_depth_vectors.shape[3].value],
                            name='patches_PHWC')
  
  patches_HWCN = tf.transpose(patches_NHWC,
                              perm=[1, 2, 3, 0],
                              name='patches_HWCP')  # tf.conv2 ready format
  
  return patches_HWCN


def calculate_relative_distances(raw_distances, axis=3):
  epsilon = 1e-5
  div = tf.reduce_min(raw_distances, axis=axis, keepdims=True)
  # div = tf.reduce_mean(self.raw_distances, axis=axis, keep_dims=True)
  relative_dist = raw_distances / (div + epsilon)
  return relative_dist


def calculate_cs_sim(y_true_vgg, y_pred_vgg, batch_size, sigma=float(1.0), b=float(1.0)):
  with tf.name_scope('contextual_similarity'):
    # prepare feature before calculating cosine distance
    y_pred_vgg, y_true_vgg = center_by_predicted(y_pred_vgg, y_true_vgg)
    with tf.name_scope('y_pred_vgg/norm'):
      y_pred_vgg = l2_normalize_channel_wise(y_pred_vgg)
    with tf.name_scope('y_true_vgg/norm'):
      y_true_vgg = l2_normalize_channel_wise(y_true_vgg)
      # work seperatly for each example in dim 1
      cosine_dist_l = []
      for i in range(batch_size):
        y_pred_vgg_i = tf.expand_dims(y_pred_vgg[i, :, :, :], 0)
        y_true_vgg_i = tf.expand_dims(y_true_vgg[i, :, :, :], 0)
        patches_i = extract_patches(y_pred_vgg_i)
        cosine_dist_i = tf.nn.conv2d(y_true_vgg_i, patches_i, strides=[1, 1, 1, 1],
                                     padding='VALID', use_cudnn_on_gpu=True, name='cosine_dist')
        cosine_dist_l.append(cosine_dist_i)
      
      cosine_dist = tf.concat(cosine_dist_l, axis=0)
      
      cosine_dist_zero_to_one = -(cosine_dist - 1) / 2
      
      relative_dist = calculate_relative_distances(cosine_dist_zero_to_one)
      cs_sim = calculate_contextual_similarity(relative_dist)
      return cs_sim


def center_by_predicted(y_pred_vgg, y_true_vgg):
  # assuming both input are of the same size
  
  # calculate stas over [batch, height, width], expecting 1x1xDepth tensor
  axes = [0, 1, 2]
  meanT, varT = tf.nn.moments(
    y_pred_vgg, axes, name='TFeatures/moments')
  # we do not divide by std since its causing the histogram
  # for the final cs to be very thin, so the NN weights
  # are not distinctive, giving similar values for all patches.
  # stdT = tf.sqrt(varT, "stdT")
  # correct places with std zero
  # stdT[tf.less(stdT, tf.constant(0.001))] = tf.constant(1)
  with tf.name_scope('y_pred_vgg/centering'):
    y_pred_vgg_centered = y_pred_vgg - meanT
  with tf.name_scope('y_true_vgg/centering'):
    y_true_vgg_centered = y_true_vgg - meanT
  
  return y_pred_vgg_centered, y_true_vgg_centered


def l2_normalize_channel_wise(features):
  norms = tf.norm(features, ord='euclidean', axis=3, name='norm')
  # expanding the norms tensor to support broadcast division
  norms_expanded = tf.expand_dims(norms, 3)
  features = tf.divide(features, norms_expanded, name='normalized')
  return features


def sum_normalize(cs, axis=3):
  reduce_sum = tf.reduce_sum(cs, axis, keepdims=True, name='sum')
  return tf.divide(cs, reduce_sum, name='sumNormalized')

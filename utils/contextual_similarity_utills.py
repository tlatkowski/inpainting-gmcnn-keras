import tensorflow as tf


# def calculate_cs(y_true_vgg, y_pred_vgg, batch_size, sigma=float(1.0), b=float(1.0)):
#   with tf.name_scope('contextual_similarity'):
#     # prepare feature before calculating cosine distance
#     y_pred_vgg, y_true_vgg = center_by_predicted(y_pred_vgg, y_true_vgg)
#     with tf.name_scope('y_pred_vgg/norm'):
#       y_pred_vgg = l2_normalize_channel_wise(y_pred_vgg)
#     with tf.name_scope('y_true_vgg/norm'):
#       y_true_vgg = l2_normalize_channel_wise(y_true_vgg)
#       # work seperatly for each example in dim 1
#       cosine_dist_l = []
#       for i in range(batch_size):
#         y_pred_vgg_i = tf.expand_dims(y_pred_vgg[i, :, :, :], 0)
#         y_true_vgg_i = tf.expand_dims(y_true_vgg[i, :, :, :], 0)
#         patches_i = extract_patches(y_pred_vgg_i)
#         cosine_dist_i = tf.nn.conv2d(y_true_vgg_i, patches_i, strides=[1, 1, 1, 1],
#                                      padding='VALID', use_cudnn_on_gpu=True, name='cosine_dist')
#         cosine_dist_l.append(cosine_dist_i)
#
#       cosine_dist = tf.concat(cosine_dist_l, axis=0)
#
#       cosine_dist_zero_to_one = -(cosine_dist - 1) / 2
#
#       relative_dist = calculate_relative_distances(cosine_dist_zero_to_one)
#       cs_sim = calculate_contextual_similarity(relative_dist, sigma, b)
#       return cs_sim

def calculate_cs(y_true_vgg, y_pred_vgg, batch_size, sigma=float(1.0), b=float(1.0)):
  with tf.name_scope('contextual_similarity'):
    cosine_dist_zero_to_one = calculate_cosine_distances(y_true_vgg, y_pred_vgg, batch_size)
    relative_dist = calculate_relative_distances(cosine_dist_zero_to_one)
    cs = calculate_contextual_similarity(relative_dist, sigma, b)
    return cs


def calculate_cosine_distances(y_true_vgg, y_pred_vgg, batch_size):
  # prepare feature before calculating cosine distance
  y_pred_vgg, y_true_vgg = center_by_predicted(y_pred_vgg, y_true_vgg)
  y_pred_vgg = l2_normalize_channel_wise(y_pred_vgg)
  y_true_vgg = l2_normalize_channel_wise(y_true_vgg)
  
  cosine_distances_per_batch = []
  for i in range(batch_size):
    y_pred_vgg_i = tf.expand_dims(y_pred_vgg[i, :, :, :], 0)
    y_true_vgg_i = tf.expand_dims(y_true_vgg[i, :, :, :], 0)
    patches_i = extract_patches(y_pred_vgg_i)
    cosine_dist_i = tf.nn.conv2d(y_true_vgg_i, patches_i, strides=[1, 1, 1, 1],
                                 padding='VALID', use_cudnn_on_gpu=True, name='cosine_dist')
    cosine_distances_per_batch.append(cosine_dist_i)
  
  cosine_distances = tf.concat(cosine_distances_per_batch, axis=0)
  return cosine_distances


def calculate_relative_distances(raw_distances, axis=3, epsilon=1e-5):
  div = tf.reduce_min(raw_distances, axis=axis, keepdims=True)
  relative_dist = raw_distances / (div + epsilon)
  return relative_dist


def calculate_contextual_similarity(scaled_distances, sigma=float(0.1), b=float(1.0)):
  cs = tf.exp((b - scaled_distances) / sigma)
  reduce_sum = tf.reduce_sum(cs, axis=3, keepdims=True)
  cs_norm = tf.divide(cs, reduce_sum)
  return cs_norm


def center_by_predicted(y_pred_vgg, y_true_vgg):
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


def extract_patches(vgg_features, patch_size=1):
  patches_as_depth_vectors = tf.extract_image_patches(images=vgg_features,
                                                      ksizes=[1, patch_size, patch_size, 1],
                                                      strides=[1, 1, 1, 1], rates=[1, 1, 1, 1],
                                                      padding='VALID',
                                                      name='patches_as_depth_vectors')
  
  patches_NHWC = tf.reshape(patches_as_depth_vectors,
                            shape=[-1, patch_size, patch_size,
                                   patches_as_depth_vectors.shape[3].value],
                            name='patches_PHWC')
  
  patches_HWCN = tf.transpose(patches_NHWC, perm=[1, 2, 3, 0], name='patches_HWCP')
  
  return patches_HWCN

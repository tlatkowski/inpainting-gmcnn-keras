import tensorflow as tf

from utils import norm_utils


def calculate_cs(y_true_vgg, y_pred_vgg, batch_size, sigma=float(1.0), b=float(1.0)):
  with tf.name_scope('contextual_similarity'):
    cosine_dist_zero_to_one = calculate_cosine_distances(y_true_vgg, y_pred_vgg, batch_size)
    relative_dist = calculate_relative_distances(cosine_dist_zero_to_one)
    cs = calculate_contextual_similarity(relative_dist, sigma, b)
    return cs

def normalize_inputs(y_true_vgg, y_pred_vgg):
  y_true_vgg = tf.nn.l2_normalize(y_true_vgg, axis=-1)
  y_pred_vgg = tf.nn.l2_normalize(y_pred_vgg, axis=-1)
  return y_true_vgg, y_pred_vgg


def calculate_cosine_distances(y_true_vgg, y_pred_vgg, batch_size):
  # prepare feature before calculating cosine distance
  # y_pred_vgg, y_true_vgg = norm_utils.center_by_predicted(y_pred_vgg, y_true_vgg)
  # y_pred_vgg, y_true_vgg = norm_utils.center_by_predicted(y_true_vgg, y_pred_vgg) # norm by true
  # y_pred_vgg = norm_utils.l2_normalize_channel_wise(y_pred_vgg)
  # y_true_vgg = norm_utils.l2_normalize_channel_wise(y_true_vgg)
  y_true_vgg, y_pred_vgg = normalize_inputs(y_true_vgg, y_pred_vgg)
  
  cosine_distances_per_batch = []
  for i in range(batch_size):
    y_pred_vgg_i = tf.expand_dims(y_pred_vgg[i, :, :, :], 0)
    y_true_vgg_i = tf.expand_dims(y_true_vgg[i, :, :, :], 0)
    # patches_i = extract_patches(y_pred_vgg_i)
    # cosine_dist_i = tf.nn.conv2d(y_true_vgg_i, patches_i, strides=[1, 1, 1, 1],
    #                              padding='VALID', use_cudnn_on_gpu=True, name='cosine_dist')
    patches_i = extract_patches(y_true_vgg_i)
    cosine_dist_i = tf.nn.conv2d(y_pred_vgg_i, patches_i, strides=[1, 1, 1, 1],
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

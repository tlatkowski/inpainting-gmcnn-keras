from enum import Enum

import tensorflow as tf


class Distance(Enum):
  L2 = 0
  DotProduct = 1


class ContextualSimilarity:
  
  def __init__(self, sigma=float(0.1), b=float(1.0)):
    self.b = b
    self.sigma = sigma
  
  def calculate_contextual_similarity(self, scaled_distances, axis_for_normalization=3):
    self.scaled_distances = scaled_distances
    self.cs_weights_before_normalization = tf.exp((self.b - scaled_distances) / self.sigma,
                                                  name='weights_before_normalization')
    self.cs_NHWC = sum_normalize(self.cs_weights_before_normalization, axis_for_normalization)
  
  def reversed_direction_CS(self):
    cs_flow_opposite = ContextualSimilarity(self.sigma, self.b)
    cs_flow_opposite.raw_distances = self.raw_distances
    work_axis = [1, 2]
    relative_dist = cs_flow_opposite.calc_relative_distances(axis=work_axis)
    cs_flow_opposite.__calculate_contextual_similarity(relative_dist, work_axis)
    return cs_flow_opposite
  
  def calc_relative_distances(self, axis=3):
    epsilon = 1e-5
    div = tf.reduce_min(self.raw_distances, axis=axis, keepdims=True)
    # div = tf.reduce_mean(self.raw_distances, axis=axis, keep_dims=True)
    relative_dist = self.raw_distances / (div + epsilon)
    return relative_dist
  
  def weighted_average_dist(self, axis=3):
    if not hasattr(self, 'raw_distances'):
      raise Exception('raw_distances property does not exists. cant calculate weighted average l2')
    
    multiply = self.raw_distances * self.cs_NHWC
    return tf.reduce_sum(multiply, axis=axis, name='weightedDistPerPatch')
  
  def patch_decomposition(self, y_pred_vgg):
    # patch decomposition
    patch_size = 1
    patches_as_depth_vectors = tf.extract_image_patches(
      images=y_pred_vgg, ksizes=[1, patch_size, patch_size, 1],
      strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID',
      name='patches_as_depth_vectors')
    
    self.patches_NHWC = tf.reshape(patches_as_depth_vectors,
                                   shape=[-1, patch_size, patch_size,
                                          patches_as_depth_vectors.shape[3].value],
                                   name='patches_PHWC')
    
    self.patches_HWCN = tf.transpose(self.patches_NHWC,
                                     perm=[1, 2, 3, 0],
                                     name='patches_HWCP')  # tf.conv2 ready format
    
    return self.patches_HWCN


def create(y_true_vgg, y_pred_vgg, distance=Distance.DotProduct, nnsigma=float(1.0), b=float(1.0)):
  if distance.value == Distance.DotProduct.value:
    cs_flow = create_using_dot_product(y_true_vgg, y_pred_vgg, nnsigma, b)
  elif distance.value == Distance.L2.value:
    cs_flow = create_using_l2_norm(y_true_vgg, y_pred_vgg, nnsigma, b)
  else:
    raise "not supported distance " + distance.__str__()
  return cs_flow


def create_using_l2_norm(y_true_vgg, y_pred_vgg, sigma=float(0.1), b=float(1.0)):
  cs_flow = ContextualSimilarity(sigma, b)
  with tf.name_scope('CS'):
    sT = y_pred_vgg.shape.as_list()
    sI = y_true_vgg.shape.as_list()
    
    Ivecs = tf.reshape(y_true_vgg, (sI[0], -1, sI[3]))
    Tvecs = tf.reshape(y_pred_vgg, (sI[0], -1, sT[3]))
    r_Ts = tf.reduce_sum(Tvecs * Tvecs, 2)
    r_Is = tf.reduce_sum(Ivecs * Ivecs, 2)
    raw_distances_list = []
    for i in range(sT[TensorAxis.N]):
      Ivec, Tvec, r_T, r_I = Ivecs[i], Tvecs[i], r_Ts[i], r_Is[i]
      A = tf.matmul(Tvec, tf.transpose(Ivec))
      cs_flow.A = A
      # A = tf.matmul(Tvec, tf.transpose(Ivec))
      r_T = tf.reshape(r_T, [-1, 1])  # turn to column vector
      dist = r_T - 2 * A + r_I
      cs_shape = sI[:3] + [dist.shape[0].value]
      cs_shape[0] = 1
      dist = tf.reshape(tf.transpose(dist), cs_shape)
      # protecting against numerical problems, dist should be positive
      dist = tf.maximum(float(0.0), dist)
      # dist = tf.sqrt(dist)
      raw_distances_list += [dist]
    
    cs_flow.raw_distances = tf.convert_to_tensor(
      [tf.squeeze(raw_dist, axis=0) for raw_dist in raw_distances_list])
    
    relative_dist = cs_flow.calc_relative_distances()
    cs_flow.calculate_contextual_similarity(relative_dist)
    return cs_flow


def create_using_dot_product(y_true_vgg, y_pred_vgg, sigma=float(1.0), b=float(1.0),
                             batch_size=1):
  cs_flow = ContextualSimilarity(sigma, b)
  with tf.name_scope('CS'):
    # prepare feature before calculating cosine distance
    y_pred_vgg, y_true_vgg = center_by_predicted(y_pred_vgg, y_true_vgg)
    with tf.name_scope('TFeatures'):
      y_pred_vgg = l2_normalize_channelwise(y_pred_vgg)
    with tf.name_scope('IFeatures'):
      y_true_vgg = l2_normalize_channelwise(y_true_vgg)
      # work seperatly for each example in dim 1
      cosine_dist_l = []
      for i in range(batch_size):
        y_pred_vgg_i = tf.expand_dims(y_pred_vgg[i, :, :, :], 0)
        y_true_vgg_i = tf.expand_dims(y_true_vgg[i, :, :, :], 0)
        patches_i = cs_flow.patch_decomposition(y_pred_vgg_i)
        cosine_dist_i = tf.nn.conv2d(y_true_vgg_i, patches_i, strides=[1, 1, 1, 1],
                                     padding='VALID', use_cudnn_on_gpu=True, name='cosine_dist')
        cosine_dist_l.append(cosine_dist_i)
      
      cs_flow.cosine_dist = tf.concat(cosine_dist_l, axis=0)
      
      cosine_dist_zero_to_one = -(cs_flow.cosine_dist - 1) / 2
      cs_flow.raw_distances = cosine_dist_zero_to_one
      
      relative_dist = cs_flow.calc_relative_distances()
      cs_flow.calculate_contextual_similarity(relative_dist)
      return cs_flow


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


def l2_normalize_channelwise(features):
  norms = tf.norm(features, ord='euclidean', axis=3, name='norm')
  # expanding the norms tensor to support broadcast division
  norms_expanded = tf.expand_dims(norms, 3)
  features = tf.divide(features, norms_expanded, name='normalized')
  return features


def sum_normalize(cs, axis=3):
  reduce_sum = tf.reduce_sum(cs, axis, keepdims=True, name='sum')
  return tf.divide(cs, reduce_sum, name='sumNormalized')

import numpy as np
from keras.backend import tensorflow_backend as K

from models import vgg
from utils import gaussian_utils
from utils import id_mrf


def reconstruction_loss(y_true, y_pred):
  diff = K.abs(y_pred - y_true)
  l1 = K.mean(diff, axis=[1, 2, 3])
  return l1


def wasserstein_loss(y_true, y_pred, wgan_loss_weight=1.0):
  """Calculates the Wasserstein loss for a sample batch.
  The Wasserstein loss function is very simple to calculate. In a standard GAN, the
  discriminator has a sigmoid output, representing the probability that samples are
  real or generated. In Wasserstein GANs, however, the output is linear with no
  activation function! Instead of being constrained to [0, 1], the discriminator wants
  to make the distance between its output for real and generated samples as
  large as possible.
  The most natural way to achieve this is to label generated samples -1 and real
  samples 1, instead of the 0 and 1 used in normal GANs, so that multiplying the
  outputs by the labels will give you the loss immediately.
  Note that the nature of this loss means that it can be (and frequently will be)
  less than 0."""
  return wgan_loss_weight * K.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples, # TODO apply mask to gp_loss
                          gradient_penalty_weight):
  """Calculates the gradient penalty loss for a batch of "averaged" samples.
  In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the
  loss function that penalizes the network if the gradient norm moves away from 1.
  However, it is impossible to evaluate this function at all points in the input
  space. The compromise used in the paper is to choose random points on the lines
  between real and generated samples, and check the gradients at these points. Note
  that it is the gradient w.r.t. the input averaged samples, not the weights of the
  discriminator, that we're penalizing!
  In order to evaluate the gradients, we must first run samples through the generator
  and evaluate the loss. Then we get the gradients of the discriminator w.r.t. the
  input averaged samples. The l2 norm and penalty can then be calculated for this
  gradient.
  Note that this loss function requires the original averaged samples as input, but
  Keras only supports passing y_true and y_pred to loss functions. To get around this,
  we make a partial() of the function with the averaged_samples argument, and use that
  for model training."""
  # first get the gradients:
  #   assuming: - that y_pred has dimensions (batch_size, 1)
  #             - averaged_samples has dimensions (batch_size, nbr_features)
  # gradients afterwards has dimension (batch_size, nbr_features), basically
  # a list of nbr_features-dimensional gradient vectors
  gradients = K.gradients(y_pred, averaged_samples)[0]
  # compute the euclidean norm by squaring ...
  gradients_sqr = K.square(gradients)
  #   ... summing over the rows ...
  gradients_sqr_sum = K.sum(gradients_sqr,
                            axis=np.arange(1, len(gradients_sqr.shape)))
  #   ... and sqrt
  gradient_l2_norm = K.sqrt(gradients_sqr_sum)
  # compute lambda * (1 - ||grad||)^2 still for each single sample
  gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
  # return the mean as loss over all the batch samples
  return K.mean(gradient_penalty)


def confidence_reconstruction_loss(y_true, y_pred, mask, num_steps, gaussian_kernel_size,
                                   gaussian_kernel_std):
  mask_blurred = gaussian_utils.blur_mask(mask, num_steps, gaussian_kernel_size,
                                          gaussian_kernel_std)
  valid_mask = 1 - mask
  diff = K.abs(y_true - y_pred)
  # l1 = K.mean(diff * valid_mask + diff * mask_blurred, axis=[1, 2, 3])
  l1 = K.mean(diff * valid_mask + diff * mask_blurred)
  return l1


def id_mrf_loss(y_true, y_pred, mask, nn_stretch_sigma, batch_size, vgg_16_layers,
                id_mrf_style_weight, id_mrf_content_weight, id_mrf_loss_weight=1.0,
                use_original_vgg_shape=False):
  vgg_model = vgg.build_vgg16(y_pred, use_original_vgg_shape, vgg_16_layers)
  
  y_pred_vgg = vgg_model(y_pred)
  y_true_vgg = vgg_model(y_true)
  content_layers = [0]
  style_layers = [1, 2]
  id_mrf_config = dict()
  id_mrf_config['crop_quarters'] = False
  id_mrf_config['max_sampling_1d_size'] = 65
  id_mrf_config['nn_stretch_sigma'] = nn_stretch_sigma
  id_mrf_style_loss = id_mrf.id_mrf_loss_sum_for_layers(y_true_vgg, y_pred_vgg, mask,
                                                        style_layers, id_mrf_config,
                                                        batch_size)
  
  id_mrf_content_loss = id_mrf.id_mrf_loss_sum_for_layers(y_true_vgg, y_pred_vgg, mask,
                                                          content_layers, id_mrf_config,
                                                          batch_size)
  
  id_mrf_loss_total = id_mrf_style_loss * id_mrf_style_weight + id_mrf_content_loss * id_mrf_content_weight
  return id_mrf_loss_weight * id_mrf_loss_total

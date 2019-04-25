from functools import partial

from keras.models import Model, Input
from keras.optimizers import Adam
from keras.layers import Lambda
from config import main_config
from layers import custom_layers
from layers.losses import wasserstein_loss, gradient_penalty_loss, \
  confidence_reconstruction_loss, id_mrf_loss
from models.discriminator import GlobalDiscriminator, LocalDiscriminator
from models.generator import Generator
from models.wgan import WassersteinGAN
from utils import constants


class GMCNNGan(WassersteinGAN):
  
  def __init__(self, batch_size, img_height, img_width, num_channels, warm_up_generator,
               config: main_config.MainConfig, output_paths: constants.OutputPaths):
    super(GMCNNGan, self).__init__(img_height, img_width, num_channels, batch_size,
                                   config.training.wgan_training_ratio, output_paths)
    
    self.img_height = img_height
    self.img_width = img_width
    self.num_channels = num_channels
    self.warm_up_generator = warm_up_generator
    self.learning_rate = config.training.learning_rate
    self.num_gaussian_steps = config.model.num_gaussian_steps
    self.gradient_penalty_loss_weight = config.model.gradient_penalty_loss_weight
    self.id_mrf_loss_weight = config.model.id_mrf_loss_weight
    self.adversarial_loss_weight = config.model.adversarial_loss_weight
    self.nn_stretch_sigma = config.model.nn_stretch_sigma
    self.vgg_16_layers = config.model.vgg_16_layers
    self.id_mrf_style_weight = config.model.id_mrf_style_weight
    self.id_mrf_content_weight = config.model.id_mrf_content_weight
    self.gaussian_kernel_size = config.model.gaussian_kernel_size
    self.gaussian_kernel_std = config.model.gaussian_kernel_std
    self.add_mask_as_generator_input = config.model.add_mask_as_generator_input
    
    self.generator_optimizer = Adam(lr=self.learning_rate, beta_1=0.5, beta_2=0.9)
    self.discriminator_optimizer = Adam(lr=self.learning_rate, beta_1=0.5, beta_2=0.9)
    
    self.local_discriminator_raw = LocalDiscriminator(self.img_height, self.img_width,
                                                      self.num_channels, output_paths)
    self.global_discriminator_raw = GlobalDiscriminator(self.img_height, self.img_width,
                                                        self.num_channels, output_paths)
    self.generator_raw = Generator(self.img_height, self.img_width, self.num_channels,
                                   self.add_mask_as_generator_input, output_paths)
    
    # define generator model
    self.global_discriminator_raw.disable()
    self.local_discriminator_raw.disable()
    self.generator_model = self.define_generator_model(self.generator_raw,
                                                       self.local_discriminator_raw,
                                                       self.global_discriminator_raw)
    
    # define global discriminator model
    self.global_discriminator_raw.enable()
    self.generator_raw.disable()
    self.global_discriminator_model = self.define_global_discriminator(self.generator_raw,
                                                                       self.global_discriminator_raw)
    
    # define local discriminator model
    self.local_discriminator_raw.enable()
    self.global_discriminator_raw.disable()
    self.local_discriminator_model = self.define_local_discriminator(self.generator_raw,
                                                                     self.local_discriminator_raw)
  
  def define_generator_model(self, generator_raw, local_discriminator_raw,
                             global_discriminator_raw):
    
    generator_inputs_img = Input(shape=(self.img_height, self.img_width, self.num_channels))
    generator_inputs_mask = Input(shape=(self.img_height, self.img_width, self.num_channels))
    
    generator_outputs = generator_raw.model([generator_inputs_img, generator_inputs_mask])
    global_discriminator_outputs = global_discriminator_raw.model(generator_outputs)
    
    local_discriminator_outputs = local_discriminator_raw.model([generator_outputs,
                                                                 generator_inputs_mask])
    
    generator_model = Model(inputs=[generator_inputs_img, generator_inputs_mask],
                            outputs=[generator_outputs, generator_outputs,
                                     global_discriminator_outputs,
                                     local_discriminator_outputs])
    
    # this partial trick is required for passing additional parameters for loss functions
    partial_cr_loss = partial(confidence_reconstruction_loss,
                              mask=generator_inputs_mask,
                              num_steps=self.num_gaussian_steps,
                              gaussian_kernel_size=self.gaussian_kernel_size,
                              gaussian_kernel_std=self.gaussian_kernel_std)
    
    partial_cr_loss.__name__ = 'confidence_reconstruction_loss'
    
    partial_id_mrf_loss = partial(id_mrf_loss,
                                  mask=generator_inputs_mask,
                                  nn_stretch_sigma=self.nn_stretch_sigma,
                                  batch_size=self.batch_size,
                                  vgg_16_layers=self.vgg_16_layers,
                                  id_mrf_style_weight=self.id_mrf_style_weight,
                                  id_mrf_content_weight=self.id_mrf_content_weight,
                                  id_mrf_loss_weight=self.id_mrf_loss_weight)
    
    partial_id_mrf_loss.__name__ = 'id_mrf_loss'
    
    partial_wasserstein_loss = partial(wasserstein_loss,
                                       wgan_loss_weight=self.adversarial_loss_weight)
    
    partial_wasserstein_loss.__name__ = 'wasserstein_loss'
    
    if self.warm_up_generator:
      # set Wasserstein loss to 0 - total generator loss will be based only on reconstruction loss
      generator_model.compile(optimizer=self.generator_optimizer,
                              loss=[partial_cr_loss, partial_id_mrf_loss, partial_wasserstein_loss,
                                    partial_wasserstein_loss],
                              loss_weights=[1., 0., 0., 0.])
      # metrics=[metrics.psnr])
    else:
      generator_model.compile(optimizer=self.generator_optimizer,
                              loss=[partial_cr_loss, partial_id_mrf_loss, partial_wasserstein_loss,
                                    partial_wasserstein_loss])
    
    return generator_model
  
  def define_global_discriminator(self, generator_raw, global_discriminator_raw):
    generator_inputs = Input(shape=(self.img_height, self.img_width, self.num_channels))
    generator_masks = Input(shape=(self.img_height, self.img_width, self.num_channels))
    
    real_samples = Input(shape=(self.img_height, self.img_width, self.num_channels))
    fake_samples = generator_raw.model([generator_inputs, generator_masks])
    # fake_samples = generator_inputs * (1 - generator_masks) + fake_samples * generator_masks
    fake_samples = Lambda(make_comp_sample)([generator_inputs, fake_samples, generator_masks])
    
    discriminator_output_from_fake_samples = global_discriminator_raw.model(fake_samples)
    discriminator_output_from_real_samples = global_discriminator_raw.model(real_samples)
    
    averaged_samples = custom_layers.RandomWeightedAverage()([real_samples, fake_samples])
    # We then run these samples through the discriminator as well. Note that we never
    # really use the discriminator output for these samples - we're only running them to
    # get the gradient norm for the gradient penalty loss.
    averaged_samples_outputs = global_discriminator_raw.model(averaged_samples)
    
    # The gradient penalty loss function requires the input averaged samples to get
    # gradients. However, Keras loss functions can only have two arguments, y_true and
    # y_pred. We get around this by making a partial() of the function with the averaged
    # samples here.
    partial_gp_loss = partial(gradient_penalty_loss,
                              averaged_samples=averaged_samples,
                              gradient_penalty_weight=self.gradient_penalty_loss_weight)
    # Functions need names or Keras will throw an error
    partial_gp_loss.__name__ = 'gradient_penalty'
    
    global_discriminator_model = Model(inputs=[real_samples, generator_inputs, generator_masks],
                                       outputs=[discriminator_output_from_real_samples,
                                                discriminator_output_from_fake_samples,
                                                averaged_samples_outputs])
    # We use the Adam paramaters from Gulrajani et al. We use the Wasserstein loss for both
    # the real and generated samples, and the gradient penalty loss for the averaged samples
    global_discriminator_model.compile(optimizer=self.discriminator_optimizer,
                                       loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss])
    
    return global_discriminator_model
  
  def define_local_discriminator(self, generator_raw, local_discriminator_raw):
    generator_inputs = Input(shape=(self.img_height, self.img_width, self.num_channels))
    generator_masks = Input(shape=(self.img_height, self.img_width, self.num_channels))
    
    real_samples = Input(shape=(self.img_height, self.img_width, self.num_channels))
    fake_samples = generator_raw.model([generator_inputs, generator_masks])
    # fake_samples = generator_inputs * (1 - generator_masks) + fake_samples * generator_masks
    # fake_samples = Lambda(make_comp_sample)([generator_inputs, fake_samples, generator_masks])
    
    discriminator_output_from_fake_samples = local_discriminator_raw.model(
      [fake_samples, generator_masks])
    discriminator_output_from_real_samples = local_discriminator_raw.model(
      [real_samples, generator_masks])
    
    averaged_samples = custom_layers.RandomWeightedAverage()([real_samples, fake_samples])
    averaged_samples_output = local_discriminator_raw.model([averaged_samples, generator_masks])
    
    partial_gp_loss = partial(gradient_penalty_loss,
                              averaged_samples=averaged_samples,
                              gradient_penalty_weight=self.gradient_penalty_loss_weight)
    partial_gp_loss.__name__ = 'gradient_penalty'
    
    local_discriminator_model = Model(inputs=[real_samples, generator_inputs, generator_masks],
                                      outputs=[discriminator_output_from_real_samples,
                                               discriminator_output_from_fake_samples,
                                               averaged_samples_output])
    
    local_discriminator_model.compile(optimizer=self.discriminator_optimizer,
                                      loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss])
    return local_discriminator_model
  
  @property
  def global_discriminator(self):
    return self.global_discriminator_model
  
  @property
  def local_discriminator(self):
    return self.local_discriminator_model
  
  @property
  def generator(self):
    return self.generator_model
  
  @property
  def generator_for_prediction(self):
    return self.generator_raw.model


def make_comp_sample(inputs):
  generator_inputs, fake_samples, generator_masks = inputs
  return generator_inputs * (1 - generator_masks) + fake_samples * generator_masks

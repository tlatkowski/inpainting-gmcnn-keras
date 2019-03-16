from functools import partial

from keras.models import Model, Input
from keras.optimizers import Adam

from layers import custom_layers
from layers.losses import reconstruction_loss, wasserstein_loss, gradient_penalty_loss, \
  confidence_reconstruction_loss, id_mrf_loss
from models.discriminator import GlobalDiscriminator, LocalDiscriminator
from models.generator import Generator
from models.wgan import WassersteinGAN


class GMCNNGan(WassersteinGAN):
  
  def __init__(self, batch_size, img_height, img_width, num_channels=3, warm_up_generator=False,
               learning_rate=0.0001, n_critic=5, num_gaussian_steps=3, gradient_penalty_weight=10):
    super(GMCNNGan, self).__init__(img_height, img_width, num_channels, batch_size, n_critic)
    
    self.img_height = img_height
    self.img_width = img_width
    self.num_channels = num_channels
    self.warm_up_generator = warm_up_generator
    self.num_gaussian_steps = num_gaussian_steps
    self.gradient_penalty_weight = gradient_penalty_weight
    
    self.generator_optimizer = Adam(lr=learning_rate, beta_1=0.5, beta_2=0.9)
    self.discriminator_optimizer = Adam(lr=learning_rate, beta_1=0.5, beta_2=0.9)
    
    self.local_discriminator_raw = LocalDiscriminator(self.img_height, self.img_width,
                                                      self.num_channels)
    self.global_discriminator_raw = GlobalDiscriminator(self.img_height, self.img_width,
                                                        self.num_channels)
    self.generator_raw = Generator(self.img_height, self.img_width, self.num_channels)
    
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
    
    partial_cr_loss = partial(confidence_reconstruction_loss,
                              mask=generator_inputs_mask,
                              num_steps=self.num_gaussian_steps)
    
    partial_cr_loss.__name__ = 'id_mrf_loss'
    
    # from models import vgg
    # vgg_model = vgg.build_vgg16(img_rows=128,
    #                             img_cols=128)
    
    # partial_id_mrf_loss = partial(id_mrf_loss, vgg_model=vgg_model)
    #
    # partial_id_mrf_loss.__name__ = 'id_mrf_loss'
    
    if self.warm_up_generator:
      # set Wasserstein loss to 0 - total generator loss will be based only on reconstruction loss
      generator_model.compile(optimizer=self.generator_optimizer,
                              loss=[reconstruction_loss, id_mrf_loss, wasserstein_loss,
                                    wasserstein_loss],
                              loss_weights=[1., 0., 0., 0.])
    else:
      generator_model.compile(optimizer=self.generator_optimizer,
                              loss=[partial_cr_loss, id_mrf_loss, wasserstein_loss,
                                    wasserstein_loss], loss_weights=[1., 0.05, 0.001, 0.001])
    
    return generator_model
  
  def define_global_discriminator(self, generator_raw, global_discriminator_raw):
    real_samples = Input(shape=(self.img_height, self.img_width, self.num_channels))
    
    generator_inputs = Input(shape=(self.img_height, self.img_width, self.num_channels))
    generator_masks = Input(shape=(self.img_height, self.img_width, self.num_channels))
    
    fake_samples = generator_raw.model([generator_inputs, generator_masks])
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
                              gradient_penalty_weight=self.gradient_penalty_weight)
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
    real_samples = Input(shape=(self.img_height, self.img_width, self.num_channels))
    
    generator_inputs = Input(shape=(self.img_height, self.img_width, self.num_channels))
    generator_masks = Input(shape=(self.img_height, self.img_width, self.num_channels))
    
    fake_samples = generator_raw.model([generator_inputs, generator_masks])
    discriminator_output_from_fake_samples = local_discriminator_raw.model(
      [fake_samples, generator_masks])
    discriminator_output_from_real_samples = local_discriminator_raw.model(
      [real_samples, generator_masks])
    
    averaged_samples = custom_layers.RandomWeightedAverage()([real_samples, fake_samples])
    averaged_samples_output = local_discriminator_raw.model([averaged_samples, generator_masks])
    
    partial_gp_loss = partial(gradient_penalty_loss,
                              averaged_samples=averaged_samples,
                              gradient_penalty_weight=self.gradient_penalty_weight)
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

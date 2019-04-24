# !/usr/bin/env python3

import numpy as np
import tqdm
from keras import callbacks

from utils import constants
from utils import metrics
from utils import training_utils


class Trainer:
  
  def __init__(self, gan_model, img_dataset, mask_dataset, batch_size, img_height, img_width,
               num_epochs, save_model_steps_period, output_paths: constants.OutputPaths):
    self.gan_model = gan_model
    self.img_dataset = img_dataset
    self.mask_dataset = mask_dataset
    self.batch_size = batch_size
    self.img_height = img_height
    self.img_width = img_width
    self.num_epochs = num_epochs
    self.save_model_steps_period = save_model_steps_period
    
    self.num_samples = self.img_dataset.train_set.samples
    self.wgan_num_steps = int(self.num_samples / self.gan_model.wgan_batch_size)
    
    self.epochs_iter = tqdm.tqdm(range(self.num_epochs), total=self.num_epochs, desc='Epochs')
    if self.gan_model.warm_up_generator:
      self.log_path = output_paths.warm_up_logs_path
      self.predicted_img_path = output_paths.predicted_pics_warm_up_path
    else:
      self.log_path = output_paths.wgan_logs_path
      self.predicted_img_path = output_paths.predicted_pics_wgan_path
  
  def train(self):
    y_real = np.ones([self.gan_model.wgan_batch_size, 1])
    y_fake = -y_real
    y_dummy = np.zeros((self.gan_model.wgan_batch_size, 1))
    
    tensorboard = callbacks.TensorBoard(self.log_path, 0)
    tensorboard.set_model(self.gan_model.generator)
    
    global_step = 0
    for epoch in self.epochs_iter:
      step = 0
      for real_img in self.img_dataset.train_set:
        mask = next(self.mask_dataset.train_set)
        if step == self.wgan_num_steps:
          break
        step += 1
        
        if self.gan_model.warm_up_generator:  # TODO do generator steps not WGAN
          generator_loss = self.gan_model.train_generator(inputs=[real_img, mask],
                                                          outputs=[real_img, real_img, y_real,
                                                                   y_real])
          logs = training_utils.create_warm_up_log(generator_loss)
          self.update_progress_bar(generator_loss[0], 0.0, 0.0, epoch, step, self.wgan_num_steps)
        else:
          global_discriminator_loss, local_discriminator_loss, generator_loss = self.gan_model.train_wgan(
            d_inputs=[real_img, real_img, mask],
            d_outputs=[y_real, y_fake, y_dummy],
            g_inputs=[real_img, mask],
            g_outputs=[real_img, real_img, y_real, y_real])
          
          logs = training_utils.create_standard_log(generator_loss, global_discriminator_loss,
                                                    local_discriminator_loss)
          self.update_progress_bar(generator_loss.total_loss, global_discriminator_loss.total_loss,
                                   local_discriminator_loss.total_loss, epoch, step,
                                   self.wgan_num_steps)
        
        if global_step % self.save_model_steps_period == 0:
          input_img = np.expand_dims(real_img[0], 0)
          input_mask = np.expand_dims(mask[0], 0)
          predicted_img = self.gan_model.predict(inputs=[input_img, input_mask])
          training_utils.save_predicted_img(self.predicted_img_path, input_img, predicted_img,
                                            input_mask, global_step)
          m = metrics.psnr(input_img, predicted_img)
          l = {'metrics/psnr': m}
          tensorboard.on_epoch_end(global_step, l)
          self.gan_model.save()
        
        tensorboard.on_epoch_end(global_step, logs)
        global_step += 1
  
  def update_progress_bar(self, generator_loss, global_discriminator_loss, local_discriminator_loss,
                          epoch, current_batch, total_batch):
    self.epochs_iter.set_postfix(
      generator_loss='{:.2f}'.format(float(generator_loss)),
      global_discriminator_loss='{:.2f}'.format(float(global_discriminator_loss)),
      local_discriminator_loss='{:.2f}'.format(float(local_discriminator_loss)),
      epoch=epoch,
      step='{:d}|{:d}'.format(current_batch, total_batch))

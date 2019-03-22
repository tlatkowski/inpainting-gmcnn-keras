import os
from collections import namedtuple
from copy import deepcopy

import cv2
import numpy as np
import tensorflow as tf


def create_standard_log(generator_loss: namedtuple, global_discriminator_loss: namedtuple,
                        local_discriminator_loss: namedtuple):
  generator_log = parse_namedtuple(generator_loss, 'generator')
  global_discriminator_log = parse_namedtuple(global_discriminator_loss, 'global_discriminator')
  local_discriminator_log = parse_namedtuple(local_discriminator_loss, 'local_discriminator')
  
  logs = {**generator_log, **global_discriminator_log, **local_discriminator_log}
  return logs


def create_warm_up_log(generator_loss):
  logs = {'generator_warm_up': generator_loss[0]}
  return logs


def parse_namedtuple(losses: namedtuple, prefix: str):
  log = {'{}/{}'.format(prefix, k): v for k, v in losses._asdict().items()}
  return log


def save_predicted_img(predicted_img_path, input_img, sample_pred, mask, epoch):
  os.makedirs(predicted_img_path, exist_ok=True)
  input_img = np.expand_dims(input_img[0], 0)
  input_mask = np.expand_dims(mask[0], 0)
  
  input_mask = 1 - input_mask
  masked = deepcopy(input_img)
  masked = masked * 127.5 + 127.5
  masked[input_mask == 0] = 255
  img = np.concatenate((masked[0][..., [2, 1, 0]],
                        sample_pred[0][..., [2, 1, 0]] * 127.5 + 127.5,
                        input_img[0][..., [2, 1, 0]] * 127.5 + 127.5),
                       axis=1)
  img_filepath = os.path.join(predicted_img_path, 'step_{0:03d}.png'.format(epoch))
  cv2.imwrite(img_filepath, img)


def get_logger():
  log = tf.logging
  tf.logging.set_verbosity(tf.logging.INFO)
  return log


def set_visible_gpu(gpu_number: str):
  get_logger().info('Setting visible GPU to {}'.format(gpu_number))
  os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

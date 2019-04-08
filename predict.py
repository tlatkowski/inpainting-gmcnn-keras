# !/usr/bin/env python3

from argparse import ArgumentParser
from copy import deepcopy

import cv2
import numpy as np

from config import main_config
from models import gmcnn_gan
from utils import training_utils

log = training_utils.get_logger()

MAIN_CONFIG_FILE = './config/main_config.ini'


def preprocess_image(image, model_input_height, model_input_width):
  image = image[..., [2, 1, 0]]
  image = (image - 127.5) / 127.5
  image = cv2.resize(image, (model_input_height, model_input_width))
  image = np.expand_dims(image, 0)
  return image


def preprocess_mask(mask, model_input_height, model_input_width):
  mask[mask == 255] = 1
  mask = cv2.resize(mask, (model_input_height, model_input_width))
  mask = np.expand_dims(mask, 0)
  return mask


def postprocess_image(image):
  image = (image + 1) * 127.5
  return image


def main():
  parser = ArgumentParser()
  
  parser.add_argument('--image',
                      required=True,
                      help='The path to the image')
  
  parser.add_argument('--mask',
                      required=True,
                      help='The path to the mask')
  
  parser.add_argument('--save_to',
                      default='predicted.jpg',
                      help='The save path of predicted image')
  
  args = parser.parse_args()
  
  config = main_config.MainConfig(MAIN_CONFIG_FILE)
  
  gmcnn_model = gmcnn_gan.GMCNNGan(batch_size=config.training.batch_size,
                                   img_height=config.training.img_height,
                                   img_width=config.training.img_width,
                                   num_channels=config.training.num_channels,
                                   warm_up_generator=False,
                                   config=config)
  log.info('Loading GMCNN model...')
  gmcnn_model.load()
  log.info('GMCNN model successfully loaded.')
  
  image = cv2.imread(args.image)
  mask = cv2.imread(args.mask)
  
  image = preprocess_image(image, config.training.img_height, config.training.img_width)
  mask = preprocess_mask(mask, config.training.img_height, config.training.img_width)
  
  log.info('Making prediction...')
  predicted = gmcnn_model.predict([image, mask])
  
  predicted = postprocess_image(predicted)
  
  masked = deepcopy(image)
  masked = postprocess_image(masked)
  masked[mask == 1] = 255
  result_image = np.concatenate((masked[0][..., [2, 1, 0]],
                                 predicted[0][..., [2, 1, 0]],
                                 image[0][..., [2, 1, 0]] * 127.5 + 127.5),
                                axis=1)
  
  cv2.imwrite(args.save_to, result_image)
  log.info('Saved results to: %s', args.save_to)


if __name__ == '__main__':
  main()

# !/usr/bin/env python3

import os
from argparse import ArgumentParser

from config import main_config
from data_generators import datasets
from models import gmcnn_gan
from utils import trainer
from utils import training_utils
from utils import constants

log = training_utils.get_logger()

MAIN_CONFIG_FILE = './config/main_config.ini'


def main():
  parser = ArgumentParser()
  
  parser.add_argument('--train_path',
                      required=True,
                      help='The path to training images')
  
  parser.add_argument('--mask_path',
                      required=True,
                      help='The path to mask images')
  
  parser.add_argument('--experiment_name',
                      required=True,
                      help='The name of experiment')
  
  parser.add_argument('-warm_up_generator',
                      action='store_true',
                      help='Training generator model only with reconstruction loss')
  
  parser.add_argument('-from_weights',
                      action='store_true',
                      help='Use this command to continue training from weights')
  
  parser.add_argument('--gpu',
                      default='0',
                      help='index of GPU to be used (default: %(default))')
  
  args = parser.parse_args()
  
  output_paths = constants.OutputPaths(experiment_name=args.experiment_name)
  training_utils.set_visible_gpu(args.gpu)
  if args.warm_up_generator:
    log.info('Performing generator training only with the reconstruction loss.')
  
  config = main_config.MainConfig(MAIN_CONFIG_FILE)
  wgan_batch_size = config.training.wgan_training_ratio * config.training.batch_size
  
  train_path = os.path.expanduser(args.train_path)
  mask_path = os.path.expanduser(args.mask_path)
  
  gmcnn_gan_model = gmcnn_gan.GMCNNGan(batch_size=config.training.batch_size,
                                       img_height=config.training.img_height,
                                       img_width=config.training.img_width,
                                       num_channels=config.training.num_channels,
                                       warm_up_generator=args.warm_up_generator,
                                       config=config,
                                       output_paths=output_paths)
  
  if args.from_weights:
    log.info('Continue training from checkpoint...')
    gmcnn_gan_model.load()
  
  img_dataset = datasets.Dataset(train_path=train_path,
                                 test_path=train_path,
                                 batch_size=wgan_batch_size,
                                 img_height=config.training.img_height,
                                 img_width=config.training.img_width)
  
  if img_dataset.train_set.samples < wgan_batch_size:
    log.error('Number of training images [%s] is lower than WGAN batch size [%s]',
              img_dataset.train_set.samples, wgan_batch_size)
    exit(0)
  
  mask_dataset = datasets.MaskDataset(train_path=mask_path,
                                      batch_size=wgan_batch_size,
                                      img_height=config.training.img_height,
                                      img_width=config.training.img_width)
  
  if mask_dataset.train_set.samples < wgan_batch_size:
    log.error('Number of training mask images [%s] is lower than WGAN batch size [%s]',
              mask_dataset.train_set.samples, wgan_batch_size)
    exit(0)
  
  gmcnn_gan_trainer = trainer.Trainer(gan_model=gmcnn_gan_model,
                                      img_dataset=img_dataset,
                                      mask_dataset=mask_dataset,
                                      batch_size=config.training.batch_size,
                                      img_height=config.training.img_height,
                                      img_width=config.training.img_width,
                                      num_epochs=config.training.num_epochs,
                                      save_model_steps_period=config.training.save_model_steps_period,
                                      output_paths=output_paths)
  
  gmcnn_gan_trainer.train()


if __name__ == '__main__':
  main()

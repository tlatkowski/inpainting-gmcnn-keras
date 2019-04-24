#!/usr/bin/env python3

from keras.layers import Conv2D, Dense, Flatten, LeakyReLU
from keras.layers import Multiply
from keras.models import Input, Model

from models.base import BaseModel


class LocalDiscriminator(BaseModel):
  
  def __init__(self, img_height, img_width, num_channels, output_paths):
    super(LocalDiscriminator, self).__init__(img_height, img_width, num_channels, output_paths,
                                             model_name='local_discriminator')
  
  def model(self):
    inputs_img = Input(shape=(self.img_height, self.img_width, self.num_channels))
    inputs_mask = Input(shape=(self.img_height, self.img_width, self.num_channels))
    
    inputs = Multiply()([inputs_img, inputs_mask])
    
    # Local discriminator
    l_dis = Conv2D(filters=64, kernel_size=5, strides=(2, 2), padding='same')(inputs)
    l_dis = LeakyReLU()(l_dis)
    l_dis = Conv2D(filters=128, kernel_size=5, strides=(2, 2), padding='same')(l_dis)
    l_dis = LeakyReLU()(l_dis)
    l_dis = Conv2D(filters=256, kernel_size=5, strides=(2, 2), padding='same')(l_dis)
    l_dis = LeakyReLU()(l_dis)
    l_dis = Conv2D(filters=512, kernel_size=5, strides=(2, 2), padding='same')(l_dis)
    l_dis = LeakyReLU()(l_dis)
    l_dis = Conv2D(filters=256, kernel_size=5, strides=(2, 2), padding='same')(l_dis)
    l_dis = LeakyReLU()(l_dis)
    l_dis = Conv2D(filters=128, kernel_size=5, strides=(2, 2), padding='same')(l_dis)
    l_dis = LeakyReLU()(l_dis)
    l_dis = Flatten()(l_dis)
    l_dis = Dense(units=1)(l_dis)
    
    model = Model(name=self.model_name, inputs=[inputs_img, inputs_mask], outputs=l_dis)
    return model


class GlobalDiscriminator(BaseModel):
  
  def __init__(self, img_height, img_width, num_channels, output_paths):
    super(GlobalDiscriminator, self).__init__(img_height, img_width, num_channels, output_paths,
                                              model_name='global_discriminator')
  
  def model(self):
    inputs = Input(shape=(self.img_height, self.img_width, self.num_channels))
    
    # Local discriminator
    g_dis = Conv2D(filters=64, kernel_size=5, strides=(2, 2), padding='same')(inputs)
    g_dis = LeakyReLU()(g_dis)
    g_dis = Conv2D(filters=128, kernel_size=5, strides=(2, 2), padding='same')(g_dis)
    g_dis = LeakyReLU()(g_dis)
    g_dis = Conv2D(filters=256, kernel_size=5, strides=(2, 2), padding='same')(g_dis)
    g_dis = LeakyReLU()(g_dis)
    g_dis = Conv2D(filters=512, kernel_size=5, strides=(2, 2), padding='same')(g_dis)
    g_dis = LeakyReLU()(g_dis)
    g_dis = Conv2D(filters=256, kernel_size=5, strides=(2, 2), padding='same')(g_dis)
    g_dis = LeakyReLU()(g_dis)
    g_dis = Conv2D(filters=128, kernel_size=5, strides=(2, 2), padding='same')(g_dis)
    g_dis = LeakyReLU()(g_dis)
    g_dis = Flatten()(g_dis)
    g_dis = Dense(units=1)(g_dis)
    
    model = Model(name=self.model_name, inputs=inputs, outputs=g_dis)
    return model

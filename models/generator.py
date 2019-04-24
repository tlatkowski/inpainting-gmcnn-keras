#!/usr/bin/env python3

from keras.layers import Conv2D, UpSampling2D, Concatenate, Multiply, ELU
from keras.models import Input, Model

from layers.custom_layers import Clip, BinaryNegation
from models.base import BaseModel


class Generator(BaseModel):
  
  def __init__(self, img_height, img_width, num_channels, add_mask_as_input, output_paths):
    self.add_mask_as_input = add_mask_as_input
    super(Generator, self).__init__(img_height, img_width, num_channels, output_paths,
                                    model_name='generator')
  
  def model(self):
    inputs_img = Input(shape=(self.img_height, self.img_width, self.num_channels))
    masks = Input(shape=(self.img_height, self.img_width, self.num_channels))
    
    neg_masks = BinaryNegation()(masks)
    inputs = Multiply()([inputs_img, neg_masks])
    
    if self.add_mask_as_input:
      inputs = Concatenate(axis=3)([inputs, masks])
    
    # Encoder-branch-1
    eb1 = Conv2D(filters=32, kernel_size=7, strides=(1, 1), padding='same')(inputs)
    eb1 = ELU()(eb1)
    eb1 = Conv2D(filters=64, kernel_size=7, strides=(2, 2), padding='same')(eb1)
    eb1 = ELU()(eb1)
    eb1 = Conv2D(filters=64, kernel_size=7, strides=(1, 1), padding='same')(eb1)
    eb1 = ELU()(eb1)
    eb1 = Conv2D(filters=128, kernel_size=7, strides=(2, 2), padding='same')(eb1)
    eb1 = ELU()(eb1)
    eb1 = Conv2D(filters=128, kernel_size=7, strides=(1, 1), padding='same')(eb1)
    eb1 = ELU()(eb1)
    eb1 = Conv2D(filters=128, kernel_size=7, strides=(1, 1), padding='same')(eb1)
    
    eb1 = Conv2D(filters=128, kernel_size=7, strides=(1, 1), padding='same', dilation_rate=(2, 2))(
      eb1)
    eb1 = ELU()(eb1)
    eb1 = Conv2D(filters=128, kernel_size=7, strides=(1, 1), padding='same', dilation_rate=(4, 4))(
      eb1)
    eb1 = ELU()(eb1)
    eb1 = Conv2D(filters=128, kernel_size=7, strides=(1, 1), padding='same', dilation_rate=(8, 8))(
      eb1)
    eb1 = ELU()(eb1)
    eb1 = Conv2D(filters=128, kernel_size=7, strides=(1, 1), padding='same',
                 dilation_rate=(16, 16))(eb1)
    eb1 = ELU()(eb1)
    eb1 = Conv2D(filters=128, kernel_size=7, strides=(1, 1), padding='same')(eb1)
    eb1 = ELU()(eb1)
    eb1 = Conv2D(filters=128, kernel_size=7, strides=(1, 1), padding='same')(eb1)
    eb1 = ELU()(eb1)
    
    eb1 = UpSampling2D(size=(4, 4))(eb1)
    
    # Encoder-branch-2
    eb2 = Conv2D(filters=32, kernel_size=5, strides=(1, 1), padding='same')(inputs)
    eb2 = ELU()(eb2)
    eb2 = Conv2D(filters=64, kernel_size=5, strides=(2, 2), padding='same')(eb2)
    eb2 = ELU()(eb2)
    eb2 = Conv2D(filters=64, kernel_size=5, strides=(1, 1), padding='same')(eb2)
    eb2 = ELU()(eb2)
    eb2 = Conv2D(filters=128, kernel_size=5, strides=(2, 2), padding='same')(eb2)
    eb2 = ELU()(eb2)
    eb2 = Conv2D(filters=128, kernel_size=5, strides=(1, 1), padding='same')(eb2)
    eb2 = ELU()(eb2)
    eb2 = Conv2D(filters=128, kernel_size=5, strides=(1, 1), padding='same')(eb2)
    eb2 = ELU()(eb2)
    
    eb2 = Conv2D(filters=128, kernel_size=5, strides=(1, 1), padding='same', dilation_rate=(2, 2))(
      eb2)
    eb2 = ELU()(eb2)
    eb2 = Conv2D(filters=128, kernel_size=5, strides=(1, 1), padding='same', dilation_rate=(4, 4))(
      eb2)
    eb2 = ELU()(eb2)
    eb2 = Conv2D(filters=128, kernel_size=5, strides=(1, 1), padding='same', dilation_rate=(8, 8))(
      eb2)
    eb2 = ELU()(eb2)
    eb2 = Conv2D(filters=128, kernel_size=5, strides=(1, 1), padding='same',
                 dilation_rate=(16, 16))(eb2)
    eb2 = ELU()(eb2)
    eb2 = Conv2D(filters=128, kernel_size=5, strides=(1, 1), padding='same')(eb2)
    eb2 = ELU()(eb2)
    eb2 = Conv2D(filters=128, kernel_size=5, strides=(1, 1), padding='same')(eb2)
    eb2 = ELU()(eb2)
    
    eb2 = UpSampling2D(size=(2, 2))(eb2)
    
    eb2 = Conv2D(filters=64, kernel_size=5, strides=(1, 1), padding='same')(eb2)
    eb2 = ELU()(eb2)
    eb2 = Conv2D(filters=64, kernel_size=5, strides=(1, 1), padding='same')(eb2)
    eb2 = ELU()(eb2)
    
    eb2 = UpSampling2D(size=(2, 2))(eb2)
    
    # Encoder-branch-3
    eb3 = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding='same')(inputs)
    eb3 = ELU()(eb3)
    eb3 = Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same')(eb3)
    eb3 = ELU()(eb3)
    eb3 = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')(eb3)
    eb3 = ELU()(eb3)
    eb3 = Conv2D(filters=128, kernel_size=3, strides=(2, 2), padding='same')(eb3)
    eb3 = ELU()(eb3)
    eb3 = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='same')(eb3)
    eb3 = ELU()(eb3)
    eb3 = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='same')(eb3)
    eb3 = ELU()(eb3)
    
    eb3 = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='same', dilation_rate=(2, 2))(
      eb3)
    eb3 = ELU()(eb3)
    eb3 = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='same', dilation_rate=(4, 4))(
      eb3)
    eb3 = ELU()(eb3)
    eb3 = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='same', dilation_rate=(8, 8))(
      eb3)
    eb3 = ELU()(eb3)
    eb3 = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='same',
                 dilation_rate=(16, 16))(eb3)
    eb3 = ELU()(eb3)
    
    eb3 = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='same')(eb3)
    eb3 = ELU()(eb3)
    eb3 = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='same')(eb3)
    eb3 = ELU()(eb3)
    
    eb3 = UpSampling2D(size=(2, 2))(eb3)
    
    eb3 = Conv2D(filters=64, kernel_size=5, strides=(1, 1), padding='same')(eb3)
    eb3 = ELU()(eb3)
    eb3 = Conv2D(filters=64, kernel_size=5, strides=(1, 1), padding='same')(eb3)
    eb3 = ELU()(eb3)
    
    eb3 = UpSampling2D(size=(2, 2))(eb3)
    
    eb3 = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')(eb3)
    eb3 = ELU()(eb3)
    eb3 = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')(eb3)
    eb3 = ELU()(eb3)
    
    decoder = Concatenate(axis=3)([eb1, eb2, eb3])
    
    decoder = Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='same')(decoder)
    decoder = ELU()(decoder)
    decoder = Conv2D(filters=3, kernel_size=3, strides=(1, 1), padding='same')(decoder)
    
    # linearly norm to (-1, 1)
    decoder = Clip()(decoder)
    
    model = Model(name=self.model_name, inputs=[inputs_img, masks], outputs=[decoder])
    return model

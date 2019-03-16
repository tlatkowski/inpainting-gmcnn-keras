from keras import backend as K
from keras.layers import Layer, subtract
from keras.layers.merge import _Merge


class RandomWeightedAverage(_Merge):
  """Takes a randomly-weighted average of two tensors. In geometric terms, this
  outputs a random point on the line between each pair of input points.
  Inheriting from _Merge is a little messy but it was the quickest solution I could
  think of. Improvements appreciated."""
  
  def _merge_function(self, inputs):
    weights = K.random_uniform((1, 1, 1, 1))
    return (weights * inputs[0]) + ((1 - weights) * inputs[1])


class Clip(Layer):
  def __init__(self):
    super(Clip, self).__init__()
  
  def call(self, inputs):
    clipped_inputs = K.clip(inputs, -1, 1)
    return clipped_inputs


class BinaryNegation(Layer):
  def __init__(self):
    super(BinaryNegation, self).__init__()
  
  def call(self, inputs):
    ones_like_tensor = K.ones_like(inputs)
    neg = subtract([ones_like_tensor, inputs])
    return neg

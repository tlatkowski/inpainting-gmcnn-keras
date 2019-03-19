import tensorflow as tf
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Lambda
from keras.models import Model

ORIGINAL_VGG_16_SHAPE = (224, 224, 3)


def build_vgg16(y_pred, use_original_vgg_shape, vgg_layers):
  """
  Load pre-trained VGG16 from keras applications
  """
  if use_original_vgg_shape:
    return build_vgg_original_shape(y_pred, vgg_layers)
  else:
    return build_vgg_img_shape(y_pred, vgg_layers)


def build_vgg_original_shape(y_pred, vgg_layers):
  input_shape = y_pred.shape.as_list()[1:4]
  img = Input(shape=input_shape)
  
  img_reshaped = Lambda(lambda x: tf.image.resize_nearest_neighbor(x, size=ORIGINAL_VGG_16_SHAPE))(
    img)
  
  img_norm = _norm_inputs(img_reshaped)
  vgg = VGG16(weights="imagenet", include_top=False)
  
  # Output the first three pooling layers
  vgg.outputs = [vgg.layers[i].output for i in vgg_layers]
  
  # Create model and compile
  model = Model(inputs=img, outputs=vgg(img_norm))
  model.trainable = False
  model.compile(loss='mse', optimizer='adam')
  
  return model


def build_vgg_img_shape(y_pred, vgg_layers):
  input_shape = y_pred.shape.as_list()[1:4]
  img = Input(shape=input_shape)
  
  img_norm = _norm_inputs(img)
  vgg = VGG16(weights="imagenet", include_top=False)
  
  # Output the first three pooling layers
  vgg.outputs = [vgg.layers[i].output for i in vgg_layers]
  
  # Create model and compile
  model = Model(inputs=img, outputs=vgg(img_norm))
  model.trainable = False
  model.compile(loss='mse', optimizer='adam')
  
  return model


def _norm_inputs(input_img):
  ones = tf.constant(1, dtype=tf.float32)
  c = tf.constant(127.5, dtype=tf.float32)
  
  img_norm = Lambda(lambda x: x + ones)(input_img)
  img_norm = Lambda(lambda x: x * c)(img_norm)
  img_norm = Lambda(preprocess_input)(img_norm)
  return img_norm

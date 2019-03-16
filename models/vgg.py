from keras.applications import VGG16
from keras.layers import Input
from keras.models import Model

VGG_LAYERS = [3, 6, 10]


def build_vgg16(img_rows, img_cols, vgg_layers=VGG_LAYERS):
  """
  Load pre-trained VGG16 from keras applications
  """
  # Input image to extract features from
  img = Input(shape=(img_rows, img_cols, 3))
  
  # Get the vgg network from Keras applications
  vgg = VGG16(weights="imagenet", include_top=False)
  
  # Output the first three pooling layers
  vgg.outputs = [vgg.layers[i].output for i in vgg_layers]
  
  # Create model and compile
  model = Model(inputs=img, outputs=vgg(img))
  model.trainable = False
  model.compile(loss='mse', optimizer='adam')
  
  return model

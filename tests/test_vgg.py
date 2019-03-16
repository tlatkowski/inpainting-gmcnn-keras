import tensorflow as tf

from models import vgg


class TestVgg(tf.test.TestCase):
  
  def test_vgg(self):
    img_rows = 224
    img_cols = 224
    vgg_model = vgg.build_vgg16(img_rows, img_cols)
    
    expected_output_vgg_shape = [(None, 112, 112, 64), (None, 56, 56, 128), (None, 28, 28, 256)]
    actual_output_vgg_shape = vgg_model.output_shape
    self.assertEquals(expected_output_vgg_shape, actual_output_vgg_shape)

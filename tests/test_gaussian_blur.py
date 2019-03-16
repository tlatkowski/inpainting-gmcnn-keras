import cv2
import numpy as np
import tensorflow as tf

from utils import gaussian_utils


class TestGaussianBlur(tf.test.TestCase):
  
  def test_gaussian_blur(self):
    num_blur_steps = 1
    mask = np.zeros(shape=(1, 3, 3, 3))
    mask[:, 1, 1, :] = 1
    with self.test_session() as s:
      blurred_mask = s.run(gaussian_utils.blur_mask(mask, num_blur_steps))
      self.assertAllLess(blurred_mask[:, 1, 1, :], 1)
  
  def test_large_nvidia_mask_gaussian_blur(self):
    num_conv_steps = [1, 2, 3, 4, 5, 10]
    mask = cv2.imread('./pics/large_mask.png')
    mask[mask == 255] = 1
    mask = np.expand_dims(mask, 0)
    with self.test_session() as s:
      for steps in num_conv_steps:
        blurred_mask = s.run(gaussian_utils.blur_mask(mask, steps))
        blurred_mask = blurred_mask[0]
        blurred_mask = cv2.resize(blurred_mask, (128, 128), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('./test_results/large_blurred_mask_{}_step.png'.format(steps),
                    blurred_mask * 255)
  
  def test_small_nvidia_mask_gaussian_blur(self):
    num_conv_steps = [1, 2, 3, 4, 5, 10]
    mask = cv2.imread('./pics/small_mask.png')
    mask[mask == 255] = 1
    mask = np.expand_dims(mask, 0)
    with self.test_session() as s:
      for steps in num_conv_steps:
        blurred_mask = s.run(gaussian_utils.blur_mask(mask, steps))
        blurred_mask = blurred_mask[0]
        blurred_mask = cv2.resize(blurred_mask, (128, 128), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('./test_results/small_blurred_mask_{}_step.png'.format(steps),
                    blurred_mask * 255)
  
  def test_rectangle_mask_gaussian_blur_1_conv_step(self):
    num_conv_steps = [1, 2, 3, 4, 5, 10]
    mask = np.zeros(shape=(512, 512, 3))
    mask[128:384, 128:384, :] = 1
    mask = np.expand_dims(mask, 0)
    with self.test_session() as s:
      for steps in num_conv_steps:
        blurred_mask = s.run(gaussian_utils.blur_mask(mask, steps))
        blurred_mask = blurred_mask[0]
        blurred_mask = cv2.resize(blurred_mask, (128, 128), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('./test_results/rectangle_blurred_mask_{}_step.png'.format(steps),
                    blurred_mask * 255)

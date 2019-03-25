import numpy as np
import tensorflow as tf

from utils import norm_utils


class TestNormalizationUtils(tf.test.TestCase):
  
  def test_center_by_predicted(self):
    with self.test_session() as session:
      x1 = np.reshape(np.array([[1, 2],
                                [3, 4]], dtype=np.float32), newshape=[1, 2, 2, 1])
      x1 = tf.convert_to_tensor(x1)
      
      x2 = np.reshape(np.array([[1, 2],
                                [3, 4]], dtype=np.float32), newshape=[1, 2, 2, 1])
      x2 = tf.convert_to_tensor(x2)
      
      norm_values = norm_utils.center_by_predicted(x1, x2)
      x1_norm, x2_norm = session.run(norm_values)
      x1_norm = np.reshape(x1_norm, newshape=(2, 2))
      x2_norm = np.reshape(x2_norm, newshape=(2, 2))
      
      expected_norm_values = np.array([[-1.5, -0.5],
                                       [0.5, 1.5]], dtype=np.float32)
      self.assertAllEqual(x1_norm, expected_norm_values)
      self.assertAllEqual(x2_norm, expected_norm_values)
  
  def test_l2_normalize_channel_wise(self):
    with self.test_session() as session:
      x1 = np.reshape(np.array([[1, 2],
                                [3, 4]], dtype=np.float32), newshape=[1, 2, 2, 1])
      
      actual_norm_values = norm_utils.l2_normalize_channel_wise(x1)
      actual_norm_values = session.run(actual_norm_values)
      
      expected_norm_values = np.reshape(np.array([[1, 1],
                                                  [1, 1]], dtype=np.float32), newshape=[1, 2, 2, 1])
      self.assertAllEqual(actual_norm_values, expected_norm_values)
  
  def test_l2_normalize_channel_wise_for_centered_input(self):
    with self.test_session() as session:
      x1 = np.reshape(np.array([[-1.5, -0.5],
                                [0.5, 1.5]], dtype=np.float32), newshape=[1, 2, 2, 1])
      
      actual_norm_values = norm_utils.l2_normalize_channel_wise(x1)
      actual_norm_values = session.run(actual_norm_values)
      
      expected_norm_values = np.reshape(np.array([[-1, -1],
                                                  [1, 1]], dtype=np.float32), newshape=[1, 2, 2, 1])
      self.assertAllEqual(actual_norm_values, expected_norm_values)

import tensorflow as tf


def gaussian_kernel(size: int, mean: float, std: float, ):
  """Makes 2D gaussian Kernel for convolution."""
  d = tf.distributions.Normal(mean, std)
  vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))
  gauss_kernel = tf.einsum('i,j->ij', vals, vals)
  return gauss_kernel / tf.reduce_sum(gauss_kernel)


def blur_mask(mask, num_conv_steps, gaussian_kernel_size, gaussian_kernel_std):
  mask = tf.convert_to_tensor(mask, dtype=tf.float32)
  mask = tf.expand_dims(mask[:, :, :, 0], 3)
  # gaussian_filter = gaussian_kernel(size=32, mean=0.0, std=40.0)
  gaussian_filter = gaussian_kernel(size=gaussian_kernel_size, mean=0.0, std=gaussian_kernel_std)
  gaussian_filter = gaussian_filter[:, :, tf.newaxis, tf.newaxis]
  neg_mask = 1 - mask
  m_w = 0
  for _ in range(num_conv_steps):
    m_i = neg_mask + m_w
    m_w = tf.nn.conv2d(m_i, filter=gaussian_filter, strides=[1, 1, 1, 1], padding='SAME') * mask
  m_w = tf.concat([m_w, m_w, m_w], axis=3)
  return m_w

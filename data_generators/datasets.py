from keras.preprocessing.image import ImageDataGenerator

NORM_MINUS_PLUS_ONE = lambda x: (x - 127.5) / 127.5
NORM_ZERO_ONE = lambda x: x * 1. / 255
NORM_MASK_WITH_NEGATION = lambda x: 1 - (x * 1. / 255)
NORM_MASK = lambda x: x * 1. / 255


class Dataset:
  
  def __init__(self, train_path, test_path, batch_size, img_height, img_width):
    data_generator = ImageDataGenerator(preprocessing_function=NORM_MINUS_PLUS_ONE)
    self._train_set = data_generator.flow_from_directory(train_path,
                                                         target_size=(img_height, img_width),
                                                         batch_size=batch_size,
                                                         class_mode=None)
    
    self._test_set = data_generator.flow_from_directory(test_path,
                                                        target_size=(img_height, img_width),
                                                        batch_size=batch_size,
                                                        class_mode=None)
  
  @property
  def train_set(self):
    return self._train_set
  
  @property
  def test_set(self):
    return self._test_set


class MaskDataset:
  def __init__(self, train_path, batch_size, img_height, img_width):
    data_generator = ImageDataGenerator(preprocessing_function=NORM_MASK)
    self._train_set = data_generator.flow_from_directory(train_path,
                                                         target_size=(img_height, img_width),
                                                         batch_size=batch_size,
                                                         class_mode=None)
  
  @property
  def train_set(self):
    return self._train_set
  
  @property
  def test_set(self):
    return self._train_set

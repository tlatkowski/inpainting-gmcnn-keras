import os

from utils import constants


class BaseModel:
  
  def __init__(self, img_height, img_width, num_channels, output_paths, model_name):
    self.img_height = img_height
    self.img_width = img_width
    self.num_channels = num_channels
    self.model_name = model_name
    self.output_paths = output_paths
    
    self.model = self.model()
    self.save_summary_to_file()
    
    self.summary()
  
  def model(self):
    raise NotImplementedError
  
  def disable(self):
    for layer in self.model.layers:
      layer.trainable = False
    self.model.trainable = False
  
  def enable(self):
    for layer in self.model.layers:
      layer.trainable = True
    self.model.trainable = True
  
  def summary(self):
    return self.model.summary()
  
  def save_summary_to_file(self):
    os.makedirs(self.output_paths.model_summary_path, exist_ok=True)
    model_summary_path = os.path.join(self.output_paths.model_summary_path, self.model_name + '.txt')
    with open(model_summary_path, 'w') as file:
      self.model.summary(print_fn=lambda x: file.write(x + '\n'))

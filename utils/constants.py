import os

OUTPUT_PATH = './outputs/{}'


class OutputPaths:
  
  def __init__(self, experiment_name):
    self.output_path = OUTPUT_PATH.format(experiment_name)
    self.output_weights_path = os.path.join(self.output_path, 'weights')
    self.predicted_pics_path = os.path.join(self.output_path, 'predicted_pics')
    self.predicted_pics_warm_up_path = os.path.join(self.predicted_pics_path, 'warm_up_generator')
    self.predicted_pics_wgan_path = os.path.join(self.predicted_pics_path, 'wgan')
    self.wgan_logs_path = os.path.join(self.output_path, 'logs/wgan')
    self.warm_up_logs_path = os.path.join(self.output_path, 'logs/warm_up_generator')
    self.model_summary_path = os.path.join(self.output_path, 'summaries')
    
    self.generator_weights_path = os.path.join(self.output_weights_path, 'gmcnn.h5')
    self.global_critic_weights_path = os.path.join(self.output_weights_path, 'global_critic.h5')
    self.local_critic_weights_file = os.path.join(self.output_weights_path, 'local_critic.h5')

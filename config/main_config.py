import configparser


def parse_list(x: str):
  values = x.strip().split(',')
  values = [int(v) for v in values]
  return values


class MainConfig:
  
  def __init__(self, config_path):
    model_config = configparser.ConfigParser()
    model_config.read(config_path)
    self.training = TrainingConfig(model_config['TRAINING'])
    self.model = ModelConfig(model_config['MODEL'])


class TrainingConfig:
  
  def __init__(self, training_section):
    self.wgan_training_ratio = int(training_section['WGAN_TRAINING_RATIO'])
    self.num_epochs = int(training_section['NUM_EPOCHS'])
    self.batch_size = int(training_section['BATCH_SIZE'])
    self.learning_rate = float(training_section['LEARNING_RATE'])
    self.img_height = int(training_section['IMG_HEIGHT'])
    self.img_width = int(training_section['IMG_WIDTH'])
    self.num_channels = int(training_section['NUM_CHANNELS'])
    self.save_model_steps_period = int(training_section['SAVE_MODEL_STEPS_PERIOD'])


class ModelConfig:
  
  def __init__(self, model_section):
    self.gradient_penalty_loss_weight = int(model_section['GRADIENT_PENALTY_LOSS_WEIGHT'])
    self.id_mrf_loss_weight = float(model_section['ID_MRF_LOSS_WEIGHT'])
    self.adversarial_loss_weight = float(model_section['ADVERSARIAL_LOSS_WEIGHT'])
    self.num_gaussian_steps = int(model_section['NUM_GAUSSIAN_STEPS'])
    self.nn_stretch_sigma = float(model_section['NN_STRETCH_SIGMA'])
    self.vgg_16_layers = parse_list(model_section['VGG_16_LAYERS'])

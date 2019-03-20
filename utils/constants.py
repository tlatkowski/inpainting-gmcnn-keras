import os

OUTPUT_PATH = './outputs'
OUTPUT_WEIGHTS_PATH = os.path.join(OUTPUT_PATH, 'weights')
PREDICTED_PICS_PATH = os.path.join(OUTPUT_PATH, 'predicted_pics')
PREDICTED_PICS_WARM_UP_PATH = os.path.join(PREDICTED_PICS_PATH, 'warm_up_generator')
PREDICTED_PICS_WGAN_PATH = os.path.join(PREDICTED_PICS_PATH, 'wgan')
WGAN_LOGS_PATH = os.path.join(OUTPUT_PATH, 'logs/wgan')
WARM_UP_LOGS_PATH = os.path.join(OUTPUT_PATH, 'logs/warm_up_generator')
MODEL_SUMMARY_PATH = os.path.join(OUTPUT_PATH, 'summaries')

GENERATOR_WEIGHTS_FILE = os.path.join(OUTPUT_WEIGHTS_PATH, 'gmcnn.h5')
GLOBAL_CRITIC_WEIGHTS_FILE = os.path.join(OUTPUT_WEIGHTS_PATH, 'global_critic.h5')
LOCAL_CRITIC_WEIGHTS_FILE = os.path.join(OUTPUT_WEIGHTS_PATH, 'local_critic.h5')

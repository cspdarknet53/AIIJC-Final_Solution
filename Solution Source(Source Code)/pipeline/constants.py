import os


SEED = 42
IMG_SIZE = (224, 224)

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
DEFAULT_DATA_PATH = os.path.join(PROJECT_PATH, 'data')
DEFAULT_EXPERIMENTS_SAVE_PATH = os.path.join('/content/drive/MyDrive', 'experiments')

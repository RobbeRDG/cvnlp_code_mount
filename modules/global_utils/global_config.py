from os.path import join
import pandas as pd
from helper_functions.label_encoder import LabelEncoder

# Set the base path
BASE_PATH = 'code_mount'

# Set the dataset folder paths
DATASET_FOLDER_PATH = join(BASE_PATH, 'meld_dataset')
TRAIN_DATA_FOLDER_PATH = join(DATASET_FOLDER_PATH, 'train_data')
TRAIN_AUDIO_DATA_FOLDER_PATH = join(TRAIN_DATA_FOLDER_PATH, 'audio')
TRAIN_VIDEO_DATA_FOLDER_PATH = join(TRAIN_DATA_FOLDER_PATH, 'video')
DEV_DATA_FOLDER_PATH = join(DATASET_FOLDER_PATH, 'dev_data')
DEV_AUDIO_DATA_FOLDER_PATH = join(DEV_DATA_FOLDER_PATH, 'audio')
DEV_VIDEO_DATA_FOLDER_PATH = join(DEV_DATA_FOLDER_PATH, 'video')
TEST_DATA_FOLDER_PATH = join(DATASET_FOLDER_PATH, 'test_data')
TEST_AUDIO_DATA_FOLDER_PATH = join(TEST_DATA_FOLDER_PATH, 'audio')
TEST_VIDEO_DATA_FOLDER_PATH = join(TEST_DATA_FOLDER_PATH, 'video')

# Set the labels path
LABELS_FOLDER_PATH = join(DATASET_FOLDER_PATH, 'labels')
TRAIN_LABELS_RAW = join(LABELS_FOLDER_PATH, 'train_sent_emo.csv')
DEV_LABELS_RAW = join(LABELS_FOLDER_PATH, 'dev_sent_emo.csv')
TEST_LABELS_RAW = join(LABELS_FOLDER_PATH, 'test_sent_emo.csv')
TRAIN_LABELS_CLEAN = join(LABELS_FOLDER_PATH, 'train_sent_clean.csv')
DEV_LABELS_CLEAN = join(LABELS_FOLDER_PATH, 'dev_sent_clean.csv')
TEST_LABELS_CLEAN = join(LABELS_FOLDER_PATH, 'test_sent_clean.csv')

# Set the labels and dataset paths combinations
LABELS_AND_DATA_PATHS_DICT = {
    'train': {
        'labels': TRAIN_LABELS_CLEAN,
        'data': TRAIN_AUDIO_DATA_FOLDER_PATH
    },
    'dev': {
        'labels': DEV_LABELS_CLEAN,
        'data': DEV_AUDIO_DATA_FOLDER_PATH
    },
    'test': {
        'labels': TEST_LABELS_CLEAN,
        'data': TEST_AUDIO_DATA_FOLDER_PATH
    }
}

# Audio parameters
SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH_SECONDS = 15.0
MAX_SPEECH_MODEL_AUDIO_LENGTH_SECONDS = 5.0

# Model parameters
MODEL_NAME = 'facebook/wav2vec2-base'
HUBERT_HIDDEN_SIZE = 768

# Static dataloader parameters
TRAINING_NUM_WORKERS = 4
VALIDATION_NUM_WORKERS = 4
VALIDATION_BATCH_SIZE = 16

# Emotion label to one-hot index mapping
EMOTION_LABEL_TO_ONE_HOT_INDEX = {
    'neutral': 0,
    'surprise': 1,
    'fear': 2,
    'sadness': 3,
    'joy': 4,
    'disgust': 5,
    'anger': 6
}

# Emotion labels encoder
EMOTION_LABEL_ENCODER = LabelEncoder(EMOTION_LABEL_TO_ONE_HOT_INDEX)

# Number of emotion classes
NUM_EMOTION_CLASSES = len(EMOTION_LABEL_TO_ONE_HOT_INDEX)


# Emotion weights
EMOTION_WEIGHTS_DICT = {
    'neutral': 2.129894344313238,
    'surprise': 8.378973105134474,
    'fear': 38.29050279329609,
    'sadness': 13.68063872255489,
    'joy': 5.9393414211438476,
    'disgust': 37.97229916897507,
    'anger': 8.530180460485376
}


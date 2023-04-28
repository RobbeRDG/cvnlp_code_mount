from os.path import join
import pandas as pd

# Set the base path
BASE_PATH = 'code_mount'

# Set the dataset folder paths
DATASET_FOLDER_PATH = join(BASE_PATH, 'meld_dataset')
TRAIN_DATA_FOLDER_PATH = join(DATASET_FOLDER_PATH, 'train_data/audio')
DEV_DATA_FOLDER_PATH = join(DATASET_FOLDER_PATH, 'dev_data/audio')
TEST_DATA_FOLDER_PATH = join(DATASET_FOLDER_PATH, 'test_data/audio')

# Set the labels path
LABELS_FOLDER_PATH = join(DATASET_FOLDER_PATH, 'labels')
TRAIN_LABELS = join(LABELS_FOLDER_PATH, 'train_sent_emo_transformed.csv')
DEV_LABELS = join(LABELS_FOLDER_PATH, 'dev_sent_emo_transformed.csv')
TEST_LABELS = join(LABELS_FOLDER_PATH, 'test_sent_emo_transformed.csv')
FILTERED_TRAIN_LABELS = join(LABELS_FOLDER_PATH, 'train_sent_emo_transformed_filtered.csv')
FILTERED_DEV_LABELS = join(LABELS_FOLDER_PATH, 'dev_sent_emo_transformed_filtered.csv')
FILTERED_TEST_LABELS = join(LABELS_FOLDER_PATH, 'test_sent_emo_transformed_filtered.csv')

# Set the labels and dataset paths combinations
LABELS_AND_DATA_PATHS_DICT = {
    'train': {
        'labels': TRAIN_LABELS,
        'data': TRAIN_DATA_FOLDER_PATH
    },
    'dev': {
        'labels': DEV_LABELS,
        'data': DEV_DATA_FOLDER_PATH
    },
    'test': {
        'labels': TEST_LABELS,
        'data': TEST_DATA_FOLDER_PATH
    }
}

# Audio parameters
SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH_SECONDS = 5.0

# Model parameters
MODEL_NAME = 'facebook/hubert-base-ls960'
HUBERT_HIDDEN_SIZE = 768

# Static dataloader parameters
TRAINING_NUM_WORKERS = 0
VALIDATION_NUM_WORKERS = 0
VALIDATION_BATCH_SIZE = 32

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

# Number of emotion classes
NUM_EMOTION_CLASSES = len(EMOTION_LABEL_TO_ONE_HOT_INDEX)


# Emotion weights
EMOTION_WEIGHTS = {
    'neutral': 2.129894344313238,
    'surprise': 8.378973105134474,
    'fear': 38.29050279329609,
    'sadness': 13.68063872255489,
    'joy': 5.9393414211438476,
    'disgust': 37.97229916897507,
    'anger': 8.530180460485376
}


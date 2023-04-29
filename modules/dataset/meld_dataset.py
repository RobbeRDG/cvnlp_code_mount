import torch
from torch.utils.data import Dataset
from os.path import join
import torchaudio
from torchaudio.utils import download_asset
from transformers import Wav2Vec2FeatureExtractor
import global_utils.global_config as global_config

class MELDDataset(Dataset):
    def __init__(
        self,
        labels,
        audio_folder_path
    ):
        # Set the object attributes
        self.labels = labels
        self.audio_folder_path = audio_folder_path
        self.sample_rate = global_config.SAMPLE_RATE
        self.max_audio_length_seconds = global_config.MAX_AUDIO_LENGTH_SECONDS
        self.max_audio_length_samples = int(self.max_audio_length_seconds * self.sample_rate)
        self.emotion_label_encoder = global_config.EMOTION_LABEL_ENCODER

        # Set the feature extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            global_config.MODEL_NAME
        )

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        # Get the label
        label_row = self.labels.iloc[index]

        # Process the item
        return self.process_item(label_row, 'extracted_features')

    def get_waveform(self, index):
        # Get the label
        label_row = self.labels.iloc[index]

        # Process the item
        return self.process_item(label_row, 'waveform')

    def process_item(self, label_row, out_type):
        # Extract label row info
        dialogue_id = label_row['Dialogue_ID']
        utterance_id = label_row['Utterance_ID']

        # Get the audio waveform
        waveform = self.load_waveform(dialogue_id, utterance_id)

        # Extract the label from the label row as a one-hot vector
        one_hot_label = self.emotion_label_encoder.label_to_one_hot(label_row['Emotion'])

        # Create a metadata tensor
        metadata = torch.tensor([
            dialogue_id,
            utterance_id
        ])

        # Get the audio features as a dict containing pt tensors
        if out_type == 'extracted_features':
            audio_features = self.feature_extractor(
                waveform,
                sampling_rate=self.sample_rate,
                padding='max_length',
                max_length=self.max_audio_length_samples,
                padding_value=0.0,
                truncation=True,
                return_tensors='pt'
            )

            # From the audio features dict, extract only the input values
            audio_features = audio_features['input_values']

            return audio_features, one_hot_label, metadata
        elif out_type == 'waveform':
            return waveform, one_hot_label, metadata
        else:
            raise Exception('Invalid out_type')

    def load_waveform(self, dialogue_id, utterance_id):
        # Get the audio file path
        audio_file_name = f'dia{dialogue_id}_utt{utterance_id}.wav'
        audio_file_path = join(self.audio_folder_path, audio_file_name)

        # Load the audio
        waveform, sample_rate = torchaudio.load(audio_file_path) # type: ignore

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True) # Recommened by torchaudio

        # Resample if needed
        if sample_rate != self.sample_rate:
            waveform = torchaudio.transforms.Resample(
                sample_rate,
                self.sample_rate
            )(waveform)

        # Convert the waveform to numpy array
        waveform = waveform[0].numpy()

        return waveform
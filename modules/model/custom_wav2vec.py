import torch
import torch.nn as nn
import transformers
import global_utils.global_config as global_config

class CustomWav2Vec2(nn.Module):
    def __init__(
            self,
            model_config
        ):
        super().__init__()

        # Extract the model config
        self.dropout_probability = model_config['dropout_probability']
        self.pooling_strategy = model_config['pooling_strategy']

        # Set the base hubert model
        self.base_wav2vec_model = transformers.Wav2Vec2Model.from_pretrained(global_config.MODEL_NAME)

        # Set the classification head
        self.classification_head = nn.Sequential(
            nn.Dropout(p=self.dropout_probability),
            nn.Linear(
                in_features=global_config.HUBERT_HIDDEN_SIZE,
                out_features=global_config.NUM_EMOTION_CLASSES
            ),
            nn.Softmax(dim=1)
        )

    def forward(self, audio_features):
        # Get the last hidden state from the base wav2vec model
        last_hidden_state = self.base_wav2vec_model(
            input_values=audio_features[:,0,:],
            output_hidden_states=False,
            return_dict=True
        ).last_hidden_state # type: ignore

        # Get the pooled hidden states
        if self.pooling_strategy == 'mean':
            pooled_last_hidden_state = torch.mean(last_hidden_state, dim=1)
        elif self.pooling_strategy == 'max':
            pooled_last_hidden_state = torch.max(last_hidden_state, dim=1)
        else:
            raise ValueError(f'Pooling strategy {self.pooling_strategy} not supported')
        
        # Get the final emotion probabilities
        emotion_probabilities = self.classification_head(pooled_last_hidden_state)

        return emotion_probabilities


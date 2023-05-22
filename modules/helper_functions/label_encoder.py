import numpy as np
import torch

class LabelEncoder():
    def __init__(self, label_to_one_hot_index_mapping):
        self.label_to_one_hot_index_mapping = label_to_one_hot_index_mapping
        self.one_hot_index_to_label_mapping = self.init_one_hot_index_to_label_mapping()
        self.label_to_one_hot_mapping = self.init_label_to_one_hot_mapping()
        self.one_hot_to_label_mapping = self.init_one_hot_to_label_mapping()

    def init_one_hot_index_to_label_mapping(self):
        one_hot_index_to_label_mapping = {}
        for label, one_hot_index in self.label_to_one_hot_index_mapping.items():
            one_hot_index_to_label_mapping[one_hot_index] = label
        return one_hot_index_to_label_mapping

    def init_label_to_one_hot_mapping(self):
        label_to_one_hot_mapping = {}
        for label, one_hot_index in self.label_to_one_hot_index_mapping.items():
            # Initialize an all zero numpy array
            one_hot = np.zeros(len(self.label_to_one_hot_index_mapping))

            # Set the one-hot index to 1
            one_hot[one_hot_index] = 1

            # Add to the mapping
            label_to_one_hot_mapping[label] = one_hot
        return label_to_one_hot_mapping
    
    def init_one_hot_to_label_mapping(self):
        one_hot_to_label_mapping = {}
        for label, one_hot in self.label_to_one_hot_mapping.items():
            one_hot_to_label_mapping[str(one_hot)] = label
        return one_hot_to_label_mapping
    
    def label_to_one_hot(self, label):
        return torch.tensor(self.label_to_one_hot_mapping[label])
    
    def label_to_one_hot_index(self, label):
        return self.label_to_one_hot_index_mapping[label]
    
    def one_hot_index_to_one_hot(self, one_hot_index):
        return torch.tensor(self.label_to_one_hot_mapping[self.one_hot_index_to_label_mapping[one_hot_index]])
    
    def one_hot_to_label(self, one_hot):
        # Convert the tensor to a numpy array
        one_hot = one_hot.numpy()

        return self.one_hot_to_label_mapping[str(one_hot)]
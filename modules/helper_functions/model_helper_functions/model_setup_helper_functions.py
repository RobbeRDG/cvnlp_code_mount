from dataset.meld_dataset import MELDDataset
from os.path import join
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
import torchvision
import torch.nn as nn
from torch.optim import AdamW
import global_utils.global_config as global_config
from model.custom_wav2vec import CustomWav2Vec2
import numpy as np

def get_dataset(
        data_id, 
        project_base_path
    ):
    # Set the labels and data full paths
    labels_path = join(project_base_path, global_config.LABELS_AND_DATA_PATHS_DICT[data_id]['labels'])
    data_path = join(project_base_path, global_config.LABELS_AND_DATA_PATHS_DICT[data_id]['data'])

    # Get the dataset
    return MELDDataset(
        labels=pd.read_csv(labels_path),    
        audio_folder_path=data_path
    )

def get_train_and_validation_dataset_and_dataloader(
        training_data_ids,
        validation_data_ids,
        train_batch_size,
        class_weights_dict,
        project_base_path       
):    
    # Get the train dataset and dataloader
    train_dataset, train_dataloader = get_train_dataset_and_dataloader(
        training_data_ids=training_data_ids,
        train_batch_size=train_batch_size,
        class_weights_dict=class_weights_dict,
        project_base_path=project_base_path    
    )

    # Get the validation dataset and dataloader
    validation_dataset, validation_dataloader = get_validation_dataset_and_dataloader(
        validation_data_ids=validation_data_ids,
        project_base_path=project_base_path
    )

    return train_dataset, train_dataloader, validation_dataset, validation_dataloader

def get_train_dataset_and_dataloader(
        training_data_ids,
        train_batch_size,
        class_weights_dict,
        project_base_path
    ):
    # Get the all the datasets we want to train on
    training_datasets = []
    for training_data_id in training_data_ids:
        training_datasets.append(
            get_dataset(
                data_id=training_data_id,
                project_base_path=project_base_path
            )
        )
    
    # For each dataset, get the labels dataframe and concatenate them
    labels_df_list = []
    for training_dataset in training_datasets:
        labels_df_list.append(training_dataset.get_labels_df())
    labels_df = pd.concat(labels_df_list)

    # Concatenate the datasets
    train_dataset = ConcatDataset(training_datasets) # type: ignore

    # Set the class weights
    class_weights_list = np.ones(len(train_dataset)).tolist()
    if class_weights_dict is not None:
        class_weights_list = get_weights_list_for_samples(labels_df, class_weights_dict)

    # Get the sampler
    train_sampler = WeightedRandomSampler(
        weights=class_weights_list,
        num_samples=len(train_dataset),
        replacement=True
    )

    # Get the train dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=global_config.TRAINING_NUM_WORKERS,
        pin_memory=True
    )

    return train_dataset, train_dataloader

def get_validation_dataset_and_dataloader(
        validation_data_ids,
        project_base_path
    ):
    # Get the all the datasets we want to validate on
    validation_datasets = []
    for validation_data_id in validation_data_ids:
        validation_datasets.append(
            get_dataset(
                data_id=validation_data_id,
                project_base_path=project_base_path
            )
        )

    # Concatenate the datasets
    validation_dataset = torch.utils.data.ConcatDataset(validation_datasets) # type: ignore

    # Get the validation dataloader
    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        batch_size=global_config.VALIDATION_BATCH_SIZE,
        shuffle=False,
        num_workers=global_config.VALIDATION_NUM_WORKERS,
        pin_memory=True
    )

    return validation_dataset, validation_dataloader

def get_weights_list_for_samples(
        labels_of_samples,
        class_weights_dict
    ):
    # Print update
    print('Assigning weights for samples based on class occurence...')

    # Iterate over the labels and assign a weight to each sample based on its class
    samples_weights = []
    for idx, label in labels_of_samples.iterrows():
        # Get the class of the sample
        sample_emotion_class = label['Emotion']

        # Get the weight of the class
        sample_weight = class_weights_dict[sample_emotion_class]

        # Add the weight to the list
        samples_weights.append(sample_weight)
    
    # Print update
    print('Done assigning weights for samples based on class occurence.')
    return samples_weights

def get_augmentation(transform_id):
    raise KeyError(f'Transform: \'{transform_id}\' not found')

def build_transform_module_lists_dict(
        augmentation_config_dict
    ):
    # Standard train transforms
    train_input_transform_module_list = []
    train_target_transform_module_list = []
    train_both_transform_module_list = []
    

    # Add the train augmentations
    if augmentation_config_dict['train_input_augmentation_ids']:
        for train_input_augmentation_id in augmentation_config_dict['train_input_augmentation_ids']:
            train_input_transform_module_list.append(get_augmentation(train_input_augmentation_id))
    if augmentation_config_dict['train_target_augmentation_ids']:
        for train_target_augmentation_id in augmentation_config_dict['train_target_augmentation_ids']:
            train_target_transform_module_list.append(get_augmentation(train_target_augmentation_id))
    if augmentation_config_dict['train_both_augmentation_ids']:
        for train_both_augmentation_id in augmentation_config_dict['train_both_augmentation_ids']:
            train_both_transform_module_list.append(get_augmentation(train_both_augmentation_id))

    # Standard validation transforms
    validation_input_transform_module_list = []
    validation_target_transform_module_list = []
    validation_both_transform_module_list = []

    # Add the validation augmentations
    if augmentation_config_dict['validation_input_augmentation_ids']:
        for validation_input_augmentation_id in augmentation_config_dict['validation_input_augmentation_ids']:
            validation_input_transform_module_list.append(get_augmentation(validation_input_augmentation_id))
    if augmentation_config_dict['validation_target_augmentation_ids']:
        for validation_target_augmentation_id in augmentation_config_dict['validation_target_augmentation_ids']:
            validation_target_transform_module_list.append(get_augmentation(validation_target_augmentation_id))
    if augmentation_config_dict['validation_both_augmentation_ids']:
        for validation_both_augmentation_id in augmentation_config_dict['validation_both_augmentation_ids']:
            validation_both_transform_module_list.append(get_augmentation(validation_both_augmentation_id))

    return {
        'train_input_transform_module_list': train_input_transform_module_list,
        'train_target_transform_module_list': train_target_transform_module_list,
        'train_both_transform_module_list': train_both_transform_module_list,
        'validation_input_transform_module_list': validation_input_transform_module_list,
        'validation_target_transform_module_list': validation_target_transform_module_list,
        'validation_both_transform_module_list': validation_both_transform_module_list,
    }

def get_model(model_id, model_config_dict):
    # Get the model
    model = None
    if model_id == 'wav2vec2':
        model = CustomWav2Vec2(model_config=model_config_dict)
    else:
        raise KeyError(f'Model: \'{model_id}\' not found')
    
    # Freeze the feature extractor layers of hubert if specified
    if model_config_dict['freeze_feature_extractor']:
        model.base_wav2vec_model.feature_extractor._freeze_parameters() # type: ignore

    return model

def get_loss_function(loss_function_id):
    if loss_function_id == 'cross_entropy':
        return nn.CrossEntropyLoss()
    else:
        raise KeyError(f'Loss: \'{loss_function_id}\' not found')

def get_optimizer(optimizer_id, learning_rate, model):
    if optimizer_id == 'adamw':
        return AdamW(
            params=model.parameters(),
            lr=learning_rate
        )
    else:
        raise KeyError(f'Optimizer: \'{optimizer_id}\' not found')




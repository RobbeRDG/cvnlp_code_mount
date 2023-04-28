from dataset.meld_dataset import MELDDataset
from os.path import join
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
from torch.optim import AdamW
import global_utils.global_config as global_config

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
        labels_and_data_paths_dict,
        train_batch_size,
        project_base_path       
):    
    # Get the train dataset and dataloader
    train_dataset, train_dataloader = get_train_dataset_and_dataloader(
        training_data_ids=training_data_ids,
        labels_and_data_paths_dict=labels_and_data_paths_dict,
        train_batch_size=train_batch_size,
        project_base_path=project_base_path    
    )

    # Get the validation dataset and dataloader
    validation_dataset, validation_dataloader = get_validation_dataset_and_dataloader(
        validation_data_ids=validation_data_ids,
        labels_and_data_paths_dict=labels_and_data_paths_dict,
        project_base_path=project_base_path
    )

    return train_dataset, train_dataloader, validation_dataset, validation_dataloader

def get_train_dataset_and_dataloader(
        training_data_ids,
        labels_and_data_paths_dict,
        train_batch_size,
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

    # Concatenate the datasets
    train_dataset = torch.utils.data.ConcatDataset(training_datasets) # type: ignore

    # Get the train dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=global_config.TRAINING_NUM_WORKERS,
        pin_memory=True
    )

    return train_dataset, train_dataloader

def get_validation_dataset_and_dataloader(
        validation_data_ids,
        labels_and_data_paths_dict,
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

def get_augmentation(transform_id):
    raise KeyError(f'Transform: \'{transform_id}\' not found')

def build_transform_module_lists_dict(
        augmentation_config_dict,
        sample_dimensions
    ):
    # Standard train transforms
    train_input_transform_module_list = [
        torchvision.transforms.Resize(sample_dimensions),
        #torchvision.transforms.Normalize(
        #    mean=[0.0126, 0.0126, 0.0126],
        #    std=[0.7618, 0.7618, 0.7618]
        #),
    ]
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
    validation_input_transform_module_list = [
        torchvision.transforms.Resize(sample_dimensions),
        #torchvision.transforms.Normalize(
        #    mean=[0.0126, 0.0126, 0.0126],
        #    std=[0.7618, 0.7618, 0.7618]
        #)
    ]
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
    model = None
    if model_id == 'resnet_18_untrained':
        pass
    else:
        raise KeyError(f'Model: \'{model_id}\' not found')
    
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




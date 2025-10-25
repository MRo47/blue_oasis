from pathlib import Path
from typing import Callable

import librosa
import numpy as np
import pandas as pd
import torch
from augment import spec_augment, time_domain_augment
from data_splitter import filter_classes, stratified_group_split
from input_output import load_sample
from preprocessing import extract_mel_db_spec, preprocess
from torch.utils.data import DataLoader, Dataset

# Assume your 'preprocess', 'extract_mel_db_spec', 'spec_augment', 
# and 'time_augment_hybrid' are defined as before.


class BirdSoundDataset(Dataset):
    def __init__(self, 
                 df: pd.DataFrame, 
                 recordings_path: Path,
                 mode: str = 'train',
                 sr: int = 22050,
                 duration: float = 3.0,
                 preprocess_fn: Callable = None,
                 time_augment_fn: Callable = None,
                 spec_augment_fn: Callable = None):
        """
        PyTorch Dataset for bird sound classification.

        Args:
            df (pd.DataFrame): DataFrame with columns 'Filename', 'Start Time (s)', 
                               'End Time (s)', and 'Species eBird Code'.
            recordings_path (Path): Base path to the directory containing audio files.
            mode (str): One of 'train', 'valid', or 'test'. Controls augmentation.
            sr (int): Target sampling rate.
            duration (float): Duration of the audio clips to load in seconds.
            preprocess_fn (callable): The preprocessing function.
            time_augment_fn (callable): The time-domain augmentation function.
            spec_augment_fn (callable): The frequency-domain (spectrogram) augmentation function.
        """
        self.df = df.reset_index(drop=True)
        self.recordings_path = recordings_path
        self.mode = mode.lower()
        self.sr = sr
        self.duration = duration
        
        # Store the functions
        self.preprocess = preprocess_fn
        self.time_augment = time_augment_fn
        self.spec_augment = spec_augment_fn
        
        # Create a mapping from species code to an integer index
        self.classes = sorted(self.df['Species eBird Code'].unique())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for i, cls_name in enumerate(self.classes)}
        
        if self.mode not in ['train', 'valid', 'test']:
            raise ValueError("Mode must be 'train', 'valid', or 'test'.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Get file path and label
        file_path = self.recordings_path / row['Filename']
        label_str = row['Species eBird Code']
        label = self.class_to_idx[label_str]
        
        # 2. Load audio sample
        center_time = (row['Start Time (s)'] + row['End Time (s)']) / 2
        # Assuming load_sample is defined elsewhere
        sample, _ = load_sample(file_path, center_time, dur=self.duration, sr_target=self.sr)
        
        # 3. Apply standard preprocessing
        # Assuming extract_mel_db_spec is part of the preprocessing module/namespace
        prep_sample = self.preprocess(sample, self.sr)

        # 4. Conditionally apply augmentations
        if self.mode == 'train':
            # Apply time-domain augmentations
            if self.time_augment:
                time_aug_sample = self.time_augment(prep_sample, sr=self.sr)
            else:
                time_aug_sample = prep_sample # No time augmentation if function not provided

            # Extract spectrogram from the (potentially) augmented audio
            mel_db_spec, _ = extract_mel_db_spec(time_aug_sample, sr=self.sr)

            # Apply frequency-domain augmentations
            if self.spec_augment:
                aug_spec = self.spec_augment(mel_db_spec)
            else:
                aug_spec = mel_db_spec # No spec augmentation if function not provided
        else: # For 'valid' and 'test' modes
            # No augmentations are applied
            mel_db_spec, _ = extract_mel_db_spec(prep_sample, sr=self.sr)
            aug_spec = mel_db_spec

        # 5. Convert to PyTorch Tensor
        # Add a channel dimension: (n_mels, time) -> (1, n_mels, time)
        spec_tensor = torch.tensor(aug_spec).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return spec_tensor, label_tensor

def main():
	annotations_csv = Path("~/data/kenya_birds/annotations.csv").expanduser()
	recordings_path = Path("~/data/kenya_birds/soundscape_data/").expanduser()
	SR = 22050
	DURATION = 3.0
	BATCH_SIZE = 32
	min_samples = 10
    
	df = pd.read_csv(annotations_csv)

	df = filter_classes(df,  min_sample_size=min_samples)

	train_df, val_df, test_df = stratified_group_split(
		df,
		label_id='Species eBird Code',
		group_id='Filename',
		test_size=0.2,
		val_size=0.2,
	)

	# Training dataset with all augmentations enabled
	train_dataset = BirdSoundDataset(
		df=train_df,
		recordings_path=recordings_path,
		mode='train',
		sr=SR,
		duration=DURATION,
		preprocess_fn=preprocess,  # Pass the whole module/object
		time_augment_fn=time_domain_augment,
		spec_augment_fn=spec_augment
	)

	# Validation dataset with NO augmentations
	val_dataset = BirdSoundDataset(
		df=val_df,
		recordings_path=recordings_path,
		mode='valid', # This ensures no augmentations are applied
		sr=SR,
		duration=DURATION,
		preprocess_fn=preprocess,
		time_augment_fn=None, # Explicitly disable augmentations
		spec_augment_fn=None
	)

	# Test dataset also has NO augmentations
	test_dataset = BirdSoundDataset(
		df=test_df,
		recordings_path=recordings_path,
		mode='test',
		sr=SR,
		duration=DURATION,
		preprocess_fn=preprocess,
		time_augment_fn=None,
		spec_augment_fn=None
	)
    
	# data loaders
	train_loader = DataLoader(
		train_dataset, 
		batch_size=BATCH_SIZE, 
		shuffle=True, 
		num_workers=4, # Use multiple CPU cores to load data in parallel
		pin_memory=True
	)

	val_loader = DataLoader(
		val_dataset, 
		batch_size=BATCH_SIZE * 2, # Can often use a larger batch size for validation
		shuffle=False, 
		num_workers=4,
		pin_memory=True
	)

	# --- Example of iterating through a batch ---
	print(f"Number of training samples: {len(train_dataset)}")
	print(f"Number of validation samples: {len(val_dataset)}")
	print(f"Number of classes: {len(train_dataset.classes)}")

	# Get one batch of training data
	spec_batch, label_batch = next(iter(train_loader))

	print("\n--- Batch Shapes ---")
	print(f"Spectrogram batch shape: {spec_batch.shape}")
	print(f"Label batch shape: {label_batch.shape}")
    
if __name__ == "__main__":
    main()
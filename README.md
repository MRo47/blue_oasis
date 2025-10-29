# Bird sound classification

This project is a development environment and report of building a bird calls classifier on a dataset recorded in Kenya.

## Setup

### Environment setup
- This project uses VSCode devcontainers for development, install the plugin: [ms-vscode-remote.remote-containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
- then build and start the container in VSCode by pressing CTRL+SHIFT+P > Dev Containers: Rebuild and Reopen Container
- this should also install the project `bird_classifier` and should be avaiable in the virtual environment `~/.venv`

### Download dataset
- change the mount point in [docker-compose.yml](.devcontainer/docker-compose.yml) for data if the dataset is downloaded on host under volumes.
- or download the dataset as follows
```bash
cd ~/data
wget -c -O soundscape_data.zip "https://zenodo.org/records/10943500/files/soundscape_data.zip?download=1"
```
- then unzip it in the same directory
```bash
unzip soundscape_data.zip
```
- optionally remove the zip
```bash
rm soundscape_data.zip
```

### Usage

- The readme explains most of the experiment, some code is just abbreviations (eg: scripts/train.py)
- Some modules have main() functions that can be executed to show some debug information (eg: bird_classifier/data_splitter.py will show number of samples in each split)
- The [notebooks](bird_classifier/notebooks) contains experiments done before creating the scripts.

## Dataset

An annotated bird datset from soundscape recordings is Kenya is used for this project published [here](https://zenodo.org/records/10943500)

The datset consists of 
- 10294 annotations
- 35 recordings
- 176 unique bird species
- location: west and southwest of Lake Baringo in Baringo County, Kenya.
- recording period: 12-07-2021 to 02-11-2022 
- duration: ~1 hour
- sample rate: 32khz

### Data quality

**Balance**
- The dataset is highly unbalanced in terms of species with annotations count ranging from 1 per species to 702 per species.
- Top 1% species have more than 498 annotations while the bottom 10% have less than 2 annotations.
- The annotations distribution plot can be seen below.
![species distribution](images/species_distribution.png)

**Duration anomalies**
- According to the accompanying metadata, the annotations are point annotations, 7 annotations were found with a non zero duration.

**Spectral ranges**
- The 98% of the bird calls range in frequency centered from 283.8 Hz to 7597.8 Hz.
- The frequency range plot of a subset of the species can be seen below. 
![spectral range](images/spectral_range.png)

**Background noise**
- The recordings have been taken over a period of time and few different locations, the noise level and quality changes between the recording.

**Overlaps**
- Many of the annotations have overlapping bird calls.

complete analysis can be found [here](notebooks/data_analysis.ipynb)

### Preprocessing

- Noise reduction was applied using the `noisereduce` library to even out the noise floor across recordings.
- High pass filtering is applied to reduce mic hum or wind noises.
- The sample is then normalized before computing the spectrogram.
- 3 different featue extractors were tested to represent the bird calls, since the calls fall in range of human hearing perception, mel spectrogram was chosen as the best representation.
- A 3 second window was chosen to compute the mel spectrogram.
- below are the 3 different feature representation of a single bird call.
  - STFT: short time fourier transform.
  - Mel Spectrogram.
  - Time frequency reassigned spectrogram. 
![features](images/features.png)

complete analysis can be found [here](notebooks/preprocessing.ipynb)

### Splitting stratergy

Given the unbalanced nature of the dataset and exitence of recording groups (different time and locations), a stratified group split was used to ensure no group leakage while distributing the species evenly across train, val and test sets.
Since some classes in the dataset have very few annotations its impossible to distribute them evenly across train, val and test sets, hence the following splitting stratergy was used.

1. Filter out classes that have less than 10 annotations.
2. Identify classes that have too few unique groups, call these problematic classes. These classes cannot be safely stratified across required number of splits to ensure no group leakage.
3. Partition the entire dataset by groups:
    - `stratifiable`: Contains only "well-behaved" groups where every class within them has sufficient group diversity.
    - `problematic`: Contains all the groups that were "contaminated" by at least one problematic class.
4. Split the stratifiable dataset using a nested StratifiedGroupKFold.
5. Split the problematic dataset using a nested GroupKFold (prioritizing group integrity only as stratification is not possible).
6. Combine the splits from both stratifiable and problematic datasets to form the final train, val, and test sets.

### Augmentations

The following data augmentations are applied to the training dataset to add variation and improve model robustness.

**Time domain augmentations**
1. Shift the audio signal fractionally left or right by 30%, to account for positional variation in recording.
2. Pitch shift the audio signal by +-4 semitones to account for pitch variation.
3. Speed shift the audio signal by 20%.
4. Volume shift the audio signal by 12dB to account for distance between recording equipment and the bird.
5. Add background noise
  - pink noise to simulate wind.
  - brown noise to simulate rain/rumble.
  - sample noise could be added from across the recordings but it was difficult to find a clean sample in teh recordings without bird sounds. Some experiments were performed [here](notebooks/noise_extraction.ipynb)

**Spectral domain augmentations**
1. Time masking was added to +-10 bins for robustness to missing sounds.
2. Frequency masking was added to +-10 bins for robustness to partial frequency data.

![spectral augmentations](images/augmentations.png)

augmentation experiments can be found [here](notebooks/augmentations.ipynb)

The full data augmentation and preprocessing pipeline can be found [here](notebooks/dataset.ipynb)

The dataset and pytorch data loaders are implemented in [dataset.py](bird_classifier/dataset.py)

## Model

### Selection
- Since the data size is limited (around 6k train 2k validation and 2k test samples) a pretrained model should be fine tuned, even better if its trained on audio datasets.
- A good starting point would be to find a model that is small and effcient to prevent overfitting.
- CNNs would be a better choice than transformers given the data size.

**ConvNext**
- for higher kernal sizes 7x7 
- benefits of transformers
- LayerNorm normalises per instace which is irrespective of levels in other spectrograms.

**EfficientNetV2**
- for training efficiency and inference speed
- BatchNorm will normalise across different samples which could have different volume levels.

### Training

**Loss function**
Weighted Cross-Entropy Loss: to account for the unbalanced dataset.
- calculate class weights
- pass as weights to the loss function

**Metrics**
- Macro F1 score (primary): Average of F1 score across all classes. Use this for hyperparameter tuning. This score treats all classes as equal hence is suitable for class imbalanced datasets.
- Weighted F1 score: Weighted average of F1 score across all classes, weighed by the fraction of samples in each class.
- Confusion matrix: Visualize the performance of the model.

**Wighted Random Sampler**
Oversample rare classes and undersample common classes.

**Optimizer**
AdamW optimizer to start with

**Learning rate**
low learning rate for fine-tuning (1e-4, 3e-3)

**Learning rate scheduler**
`torch.optim.lr_scheduler.CosineAnnealingLR` smoothly decreases the LR following a cosine curve down to 0 over a set of epochs.

Example training script can be found [here](scripts/train.py)
```bash
python train.py \
  --annotations_csv ~/data/kenya_birds/annotations.csv \
  --recordings_path ~/data/kenya_birds/soundscape_data \
  --save_path ~/models/bird_classifier \
  --batch_size 64 \
  --num_epochs 30 \
  --lr 5e-4
```

### Tuning (Hyperparameters)

- The datasets are already set up for cross validation.
- Use libraries like [Optuna](https://optuna.org/) to find the best hyperparameters.
- an examlple tuning using optuna can be found [here](scripts/tune.py)

Finally train the model on the whole dataset (except rare classes) after finding the best hyperparameters.

### Computational requirements

For example to train ConvNeXt-T model which has 28.6M parameters and needs 4.5 GFlops. [reference](https://github.com/facebookresearch/ConvNeXt)

**Training**


VRAM required(FP32) = model weights + gradients + optimizer states + activations

- Model weights = 28.6M params * 4 bytes/param = ~114 MB
- Gradients = 28.6M params * 4 bytes/param = ~114 MB
- Optimizer states = 28.6M params * 2 states * 4 bytes/param = ~228 MB
- Activations = sum (4 * product(output_size_of_layer) )

VRAM required(FP32) = 114M + 114M + 228M + activation memory

GFLOPS = 4.5 GFlops for 224 x 224 = 50176 pixels

model input = 128 x 130 = 16640 pixels

factor = 16640 / 50176 = 0.33

GFLOPS = 4.5 * 0.33 = ~1.5 GFlops per input spectrogram

One epoch = 6000 train images * (1.5 GFLOPs forward + ~3.0 GFLOPs backward) ≈ 27,000 GFLOPs or 27 TFLOPs.

Assuming 50 training epochs: 50 * 27 = 1,350 TFLOPs.

**Deployment**


VRAM required(FP32) = model weights + activations

model weights = 28.6M params * 4 bytes/param = ~114 MB

Activations = sum (4 * product(output_size_of_layer) )

VRAM required(FP16) = 114 + Activations



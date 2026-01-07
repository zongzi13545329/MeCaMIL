# MeCaMIL: Causality-Aware Multiple Instance Learning for Fair and Interpretable Whole Slide Image Diagnosis

Pytorch implementation for the causality-aware multiple instance learning model described in the paper "MeCaMIL: Causality-Aware Multiple Instance Learning for Fair and Interpretable Whole Slide Image Diagnosis".

## Installation
Install [anaconda/miniconda](https://docs.conda.io/en/latest/miniconda.html)  
Required packages
```
  $ conda env create --name mecamil --file env.yml
  $ conda activate mecamil
```
Install [PyTorch](https://pytorch.org/get-started/locally/)  
Install [OpenSlide and openslide-python](https://pypi.org/project/openslide-python/).

## Download feature vectors for MIL network

Precomputed features for [TCGA Lung Cancer dataset](https://portal.gdc.cancer.gov/repository?filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.cases.primary_site%22%2C%22value%22%3A%5B%22bronchus%20and%20lung%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.data_format%22%2C%22value%22%3A%5B%22svs%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.experimental_strategy%22%2C%22value%22%3A%5B%22Diagnostic%20Slide%22%5D%7D%7D%5D%7D) can be downloaded via:  
```
  $ python download.py --dataset=tcga
```
Precomputed features for [Camelyon16 dataset](https://camelyon16.grand-challenge.org/Data/)
```
  $ python download.py --dataset=c16
```
This dataset requires 30GB of free disk space.

## Training on default datasets

### WSI datasets 

#### Train with CausalMIL model (MeCaMIL)
>Train MeCaMIL on TCGA Lung Cancer dataset **without** demographic information:
 ```
  $ python train_tcga_causal.py --model=causalmil --dataset=TCGA-lung-default
```
>Train MeCaMIL on TCGA Lung Cancer dataset **with** demographic information:
 ```
  $ python train_tcga_causal.py --model=causalmil --dataset=TCGA-lung-default --use_demo --demographic_file=./datasets/tcga-dataset/processed_clinical_data.json
```
>Train MeCaMIL on Camelyon16 dataset:
 ```
  $ python train_tcga_causal.py --model=causalmil --dataset=Camelyon16 --num_classes=1
```

### Training evaluation
The training script automatically performs **5-fold cross-validation**. For each fold:
- A separate model is saved with optimal thresholds
- Training progress and metrics are displayed
- Final averaged results across all folds are reported

Results are saved in `weights/[DATE]/` directory:
- `[model]_[dataset]_fold_[X]_[run].pth`: Model weights
- `fold_[X]_[run].json`: Optimal thresholds for each class
- `training_summary.json`: Complete training statistics

### Useful arguments:
```
[--model]             # Model type: dsmil | abmil | causalmil [abmil]
[--num_classes]       # Number of non-negative classes, for binary classification set to 1 [2]
[--feats_size]        # Size of feature vector (depends on the CNN backbone) [512]
[--lr]                # Initial learning rate [0.0001]
[--num_epochs]        # Number of training epochs [50]
[--stop_epochs]       # Early stopping: skip remaining epochs if no improvement after N epochs [10]
[--weight_decay]      # Weight decay [1e-3]
[--dataset]           # Dataset folder name [TCGA-lung-default]
[--split]             # Training/validation split [0.2]
[--dropout_patch]     # Randomly dropout patches during training [0]
[--dropout_node]      # Dropout rate in bag classifier [0]
[--average]           # Average max-pooling and bag prediction scores [False]

# CausalMIL-specific arguments:
[--use_demo]          # Enable demographic information (requires demographic_file)
[--demographic_file]  # Path to demographic JSON file [./datasets/tcga-dataset/processed_clinical_data.json]
[--structural_depth]  # Depth of causal graph convolutional layers [1]
[--use_causal_graph]  # Enable causal graph structure in the model
```

### Demographic data format
When using `--use_demo`, the demographic JSON file should contain entries with the following structure:
```json
[
  {
    "submitter_id": "TCGA-XX-XXXX",
    "gender_vector": [0, 1],
    "race_vector": [1, 0, 0, 0, 0],
    "age_normalized": 0.65
  }
]
```
The demographic vector is 8-dimensional: 2 for gender + 5 for race + 1 for normalized age.

## Testing and generating detection maps from WSI

### Camelyon16 dataset
> Generating detection maps for Camelyon16. Direct download [link](https://uwmadison.box.com/shared/static/qs717clgaux5hx2mf5qnwmlsoz2elci2.zip) for the sample slides.
```
  $ python download.py --dataset=c16-test
  $ python test_crop_single.py --dataset=c16
  $ python testing_c16.py
```

## Processing raw WSI data
If you are processing WSI from raw images, you will need to download the WSIs first.

**Download WSIs.**  
* >**From GDC data portal.** You can use [GDC data portal](https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Getting_Started/) with a manifest file and configuration file. The raw WSIs take about 1TB of disc space and may take several days to download. Please check [details](https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Getting_Started/) regarding the use of TCGA data portal. Otherwise, individual WSIs can be download manually in GDC data portal [repository](https://portal.gdc.cancer.gov/repository?filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22content%22%3A%7B%22field%22%3A%22files.cases.primary_site%22%2C%22value%22%3A%5B%22bronchus%20and%20lung%22%5D%7D%2C%22op%22%3A%22in%22%7D%2C%7B%22content%22%3A%7B%22field%22%3A%22files.data_format%22%2C%22value%22%3A%5B%22svs%22%5D%7D%2C%22op%22%3A%22in%22%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.experimental_strategy%22%2C%22value%22%3A%5B%22Diagnostic%20Slide%22%5D%7D%7D%5D%7D)  
* >**From Google Drive.** The svs files are also [uploaded](https://drive.google.com/drive/folders/1UobMSqJEqINX2izxrwbgprugjlTporSQ?usp=sharing). The dataset contains in total 1053 slides, including 512 LUSC and 541 LUAD. 10 low-quality LUAD slides are discarded. 

**Sort svs files in the folder.**
Separate LUAD and LUSC slides according to the IDs and place the files into folder `WSI/TCGA-lung/LUAD` or `WSI/TCGA-lung/LUSC`. 

**Prepare the patches.**  
>We will be using [OpenSlide](https://openslide.org/), a C library with a [Python API](https://pypi.org/project/openslide-python/) that provides a simple interface to read WSI data. We refer the users to [OpenSlide Python API document](https://openslide.org/api/python/) for the details of using this tool.  
>The patches are cropped at a single level magnification and saved in './WSI/TCGA-lung/single'. For example, extract patches at 10x magnification:  
```
  $ python deepzoom_tiler.py -m 0 -b 10
```
>For Camelyon16:
```
 $ python deepzoom_tiler.py -m 1 -b 20 -d Camelyon16 -v tif
```

**Train the embedder.**  
>We provided a modified script from this repository [Pytorch implementation of SimCLR](https://github.com/sthalles/SimCLR) For training the embedder.  
Navigate to './simclr' and edit the attributes in the configuration file 'config.yaml'. You will need to determine a batch size that fits your gpu(s). We recommend using a batch size of at least 512 to get good simclr features. The trained model weights and loss log are saved in folder './simclr/runs'.  
```
  cd simclr
  $ python run.py
```
> Otherwise, you could use the trained embedder for [Camelyon16](https://drive.google.com/drive/folders/14pSKk2rnPJiJsGK2CQJXctP7fhRJZiyn?usp=sharing) and [TCGA](https://drive.google.com/drive/folders/1Rn_VpgM82VEfnjiVjDbObbBFHvs0V1OE?usp=sharing)  

**Compute the features.**  
>Compute the features:  
```
  $ cd ..
  $ python compute_feats.py --dataset=TCGA-lung
```
>The last trained embedder will be used by default. To use a specific embedder, set the option `--weights=[RUN_NAME]`, where `[RUN_NAME]` is a folder name inside `simclr/runs/`. 
>To use ImageNet pretrained CNN for feature computing (batch normalization needed):
```
  $ python compute_feats.py --weights=ImageNet --norm_layer=batch
```

**Start training.**  
```
  $ python train_tcga_causal.py --model=causalmil --dataset=TCGA-lung
```
>To use demographic information:
```
  $ python train_tcga_causal.py --model=causalmil --dataset=TCGA-lung --use_demo --demographic_file=[PATH_TO_DEMOGRAPHIC_JSON]
```

## Training on your own datasets
1. Place WSI files as `WSI\[DATASET_NAME]\[CATEGORY_NAME]\[SLIDE_FOLDER_NAME] (optional)\SLIDE_NAME.svs`. 
> For binary classifier, the negative class should have `[CATEGORY_NAME]` at index `0` when sorted alphabetically. For multi-class classifier, if you have a negative class (not belonging to any of the positive classes), the folder should have `[CATEGORY_NAME]` at **the last index** when sorted alphabetically. The naming of the class folders does not matter if you do not have a negative class.

2. Crop patches.  
```
  $ python deepzoom_tiler.py -m 0 -b 20 -d [DATASET_NAME]
```

3. Train an embedder.  
```
  $ cd simclr
  $ python run.py --dataset=[DATASET_NAME]
```

4. Compute features using the embedder.  
```
  $ cd ..
  $ python compute_feats.py --dataset=[DATASET_NAME]
```
>This will use the last trained embedder to compute the features, if you want to use an embedder from a specific run, add the option `--weights=[RUN_NAME]`, where `[RUN_NAME]` is a folder name inside `simclr/runs/`. If you have an embedder you want to use, you can place the weight file as `simclr/runs/[RUN_NAME]/checkpoints/model.pth` and pass the `[RUN_NAME]` to this option. The embedder architecture is ResNet18 with **instance normalization**.     

5. Prepare demographic data (optional, for CausalMIL only).
>If you want to use demographic information with CausalMIL, prepare a JSON file following the format:
```json
[
  {
    "submitter_id": "PATIENT-ID",
    "gender_vector": [0, 1],
    "race_vector": [1, 0, 0, 0, 0],
    "age_normalized": 0.65
  }
]
```
>The `submitter_id` should match the first three segments of your slide filename (e.g., for `TCGA-XX-XXXX.svs`, use `TCGA-XX-XXXX`).

6. Training.
```
  # Train with CausalMIL (without demographics)
  $ python train_tcga_causal.py --model=causalmil --dataset=[DATASET_NAME]
  
  # Train with CausalMIL (with demographics)
  $ python train_tcga_causal.py --model=causalmil --dataset=[DATASET_NAME] --use_demo --demographic_file=[PATH_TO_JSON]
```
>You will need to adjust `--num_classes` option if the dataset contains more than 2 positive classes or only 1 positive class and 1 negative class (binary classifier).

7. Testing.
```
  $ python attention_map.py --bag_path test/patches --map_path test/output --thres 0.73 0.28
```
Useful arguments:
```
[--num_classes]         # Number of non-negative classes.
[--feats_size]          # Size of feature vector (depends on the CNN backbone).
[--thres]               # List of thresholds for the classes returned by the training function.
[--embedder_weights]    # Path to the embedder weights file (saved by SimCLR). Use 'ImageNet' if ImageNet pretrained embedder is used.
[--aggregator_weights]  # Path to the aggregator weights file.
[--bag_path]            # Path to a folder containing folders of patches.
[--patch_ext]           # File extension of patches.
[--map_path]            # Path of output attention maps.
```

## Folder structures
Data is organized in two folders, `WSI` and `datasets`. `WSI` folder contains the images and `datasets` contains the computed features.
```
root
|-- WSI
|   |-- DATASET_NAME
|   |   |-- CLASS_1
|   |   |   |-- SLIDE_1.svs
|   |   |   |-- ...
|   |   |-- CLASS_2
|   |   |   |-- SLIDE_1.svs
|   |   |   |-- ...
```
Once patch extraction is performed, `single` folder will appear.
```
root
|-- WSI
|   |-- DATASET_NAME
|   |   |-- single
|   |   |   |-- CLASS_1
|   |   |   |   |-- SLIDE_1
|   |   |   |   |   |-- PATCH_1.jpeg
|   |   |   |   |   |-- ...
|   |   |   |   |-- ...
```
Once feature computing is performed, `DATASET_NAME` folder will appear inside `datasets` folder.
```
root
|-- datasets
|   |-- DATASET_NAME
|   |   |-- CLASS_1
|   |   |   |-- SLIDE_1.csv
|   |   |   |-- ...
|   |   |-- CLASS_2
|   |   |   |-- SLIDE_1.csv
|   |   |   |-- ...
|   |   |-- CLASS_1.csv
|   |   |-- CLASS_2.csv
|   |   |-- DATASET_NAME.csv
```

During training, temporary files are created:
```
root
|-- temp_train
|   |-- SLIDE_1.pt
|   |-- SLIDE_2.pt
|   |-- ...
```
These `.pt` files contain stacked data:
- Without demographics: `[features (512-dim), label (num_classes-dim)]`
- With demographics: `[features (512-dim), demographics (8-dim), label (num_classes-dim)]`

Training results are saved in:
```
root
|-- weights
|   |-- YYYYMMDD
|   |   |-- [model]_[dataset]_fold_0_1.pth
|   |   |-- fold_0_1.json
|   |   |-- ...
|   |   |-- training_summary.json
```
  

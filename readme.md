# Project Documentation

## Overview

The data for this project was downloaded from Supervisely and is organized based on the locations where the images were taken. Initially, the images were downloaded using the Supervisely format, where each image includes a JSON file containing a bitmap. This format was difficult to work with.

To simplify this, I used the [Export as masks app](https://app.supervisely.com/ecosystem/apps/export-as-masks?id=78). This app  outputs a directory with the original images and another directory containing the mask files. The pixel values in these masks correspond to the classes defined in the obj_class_to_machine_color.json file.

Since the data was initially separated into directories based on location, I used the `copy_files.py` script to combine all the images into a single directory called `dataset/`, which includes:
- `images/` : Contains all the images
- `masks/` : Contains all the corresponding masks.

##  Training

All training code is located in the `train.py` script.
I conducted several experiments using different architectures, input sizes, and augmentation techniques. The results of these experiments are stored in the `runs/` directory, which contains:
- Training logs.
- Model checkpoints.

### Data Splitting
The dataset was split using the sklearn train_test_split function with a random state of 19. The splits were as follows:
- Training set: 75%
- Validation set: 15%
- Test set: 10%

**Dataset sizes:**
- Train: 375 samples
- Validation: 67 samples
- Test: 50 samples

### Fixed Training Parameters
- **Loss Function**: Cross Entropy Loss
- **Optimizer**: AdamW
- **Scheduler**: OneCycleLR

### Input Resolutions and Augmentations
- 224 x 224
- 512 x 512
- 1024 x 1024
- 704 x 1052
- Crop and pad
- Tiling

### Architectures and Feature Extractors
- Architectures
    - UNet
    - UNet++ (UNetPlusPlus)
    - Segformer

- Feature Extractors
    - ResNet18 and ResNet34
    - Mix Transformer (mit_b0 to mit_b5)

## Evaluation

### 1. Dataset Analysis (`dataset_analysis.ipynb`)

The dataset distribution was analyzed by examining the number of pixels per pattern. A histogram was created to show the percentage of decorations in the dataset.

- **First plot**: Shows the percentage occurrence of each pattern. The green-highlighted classes are the ones where high accuracy is most desired.
- **Second plot**: Shows the pixel accuracy after resizing each image to 224 × 224. This resizing changes the distribution; patterns that previously appeared frequently may now appear even more prominently.
    - **Background pixel percentage**:
        - Before resizing: 70.52%
        - After resizing: 63.84%

<img src="Pixel distribution of all classes (not resized to a single size).png"  height="500">
<img src="Pixel distribution of all classes after resizing (224x224).png" height="500">

### 2. Semantic Segmentation evaluation (evaluate.ipynb)

Evaluation was conducted using a custom evaluation script to visualize results and calculate Intersection over Union (IoU).
- **Metric used**: Jaccard Score (2D-IoU)

The `runs/` directory contains:
- TensorBoard logs
- Model checkpoints

You can adjust the evaluate.ipynb notebook to:
1. Load different model checkpoints.
2. Apply the necessary transformations for the corresponding input resolution used while training the corresponding model.
3. Generate predictions and calculate IoU for important classes.

The results from all experiments (different architectures and input resolutions) are summarized in the [Excel sheet](resultaten_semantic_segmenation.xlsx).

### 3. Embedding space (embedding_space.ipynb)

The best-performing model was the [Segformer (mit_b2) trained at 512 × 512 resolution](runs/512x512/segformer_mit_b2/best_model_epoch.pth). This model's feature extractor (encoder) was used to extract embedding vectors, containing important feature information, for every pixel in an image.

**Embedding Workflow**
1. Extract an embedding vector for each pixel in a class.
2. Average these vectors to create a single feature vector per image per class.
3. Write all embedding vectors for the important classes into a .pkl file for reuse.

**Visualization**
- **Dimensionality Reduction**: PCA followed by T-SNE was applied to reduce embedding vectors to 2 or 3 dimensions for visualization.
- **Clusters**: Clear clusters appear, showing how well the model differentiates between decorations within the same class.
We can then reduce these vectors to less dimensions, ideally 2 or 3 so we are able to visualize these. 

An interactive plot allows hovering over points to see the image corresponding to the embedding vector.
Future work involves linking these embeddings to the [Nodegoat database](https://nodegoat.net/).

**Potential Features**

The distances between embedding vectors may reveal additional information, such as:
- Geographic location.
- Possible time periods.

### 4. Inference script (inference.py)

- You can run this script to visualize the test dataset results. 
- Just type a number between 0-49 to select an image you want to see the prediction of.  

---
Code inspired by [kaggle](https://www.kaggle.com/code/ligtfeather/semantic-segmentation-is-easy-with-pytorch/notebook#Training).
Models from [Github](https://github.com/qubvel-org/segmentation_models.pytorch).

If you have any questions: aarontenzing at gmail dot com :)
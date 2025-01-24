# Project Structure

Data directory contains the data downloaded from supervisely. The data is structured like on supervisely, based on the different places they were taken. 

I first downloaded the images using the supervisely format but here every single image has a json containing a bitmap, which is super difficult to work with. 

Then I tried downloading the annotations from supervisely using the [Export as masks app](https://app.supervisely.com/ecosystem/apps/export-as-masks?id=78). This super to use, because you just have a directory containing the original images and one directory containing the machine masks. The pixel values of these masks are linked to the classes, as written in de  **obj_class_to_machine_color.json**. 

Because the data was in different directories based on their location, I used the copy_files.py script to combine all the images to one directory called: dataset/. This consists of a images/ and masks/ directory. 

##  Training

All the code I used for training can be found in train.py. 

I tried a lot of experiments with different architectures, input sizes and augmentation techniques, where the results of can be found of in the directory runs/. They contain the training logs and model checkpoint. 

Input resolutions (augmentations):
- 224 x 224
- 512 x 512
- 1024 x 1024
- 704 x 1052
- Crop and pad
- Tiling

Architectures:
- Unet
- UnetPlusPlus
- Segformer

Different feature extractors: 
- Resnet18 and 34
- Mix Transformer (mit_b(0-5))

## Evaluation

1. Dataset

I first started looking at the dataset distribution, what the imbalance..
I created a histogram containing the percentage of decoration the dataset contains. 
This can be found in the data.ipynb notebook 



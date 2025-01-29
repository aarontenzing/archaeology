import numpy as np
import pandas as pd
import os
import json

from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # or 'Agg', 'Qt5Agg', etc.
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable

import cv2 as cv
from PIL import Image

import albumentations as A
import segmentation_models_pytorch as sm
from segmentation_models_pytorch.encoders import get_preprocessing_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

from sklearn.metrics import *
from train import ArchaeologyDataset

n_classes = 37
IMAGE_PATH = 'dataset/images/'
MASK_PATH = 'dataset/masks/'

# Read class names
with open('obj_class_to_machine_color.json', 'r') as f:
    classes = json.load(f)

# List of classes
class_labels = ["background"]
class_labels.extend(list(classes.keys()))

# Read class colors
with open('class_colors.json', 'r') as f:
    classes = json.load(f)
classes = classes['classes']

class_colors = ['#E0E0E0'] 
for i in classes:
    class_colors.append(i['color']) 


def create_df():
    """ Creates a dataframe with the image names (ids) """
    name = []
    filenames = os.listdir(IMAGE_PATH)
    filenames = sorted(filenames, key=lambda x : int(x.split('.')[0]))
    for filename in filenames:
        name.append(filename.split('.')[0])
    
    return pd.DataFrame({'id': name}, index = np.arange(0, len(name)))

def mIoU(pred_mask, mask, smooth=1e-10, n_classes=37):
    with torch.no_grad():
        pred_mask = torch.argmax(F.softmax(pred_mask, dim=1), dim=1) # softmax: om logits --> probabilities --> klasse index
        pred_mask = pred_mask.contiguous().view(-1) # flatten to 1D array
        mask = mask.contiguous().view(-1) # flatten to 1D array

        iou_per_class = []
        for class_idx in range(1, n_classes): 
            # Maak een binair masker voor de huidige klasse
            pred_class = (pred_mask == class_idx)
            gt_class = (mask == class_idx)

            if gt_class.long().sum().item() == 0: # no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                # IOU = (A ∩ B) / (A ∪ B)
                intersect = torch.logical_and(gt_class, pred_class).sum().float().item()
                union = torch.logical_or(gt_class, pred_class).sum().float().item()

                iou = (intersect + smooth) / (union + smooth)
                iou_per_class.append(iou)
        
        # Avarage over alle klasssen, buiten de klassen die niet in de GT zitten
        return np.nanmean(iou_per_class)

def predict_image_mask_miou(model, image, mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)]) # Normaliseer de afbeelding
    image = t(image)
    model.to(device) 
    mask = mask.to(device); image=image.to(device)
    
    with torch.no_grad():
        image = image.unsqueeze(0) # voeg een extra dimensie toe
        mask = mask.unsqueeze(0)   
        output = model(image)
        score = mIoU(output, mask) 
        masked = torch.argmax(output, dim=1) #visualisatie pred mask
        masked = masked.cpu().squeeze(0)
    return masked, score

def show_predictions(model, dataset, image_id):

    image, mask = dataset[image_id]

    print("Image id: ", dataset.data[image_id])

    pred_mask, score = predict_image_mask_miou(model, image, mask)

    # Determine unique classes in the mask and the predicted mask
    unique_classes = np.concatenate((np.unique(mask), np.unique(pred_mask)))
    unique_classes = np.unique(unique_classes) # remove duplicates
    unique_colors = np.array(class_colors)[unique_classes] # Kleuren van Supervisely selecteren

    cmap = ListedColormap(unique_colors)

    class_to_color = {cls: cmap(i) for i, cls in enumerate(unique_classes)} # class and rgb color

    # Prepare colors for the masks
    colored_mask = np.zeros((*mask.shape, 3))
    colored_pred_mask = np.zeros((*pred_mask.shape, 3))

    for cls in unique_classes:
        colored_mask[mask == cls] = class_to_color[cls][:3]  # Ignore alpha
        colored_pred_mask[pred_mask == cls] = class_to_color[cls][:3]  # Ignore alpha


    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,10))
    ax1.imshow(image)
    ax1.set_title('Picture')
    ax1.set_axis_off()

    ax2.imshow(colored_mask)
    ax2.set_title('Ground truth')
    ax2.set_axis_off()

    ax3.imshow(colored_pred_mask)    
    ax3.set_title('Segformer | mIoU {:.3f}'.format(score))
    ax3.set_axis_off()

    # Define a legend with class names and corresponding colors
    patches = [
        mpatches.Patch(color=class_to_color[cls], label=class_labels[cls])
        for cls in unique_classes
    ]

    ax3.legend(
        handles=patches,
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        title="Classes"
    )

    plt.tight_layout()
    plt.show()

class ArchaeologyTestDataset(Dataset):
    
    def __init__(self, img_path, mask_path, data, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.data = data
        self.transform = transform
      
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = cv.imread(self.img_path + self.data[idx] + '.jpg')
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        mask = cv.imread(self.mask_path + self.data[idx] + '.png', cv.IMREAD_GRAYSCALE)
        
        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug['image'])
            mask = aug['mask']
        else:
            img = Image.fromarray(img)
        
        mask = torch.from_numpy(mask).long()
        
        return img, mask

if __name__ == "__main__":
    df = create_df()
    
    # Create the dataset
    X_train, X_test = train_test_split(df['id'].values, test_size=0.1, random_state=19) # split the data into train and test sets
    X_train, X_val = train_test_split(X_train, test_size=0.15, random_state=19) # validate on 15% of the training data

    print('Train Size   : ', len(X_train))
    print('Val Size     : ', len(X_val))
    print('Test Size    : ', len(X_test))

    # Define normalization 
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    # Define transformations
    height = 512
    width = 512

    t_test = A.Resize(height, width, interpolation=cv.INTER_NEAREST)
      
    test_set = ArchaeologyTestDataset(IMAGE_PATH, MASK_PATH, X_test, transform=t_test)

    # Load the model
    print("Segformer")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
    model = sm.Segformer('mit_b2', encoder_weights='imagenet', classes=37, activation=None)
    model.to(device)

    model_path = 'runs/512x512/segformer_mit_b2/best_model_epoch.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully...")

    # Continuous prediction loop
    while True:
        image_id = input("Enter image index (or type 'exit' to quit): ")

        if image_id.lower() == 'exit':
            print("Exiting prediction loop.")
            break

        try:
            image_id = int(image_id)
            if image_id < 0 or image_id >= len(test_set):
                print("Invalid index! Enter a number between 0 and", len(test_set) - 1)
                continue
                
            # Call the function to visualize predictions
            show_predictions(model, test_set, image_id)

        except ValueError:
            print("Please enter a valid number.")









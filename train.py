import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import segmentation_models_pytorch as sm
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from sklearn.model_selection import train_test_split
import albumentations as A
import os
from tqdm.notebook import tqdm

import segmentation_models_pytorch as smp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
from tqdm import tqdm

n_classes = 37 # including background
IMAGE_PATH = 'tiled_dataset/images/'
MASK_PATH = 'tiled_dataset/masks/'   

def create_df():
    name = []
    filenames = os.listdir(IMAGE_PATH)
    filenames = sorted(filenames, key=lambda x : int(x.split('.')[0]))
    for filename in filenames:
        name.append(filename.split('.')[0])
    
    return pd.DataFrame({'id': name}, index = np.arange(0, len(name)))


def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1) 
        # Uitgezonderd de achtergrondklasse, bereken de pixelnauwkeurigheid
        background = (mask != 0)
        correct = (torch.eq(output, mask) & background) # compare output and mask, berekent element-wise gelijkheid 
        accuracy = float(correct.sum()) / float(background.sum()) # numel = number of elements in de tensor
    return accuracy


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

# Training function for one epoch
def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, patch=False):
    running_loss = 0.0
    running_iou = 0.0
    running_acc = 0.0
    model.train()  # Set model to training mode

    for img, mask in tqdm(train_loader, desc='Training', colour='green'):
        if patch:
            bs, n_tiles, c, h, w = img.size()
            img = img.view(-1,c, h, w)
            mask = mask.view(-1, h, w)

        # Move images and masks to the correct device
        img, mask = img.to(device), mask.to(device)

        # Forward pass
        output = model(img)
        loss = criterion(output, mask)  # Calculate the loss
    
        # Accumulate loss and metrics
        running_loss += loss.item()
        running_iou += mIoU(output, mask)  # Make sure mIoU is defined elsewhere
        running_acc += pixel_accuracy(output, mask)  # Make sure pixel_accuracy is defined elsewhere

        # Backward pass
        loss.backward()
        optimizer.step()  # Update the weights
        optimizer.zero_grad()  # Reset the gradients

        if scheduler is not None:
            scheduler.step()  # Update the learning rate

    # Return average metrics
    return running_loss / len(train_loader), running_iou / len(train_loader), running_acc / len(train_loader)

# Validation function for one epoch
def validate_one_epoch(model, val_loader, criterion, device, patch=False):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    running_iou = 0.0
    running_acc = 0.0

    with torch.no_grad():  # No need to compute gradients during validation
        for img, mask in tqdm(val_loader, desc='Validation', colour='red'):
            if patch:
                bs, n_tiles, c, h, w = img.size()
                img = img.view(-1,c, h, w)
                mask = mask.view(-1, h, w)
                
            # Move images and masks to the correct device
            img, mask = img.to(device), mask.to(device)

            # Forward pass
            output = model(img)

            # Evaluate loss and metrics
            running_iou += mIoU(output, mask)  # Make sure mIoU is defined elsewhere
            running_acc += pixel_accuracy(output, mask)  # Make sure pixel_accuracy is defined elsewhere
            loss = criterion(output, mask)
            running_loss += loss.item()

    # Return average metrics
    return running_loss / len(val_loader), running_iou / len(val_loader), running_acc / len(val_loader)

# Function to train the model
def fit(epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, patch=False, log_dir="runs", patience=20):
    torch.cuda.empty_cache()  # Clear cache to free up GPU memory
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)  # Move the model to the selected device

    writer = SummaryWriter(log_dir=log_dir)
    min_loss = np.inf  # Initialize the minimum loss to infinity
    patience_counter = 0

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')

        # Training phase
        train_loss, train_iou, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, patch=patch)

        # Log training metrics
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('IoU/Train', train_iou, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)

        # Validation phase
        val_loss, val_iou, val_acc = validate_one_epoch(model, val_loader, criterion, device, patch=patch)

        # Log validation metrics
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('IoU/Validation', val_iou, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)

        # Save the best model based on validation loss
        if val_loss < min_loss:
            print(f'Saving current best model...')
            torch.save(model.state_dict(), f'best_model_epoch.pth')
            min_loss = val_loss
            patience_counter = 0

        else:
            patience_counter += 1

        if patience_counter > patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    writer.close()
    print('Training completed!')


def predict_image_mask_miou(model, image, mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device); image=image.to(device)
    mask = mask.to(device)
    
    with torch.no_grad():
        
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        
        output = model(image)
        score = mIoU(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, score


def predict_image_mask_pixel(model, image, mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    model.eval()
    model.to(device)
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)]); image = t(image) # Normaliseer de afbeelding
    image = image.to(device); mask = mask.to(device)
    with torch.no_grad():
        image = image.unsqueeze(0) # voeg een extra dimensie toe, want het model verwacht een batch
        mask = mask.unsqueeze(0)
        output = model(image)
        acc = pixel_accuracy(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0) # verwijder de extra dimensie
    return masked, acc


def miou_score(model, test_set):
    score_iou = []
    for i in tqdm(range(len(test_set))):
        img, mask = test_set[i]
        pred_mask, score = predict_image_mask_miou(model, img, mask)
        score_iou.append(score)
    return score_iou


def pixel_acc(model, test_set):
    accuracy = []
    for i in tqdm(range(len(test_set))):
        img, mask = test_set[i]
        pred_mask, acc = predict_image_mask_pixel(model, img, mask)
        accuracy.append(acc)
    return accuracy


class ArchaeologyDataset(Dataset):
    def __init__(self, img_path, mask_path, data, mean, std, transform=None, patch=False):
        self.img_path = img_path
        self.mask_path = mask_path
        self.data = data # lijst van image/mask ids (without extensions)
        self.mean = mean 
        self.std = std
        self.transform = transform
        self.patch = patch # True als je met patches werkt, kleinere patches van de afbeeldingen maken
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = cv.imread(self.img_path + self.data[idx] + '.jpg')
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        mask = cv.imread(self.mask_path + str(self.data[idx]) + '.png', cv.IMREAD_GRAYSCALE)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = Image.fromarray(augmented['image'])
            mask = augmented['mask']

        # Normaliseren
        t = T.Compose([T.ToTensor(), T.Normalize(mean=self.mean, std=self.std)])
        img = t(img)
        mask = torch.from_numpy(mask).long() # numpy array naar tensor

        if self.patch:
            img, mask = self.tiles(img, mask) 
            
        return img, mask

    def tiles(self, img, mask):
        # Unfold the image and mask into patches
        img_patches = img.unfold(1, 512, 512).unfold(2, 768, 768) 
        img_patches  = img_patches.contiguous().view(3,-1, 512, 768) 
        img_patches = img_patches.permute(1,0,2,3)
        
        mask_patches = mask.unfold(0, 512, 512).unfold(1, 768, 768)
        mask_patches = mask_patches.contiguous().view(-1, 512, 768)
        
        return img_patches, mask_patches


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred_mask, gt_mask):
        """
        Inputs: 
        pred_mask: [Batch, Classes, Height, Width] - Predicted probabilities (logits)
        gt_mask: [Batch, Height, Width] - Ground truth masks (class indices)
        """
        # Convert predicted logits to probabilities
        pred_mask = F.softmax(pred_mask, dim=1) # on class dimension

        # One-hot encode ground truth masks
        num_classes = pred_mask.shape[1]
        gt_mask_one_hot = F.one_hot(gt_mask, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # Flatten the tensors for Dice computation --> [Batch, Classes, Height * Width]
        pred_mask = pred_mask.contiguous().view(pred_mask.size(0), pred_mask.size(1), -1)
        gt_mask_one_hot = gt_mask_one_hot.contiguous().view(gt_mask_one_hot.size(0), gt_mask_one_hot.size(1), -1)

        # Compute IOU over all pixels (dim=2)
        intersection = (pred_mask * gt_mask_one_hot).sum(dim=2)
        union = pred_mask.sum(dim=2) + gt_mask_one_hot.sum(dim=2)

        # Dice coefficient per class 
        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        # Average over all classes and batch
        dice_loss = 1 - dice.mean()
        return dice_loss


class CombinedLoss(nn.Module):
    def __init__(self, dice_loss, ce_loss, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = dice_loss
        self.ce_loss = ce_loss
        self.dice_weight = dice_weight
        self.ce_weight = 1 - dice_weight

    def forward(self, pred, target):
        dice_loss = self.dice_loss(pred, target)
        ce_loss = self.ce_loss(pred, target)
        return self.dice_weight * dice_loss + self.ce_weight * ce_loss
    

def compute_class_weights(mask_path, n_classes):
    count = np.zeros(n_classes)
    for mask_file in os.listdir(mask_path):
        mask = cv.imread(mask_path + mask_file, cv.IMREAD_GRAYSCALE)
        for cls in range(1, n_classes):
            count[cls] += np.sum(mask == cls)
    
    total_pixels = np.sum(count)
    class_weights = total_pixels / (count + 1e-6)
    return torch.tensor(class_weights, dtype=torch.float32)


if __name__ == "__main__":
    # Load the data and split it into train, validation, and test sets
    # df = create_df()
    # print('Total Images: ', len(df))
    # print(df.head())

    # X_train, X_test = train_test_split(df['id'].values, test_size=0.1, random_state=19) # split the data into train and test sets
    # X_train, X_val = train_test_split(X_train, test_size=0.15, random_state=19) # validate on 15% of the training data


    # Read splits from csv - if using (stratified random sampling) 
    # print('Reading data splits...')
    # df_train = pd.read_csv('data_splits/least_freq_label/train.csv')
    # df_val = pd.read_csv('data_splits/least_freq_label/val.csv')
    # df_test = pd.read_csv('data_splits/least_freq_label/test.csv')
    # X_train = df_train['id'].values.astype(str)
    # X_val = df_val['id'].values.astype(str)
    # X_test = df_test['id'].values.astype(str)

    # Print image ids in the test dataset
    # print('Train Size   : ', len(X_train))
    # print('Val Size     : ', len(X_val))
    # print('Test Size    : ', len(X_test))

    df_tiled = create_df()
    print('Total Images: ', len(df_tiled))
    print(df_tiled.head())

    X_train, X_val = train_test_split(df_tiled['id'].values, test_size=0.14, random_state=19)

    print('Train Size   : ', len(X_train))
    print('Val Size     : ', len(X_val))

    # Train is 76 % of the data, Test is 10 % of the data, Val is 14 % of the data

    # Mean and std
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    # Augmentations

    height, width = 512, 512   

    """ Resize and keep same ratio by padding. """
    # t_train = A.Compose([A.LongestMaxSize(max_size=max(height, width)),  # Resize longest side
    #                     A.PadIfNeeded(min_height=height, min_width=width, border_mode=cv.BORDER_CONSTANT, value=0),
    #                     A.HorizontalFlip(), A.VerticalFlip(), 
    #                     A.GridDistortion(p=0.2), A.RandomBrightnessContrast((0,0.5),(0,0.5)),
    #                     A.GaussNoise()])
    
    # t_val = A.Compose([A.LongestMaxSize(max_size=max(height, width)), 
    #                    A.PadIfNeeded(min_height=height, min_width=width, border_mode=cv.BORDER_CONSTANT, value=0)])


    """ Resize to the appropriate size """
    t_train = A.Compose([# A.Resize(height, width, interpolation=cv.INTER_NEAREST), 
                        A.HorizontalFlip(), A.VerticalFlip(), 
                        A.GridDistortion(p=0.2), A.RandomBrightnessContrast((0,0.5),(0,0.5)),
                        A.GaussNoise()])

    """ Center Crop the image and pad if image is smaller. """
    # t_train = A.Compose([A.PadIfNeeded(min_height=height, min_width=width, border_mode=cv.BORDER_CONSTANT, value=0),
    #                      A.CenterCrop(height, width), 
    #                      A.HorizontalFlip(), A.VerticalFlip(), 
    #                      A.GridDistortion(p=0.2), A.RandomBrightnessContrast((0,0.5),(0,0.5)),
    #                      A.GaussNoise()])
    
    t_val = A.Compose([A.Resize(height, width, interpolation=cv.INTER_NEAREST)])

    # Dataset
    patch = False
    train_set = ArchaeologyDataset(IMAGE_PATH, MASK_PATH, X_train, mean, std, transform=t_train, patch=patch)
    val_set = ArchaeologyDataset(IMAGE_PATH, MASK_PATH, X_val, mean, std, transform=t_val, patch=patch)

    # Dataloader
    batch_size = 16

    # Create the dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)      

    # Model architectures

    # Unet
    # print("Unet")
    # model = sm.Unet('resnet34', encoder_weights='imagenet', classes=37, activation=None)

    # print("Unet++")
    # model = sm.UnetPlusPlus('resnet18', encoder_weights='imagenet', classes=37, activation=None)

    # Segformer
    print("Segformer")
    model = sm.Segformer('mit_b2', encoder_weights='imagenet', classes=37, activation=None)

    # load checkpoint
    model.load_state_dict(torch.load('runs/512x512/segformer_mit_b2/best_model_epoch.pth'))

    # Hyperparameters
    max_lr = 1e-3
    weight_decay = 1e-4
    epochs = 1000
    log_dir = "runs/512x512/segformer_mit_b2_tiles"   
   
    # Loss function and optimizer
    # weights = compute_class_weights(MASK_PATH, n_classes)
    # print(weights)

    print("Cross Entropy Loss")
    ce_criterion = nn.CrossEntropyLoss().to(device)

    # print("Dice Loss")
    # criterion_dice = smp.losses.DiceLoss(mode='multiclass').to(device)
    #criterion = CombinedLoss(ce_dice, ce_criterion, dice_weight=0.4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))

    # Training 
    print('Training the model...')
    fit(epochs, model, train_loader, val_loader, ce_criterion, optimizer, scheduler, patch=patch, log_dir=log_dir, patience=50)
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import pickle
from collections import defaultdict
import cv2 as cv
from torchvision import transforms as T
import segmentation_models_pytorch as smp
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from tqdm import tqdm

all_embedding_vectors = defaultdict(list) # Dictionary to store embeddings for each class

# Read embeddings from pkl file
with open('embeddings_pkl/global_label_embeddings_segformer_all_feature_maps_th_0.075.pkl', 'rb') as f:
    all_embedding_vectors = pickle.load(f)

# List of important classes and their corresponding labels
important_classes = [
    'Ionic kyma decoration', 
    'Lesbian kyma decoration', 
    'Bead-and-reel (double double)', 
    'Scroll pattern (large)', 
    'Anthemion (large & capital & pulvinus)', 
    'Anthemion (small & soffit & top moulding architrave)', 
    'Acanthus leaves (solo) (capital & frieze & modillion)'
]

labels_import_classes = [6, 1, 15, 12, 2, 30, 29]

label_to_class = dict(zip(labels_import_classes, important_classes))

# Process the embeddings
# Combine embeddings into a single array and track labels
embeddings = []
labels = []
image_names = []

for label, arrays in all_embedding_vectors.items():
    for embedding, img_name in arrays:
        embeddings.append(embedding)  # Add (embedding, image name)
        image_names.append(img_name)  # Add corresponding image name
        labels.append(label)    # Add corresponding (label)

embeddings = np.vstack(embeddings)  # Convert into 2D NumPy array

class_names = [label_to_class[label] for label in labels]

# FUNCTIONS

def get_prediction(image, model, target_size=(512, 512), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """ Returns the predictions for a given image path """
    image = cv.resize(image, target_size)
    t = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])
    image = t(image)
    image = image.unsqueeze(0) # add batch dimension

    with torch.no_grad():
        output = model(image)

    output = torch.argmax(F.softmax(output, dim=1), dim=1) # softmax: om logits --> probabilities --> klasse index

    return output

def images_to_sprite(data):
    """Creates the sprite image along with any necessary padding

    Args:
      data: NxHxW[x3] tensor containing the images.

    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """
    if len(data.shape) == 3:
        data = np.tile(data[...,np.newaxis], (1,1,1,3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) - min).transpose(3,0,1,2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) / max).transpose(3,0,1,2)
    # Inverting the colors seems to look better for MNIST
    #data = 1 - data

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
            (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
            constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
            + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data

create_sprite = True

if create_sprite:
  # Load the model - needed to get the mask
  # Segformer
  model = smp.Segformer('mit_b2', encoder_weights='imagenet', classes=37, activation=None)
  model.load_state_dict(torch.load('runs/512x512/segformer_mit_b2/best_model_epoch.pth', map_location=device))

  # Torch tensors of each image
  height, width = 224, 224
  img_data = []
  for idx in tqdm(range(len(image_names)),colour='green'):
      img_path = "dataset/images/" + str(image_names[idx]) + ".jpg"
      image = cv.imread(img_path)
      image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

      output = get_prediction(image, model, target_size=(512, 512)) # size has to be 512x512 because model was trained on this
      output = output.squeeze(0).numpy()
      output = cv.resize(output, (height, width), interpolation=cv.INTER_NEAREST) # you can chose
      
      mask = (output == labels[idx])
      
      image = cv.cvtColor(image, cv.COLOR_BGR2RGBA)
      image = cv.resize(image, (height, width))
      
      # Only keep mask, make alpha channel 0 for non-masked pixels
      image[~mask] = [0, 0, 0, 0]

      img_data.append(image)

  img_data = np.array(img_data) 

  sprite = images_to_sprite(img_data)
  cv.imwrite(('sprite_test.png'), sprite) 

# Generate SPRITE IMAGE
# Copy the sprite image to "runs/segmentation_embeddings/00000/default/" directory.
"""
When generating own sprite, fix config settings:
embeddings {
  tensor_name: "default:00000"
  metadata_path: "00000/default/metadata.tsv"
  sprite {
    image_path: "00000/default/sprite.png"
    single_image_dim: 224
    single_image_dim: 224
  }
  tensor_path: "00000/default/tensors.tsv"
}
"""

# # Create a metadata 
# metadata = list(zip(labels, class_names, image_names))

# # Create a SummaryWriter instance
# writer = SummaryWriter('runs/segmentation_embeddings_th_0.05')
# writer.add_embedding(torch.from_numpy(embeddings).float(), metadata=metadata, metadata_header=["Class Label", "Class Names", "Image ID"],  global_step=0)

# writer.close()

# Print instructions
print("Run the following command in your terminal to start TensorBoard:")
print("tensorboard --logdir=runs/segmentation_embeddings")
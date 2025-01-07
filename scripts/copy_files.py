import os
import shutil
from tqdm import tqdm
import cv2

if __name__ == "__main__":
    count = 0
    print("Copying files...")
    dirs = sorted(os.listdir('data/'))
    print(dirs)
    for dir in tqdm(dirs):
        files = sorted(os.listdir('data/' + dir + '/img'), key=lambda x: int(x.split('.')[0]))
        for file in files:
            mask_path = 'data/' + dir + '/masks_machine/' + file.split('.')[0] + '.png'
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if (mask_img == 0).all():
                continue
            else:
                shutil.copy('data/' + dir + '/img/' + file, 'dataset/images/' + str(count) + '.jpg') # copy image
                shutil.copy('data/' + dir + '/masks_machine/' + file.split('.')[0] + '.png', 'dataset/masks/' + str(count) + '.png')
                count += 1
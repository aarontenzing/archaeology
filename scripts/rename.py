import os
from tqdm import tqdm

def rename_files(image_dir, json_dir, human_mask_dir, machine_mask_dir):
    # List files in each directory and sort them to maintain order
    img_list = os.listdir(image_dir)
    print(img_list)

    # Iterate through the files and rename them
    count = 0
    for name in img_list:
        # Get the name without the extension
        if name.endswith(".png"):
            name_no_ext = name.rsplit(".png", 1)[0]

        elif name.endswith(".jpeg"):
            name_no_ext = name.rsplit(".jpeg", 1)[0]

        elif name.endswith(".JPG"):
            name_no_ext = name.rsplit(".JPG", 1)[0]

        else:
            name_no_ext = name.rsplit(".jpg", 1)[0]

        os.rename(os.path.join(image_dir, str(name)), os.path.join(image_dir, str(count) + ".jpg"))
        os.rename(os.path.join(json_dir, str(name)) + ".json", os.path.join(json_dir, str(count) + ".json"))
        os.rename(os.path.join(human_mask_dir, str(name_no_ext) + ".png"), os.path.join(human_mask_dir, str(count) + ".png"))
        os.rename(os.path.join(machine_mask_dir, str(name_no_ext) + ".png"), os.path.join(machine_mask_dir, str(count) + ".png"))
        count += 1

# Rename files in each directory
for dir in tqdm(os.listdir('data/')):
    rename_files("data/" + dir + "/img", "data/" + dir + "/ann", "data/" + dir + "/masks_human", "data/" + dir + "/masks_machine")
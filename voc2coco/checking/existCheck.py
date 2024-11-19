###################### All code referenced from datnguyen-tien204 #########
####################  Github: https://github.com/datnguyen-tien204  #########
### This code is used to check if the images in the COCO JSON file exist in the specified directory. #######


import json
import os

def check_exist(json_path, images_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)
    filtered_images = []
    filtered_annotations = []
    existing_image_ids = set()
    for image_info in data['images']:
        file_name = image_info['file_name']
        image_path = os.path.join(images_dir, file_name)
        if os.path.exists(image_path):
            filtered_images.append(image_info)
            existing_image_ids.add(image_info['id'])

    for annotation in data['annotations']:
        if annotation['image_id'] in existing_image_ids:
            filtered_annotations.append(annotation)

    data['images'] = filtered_images
    data['annotations'] = filtered_annotations
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)



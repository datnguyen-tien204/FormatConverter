"""
Alright reserved belonging to datnguyen-tien204
Profile: http://tien-datnguyen-blogs.me/
Convert from COCO -> YOLO

"""

import json
from PIL import Image
import os
from rich import print as rprint
from rich.text import Text


def checkingSize(annotation_files, images_dir,extensions=".jpg"):
    with open(annotation_files, 'r') as f:
        data = json.load(f)
    images = {image['file_name']: image for image in data['images']}
    for file_name, image_info in images.items():
        if not file_name.lower().endswith(extensions):
            continue
        image_path = os.path.join(images_dir, file_name)
        if os.path.exists(image_path):
            with Image.open(image_path) as img:
                actual_width, actual_height = img.size
                if image_info['width'] != actual_width or image_info['height'] != actual_height:
                    rprint(Text(
                        f"Edited size for {file_name}: ({image_info['width']}, {image_info['height']}) -> ({actual_width}, {actual_height})",
                        style="bold red"))
                    image_info['width'] = actual_width
                    image_info['height'] = actual_height

    with open(annotation_files, 'w') as f:
        json.dump(data, f)
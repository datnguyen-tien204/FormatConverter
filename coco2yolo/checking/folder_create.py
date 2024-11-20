"""
Alright reserved belonging to datnguyen-tien204
Profile: http://tien-datnguyen-blogs.me/
Convert from COCO -> YOLO

"""

import os
import shutil
import logging
from rich.logging import RichHandler
from rich import print as rprint

def create_folder_structure(output_folder):
    annotations_dir = os.path.join(output_folder, "labels")
    jpeg_images_dir = os.path.join(output_folder, "images")
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(jpeg_images_dir, exist_ok=True)
    return annotations_dir, jpeg_images_dir

def move_files_and_generate_yolo_v1(images_folder, labels_folder, output_folder,extensions=".jpg"):
    annotations_dir, jpeg_images_dir = create_folder_structure(output_folder)
    file_names = []
    for file_name in os.listdir(images_folder):
        if file_name.endswith(extensions):
            src_path = os.path.join(images_folder, file_name)
            dest_path = os.path.join(jpeg_images_dir, file_name)
            shutil.move(src_path, dest_path)
            file_names.append(os.path.splitext(file_name)[0])

    for file_name in os.listdir(labels_folder):
        if file_name.endswith(".txt"):
            src_path = os.path.join(labels_folder, file_name)
            dest_path = os.path.join(annotations_dir, file_name)
            shutil.move(src_path, dest_path)
            file_names.append(os.path.splitext(file_name)[0])

    return annotations_dir, jpeg_images_dir
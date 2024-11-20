###################### All code referenced from datnguyen-tien204 #########
####################  Github: https://github.com/datnguyen-tien204  #########
############ Profile: http://tien-datnguyen-blogs.me/
### This code is used to check if the images in the YOLO images and YOLO labels match exist in the specified directory. #######

import os
import shutil
import logging
from rich.logging import RichHandler
from rich import print as rprint

def create_folder_structure(output_folder):
    annotations_dir = os.path.join(output_folder, "images")
    jpeg_images_dir = os.path.join(output_folder, "labels")
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(jpeg_images_dir, exist_ok=True)
    return annotations_dir, jpeg_images_dir

def rich_logger(num_first,num_last, notification):
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )
    log = logging.getLogger("rich")
    formatted_numbers = f"[{num_first}|{num_last}]"
    log.info(f"{formatted_numbers} - {notification}!")

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

def move_files_and_generate_type2(images_folder, labels_folder, sets_folder, output_folder, set_type="train", extension=".jpg"):
    set_file_path = os.path.join(sets_folder, f"{set_type}.txt")
    if not os.path.exists(set_file_path):
        raise FileNotFoundError(f"[Error] File not found: {set_file_path}")

    output_images_dir = os.path.join(output_folder, set_type, "images")
    output_labels_dir = os.path.join(output_folder, set_type, "labels")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    with open(set_file_path, "r") as file:
        image_ids = [line.strip() for line in file.readlines()]

    for image_id in image_ids:
        image_file = os.path.join(images_folder, f"{image_id}{extension}")
        label_file = os.path.join(labels_folder, f"{image_id}.txt")

        if not os.path.exists(image_file):
            print(f"[Warning] Missing image file: {image_file}")
            continue
        if not os.path.exists(label_file):
            print(f"[Warning] Missing label file: {label_file}")
            continue

        shutil.copy(image_file, os.path.join(output_images_dir, f"{image_id}{extension}"))
        shutil.copy(label_file, os.path.join(output_labels_dir, f"{image_id}.txt"))
    return output_images_dir, output_labels_dir

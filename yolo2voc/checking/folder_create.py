import os
import shutil
import logging
from rich.logging import RichHandler
from rich import print as rprint

def create_folder_structure(output_folder):
    annotations_dir = os.path.join(output_folder, "Annotations")
    jpeg_images_dir = os.path.join(output_folder, "JPEGImages")
    image_sets_dir = os.path.join(output_folder, "ImageSets/Main")
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(jpeg_images_dir, exist_ok=True)
    os.makedirs(image_sets_dir, exist_ok=True)
    return annotations_dir, jpeg_images_dir, image_sets_dir

def rich_logger(num_first,num_last, notification):
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )
    log = logging.getLogger("rich")
    formatted_numbers = f"[{num_first}|{num_last}]"
    log.info(f"{formatted_numbers} - {notification}!")

def move_files_and_generate_trainval(images_folder, labels_folder, output_folder,extensions=".jpg"):
    annotations_dir, jpeg_images_dir, image_sets_dir = create_folder_structure(output_folder)
    trainval_path = os.path.join(image_sets_dir, "trainval.txt")
    file_names = []
    for file_name in os.listdir(images_folder):
        if file_name.endswith(extensions):
            src_path = os.path.join(images_folder, file_name)
            dest_path = os.path.join(jpeg_images_dir, file_name)
            shutil.move(src_path, dest_path)
            file_names.append(os.path.splitext(file_name)[0])

    for file_name in os.listdir(labels_folder):
        if file_name.endswith(".xml"):
            src_path = os.path.join(labels_folder, file_name)
            dest_path = os.path.join(annotations_dir, file_name)
            shutil.move(src_path, dest_path)
            file_names.append(os.path.splitext(file_name)[0])

    rich_logger(4, 6, "Starting to generate trainval.txt file")
    file_names = sorted(set(file_names))
    with open(trainval_path, "w") as f:
        for name in file_names:
            f.write(name + "\n")
    rich_logger(4, 6, "Finished generating trainval.txt file")

    return annotations_dir, jpeg_images_dir, trainval_path

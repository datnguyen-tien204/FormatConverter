###################### All code referenced from datnguyen-tien204 #########
####################  Github: https://github.com/datnguyen-tien204  #########
### This code is used to check if the images in the COCO JSON file exist in the specified directory. #######


import json
import os
import logging

from rich.logging import RichHandler
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

def rich_logger(num_first, num_last, notification):
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )
    log = logging.getLogger("rich")
    formatted_numbers = f"[{num_first}|{num_last}]"
    log.info(f"{formatted_numbers} - {notification}")

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


def check_and_handle_files(images_folder, labels_folder,extensions=".jpg"):
    images = {os.path.splitext(f)[0] for f in os.listdir(images_folder) if f.endswith(extensions)}
    labels = {os.path.splitext(f)[0] for f in os.listdir(labels_folder) if f.endswith(".txt")}
    extra_images = images - labels
    extra_labels = labels - images

    if extra_images or extra_labels:
        rich_logger(1, 6, f"Unmatch file detected!")
        if extra_images:
            rich_logger(1,6,f"Image no labels :{', '.join(extra_images)}")
        if extra_labels:
            rich_logger(1,6,f"Label no images :{', '.join(extra_labels)}")

        action = Prompt.ask(
            "[bold blue]What do you want to do?[/bold blue] (delete/skip)",
            choices=["del", "skip"]
        )
        if action == "del":
            for img in extra_images:
                os.remove(os.path.join(images_folder, f"{img}.{extensions}"))
                print(f"[green]Deleted:[/green] {img}.{extensions}")
            for lbl in extra_labels:
                os.remove(os.path.join(labels_folder, f"{lbl}.txt"))
                print(f"[green]Deleted:[/green] {lbl}.txt")
            rich_logger(1,6,"All extra files deleted!")
        else:
            rich_logger(1,6,"Skipping file deletion.")
        return True
    else:
        rich_logger(1, 6, f"All files are matched!")
        return False



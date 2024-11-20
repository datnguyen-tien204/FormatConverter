###################### All code referenced from datnguyen-tien204 #########
####################  Github: https://github.com/datnguyen-tien204  #########
############ Profile: http://tien-datnguyen-blogs.me/
### This code is used to check if the images in the YOLO images and YOLO labels match exist in the specified directory. #######
import os
from rich import print
from rich.prompt import Prompt
from .folder_create import rich_logger

def check_and_handle_files(images_folder, labels_folder,extensions=".jpg"):
    images = {os.path.splitext(f)[0] for f in os.listdir(images_folder) if f.endswith(extensions)}
    labels = {os.path.splitext(f)[0] for f in os.listdir(labels_folder) if f.endswith(".xml")}
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
                os.remove(os.path.join(labels_folder, f"{lbl}.xml"))
                print(f"[green]Deleted:[/green] {lbl}.xml")
            rich_logger(1,6,"All extra files deleted!")
        else:
            rich_logger(1,6,"Skipping file deletion.")
        return True
    else:
        rich_logger(1, 6, f"All files are matched!")
        return False




###################### All code referenced from datnguyen-tien204 #########
####################  Github: https://github.com/datnguyen-tien204  #########
############ Profile: http://tien-datnguyen-blogs.me/
### This code is used to check if the images in the YOLO images and YOLO labels match exist in the specified directory. #######
import os
from rich import print
from rich.prompt import Prompt

def check_and_handle_files(images_folder, labels_folder,extensions=".jpg"):
    images = {os.path.splitext(f)[0] for f in os.listdir(images_folder) if f.endswith(extensions)}
    labels = {os.path.splitext(f)[0] for f in os.listdir(labels_folder) if f.endswith(".txt")}
    extra_images = images - labels
    extra_labels = labels - images

    if extra_images or extra_labels:
        print("[bold yellow] Unmatch file detected![/bold yellow]")
        if extra_images:
            print(f"[red]Image no labels :[/red] {', '.join(extra_images)}")
        if extra_labels:
            print(f"[red]Label no images:[/red] {', '.join(extra_labels)}")

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
            print("[bold green]All extra files deleted![/bold green]")
        else:
            print("[bold cyan]Skipping file deletion.[/bold cyan]")
        return True
    else:
        print("[bold green]All files are matched![/bold green]")
        return False




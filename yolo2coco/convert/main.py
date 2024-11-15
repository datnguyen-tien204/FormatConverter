###################### All code referenced from datnguyen-tien204 #########
####################  Github: https://github.com/datnguyen-tien204  #########
### This code is used to check if the images in the COCO JSON file exist in the specified directory. #######


from pathlib import Path
from create_annotations import (
    create_image_annotation,
    create_annotation_from_yolo_format,
    create_annotation_from_yolo_results_format,
    coco_format,
)
import cv2
import argparse
import json
import numpy as np
import imagesize
import os
import shutil
import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.markdown import Markdown
from rich import print as rprint
from rich.text import Text
from rich.progress import Progress
from checking.existCheck import check_exist
from checking.checkSize import checkingSize

YOLO_DARKNET_SUB_DIR = "YOLO_darknet"

classes = [
    "car",
    "person"
]

def get_images_info_and_annotations(opt):
    path = Path(opt.path)
    annotations = []
    images_annotations = []
    if path.is_dir():
        file_paths = sorted(path.rglob("*.jpg"))
        file_paths += sorted(path.rglob("*.jpeg"))
        file_paths += sorted(path.rglob("*.png"))
    else:
        with open(path, "r") as fp:
            read_lines = fp.readlines()
        file_paths = [Path(line.replace("\n", "")) for line in read_lines]

    image_id = 0
    annotation_id = 1  # In COCO dataset format, you must start annotation id with '1'
    with Progress() as progress:
        task = progress.add_task("[cyan]Processing Images...", total=len(file_paths))

        for file_path in file_paths:
            progress.update(task, advance=1)

            # Build image annotation, known the image's width and height
            w, h = imagesize.get(str(file_path))
            image_annotation = create_image_annotation(
                file_path=file_path, width=w, height=h, image_id=image_id
            )
            images_annotations.append(image_annotation)

            label_file_name = f"{file_path.stem}.txt"
            if opt.yolo_subdir:
                annotations_path = file_path.parent / YOLO_DARKNET_SUB_DIR / label_file_name
            else:
                annotations_path = file_path.parent / label_file_name

            if annotations_path.exists(): # The image may not have any applicable annotation txt file.
                with open(str(annotations_path), "r") as label_file:
                    label_read_line = label_file.readlines()

                # yolo format - (class_id, x_center, y_center, width, height)
                # coco format - (annotation_id, x_upper_left, y_upper_left, width, height)
                for line1 in label_read_line:
                    label_line = line1
                    category_id = (
                        int(label_line.split()[0]) + 1
                    )  # you start with annotation id with '1'
                    x_center = float(label_line.split()[1])
                    y_center = float(label_line.split()[2])
                    width = float(label_line.split()[3])
                    height = float(label_line.split()[4])

                    float_x_center = w * x_center
                    float_y_center = h * y_center
                    float_width = w * width
                    float_height = h * height

                    min_x = int(float_x_center - float_width / 2)
                    min_y = int(float_y_center - float_height / 2)
                    width = int(float_width)
                    height = int(float_height)

                    if opt.results == True: #yolo_result to Coco_result (saves confidence)
                        conf = float(label_line.split()[5])
                        annotation = create_annotation_from_yolo_results_format(
                            min_x,
                            min_y,
                            width,
                            height,
                            image_id,
                            category_id,
                            conf
                        )

                    else:
                        annotation = create_annotation_from_yolo_format(
                            min_x,
                            min_y,
                            width,
                            height,
                            image_id,
                            category_id,
                            annotation_id,
                            segmentation=opt.box2seg,
                        )
                    annotations.append(annotation)
                    annotation_id += 1

            image_id += 1  # if you finished annotation work, updates the image id.

        return images_annotations, annotations


def get_args():
    parser = argparse.ArgumentParser("Yolo format annotations to COCO dataset format")

    parser.add_argument(
        "-p",
        "--path_project_folder",
        type=str,
        help="Absolute path project folder. If use this not need to use -p_image and -p_labels",
    )

    parser.add_argument(
        "-p_image",
        "--path_image_folder",
        type=str,
        help="Absolute path for image folder. If use this not need to use -p",
    )

    parser.add_argument(
        "-p_labels",
        "--path_label_folder",
        type=str,
        help="Absolute path labels folder. If use this not need to use -p",
    )

    parser.add_argument("--train", action="store_true",
                        help="Set output to train2017.json and move images to train2017")

    parser.add_argument("--val", action="store_true",
                        help="Set output to train2017.json and move images to val2017")

    parser.add_argument("--test", action="store_true",
                        help="Set output to train2017.json and move images to test2017")

    parser.add_argument(
        "--output",
        default="train_coco.json",
        type=str,
        help="Name the output json file",
    )
    parser.add_argument(
        "--yolo-subdir",
        action="store_true",
        help="Annotations are stored in a subdir not side by side with images.",
    )
    parser.add_argument(
        "--box2seg",
        action="store_true",
        help="Coco segmentation will be populated with a polygon "
        "that matches replicates the bounding box data.",
    )
    parser.add_argument(
    "--results",
    action="store_true",
    help="Saves confidence scores of the results"
    "yolo results to Coco results.",
    )
    args = parser.parse_args()
    return args

def rich_logger(num_first,num_last, notification):
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )
    log = logging.getLogger("rich")
    formatted_numbers = f"[{num_first}|{num_last}]"
    log.info(f"{formatted_numbers} - {notification}!")

def move_files(imgs_folder, labels_folder, output_folder):
    rich_logger(1,8,"Merging files folders")
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(imgs_folder):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            src_path = os.path.join(imgs_folder, filename)
            dst_path = os.path.join(output_folder, filename)
            if os.path.isfile(src_path):
                shutil.move(src_path, dst_path)
        else:
            print("Skipping file: ", filename)

    for filename in os.listdir(labels_folder):
        if filename.endswith(".txt"):
            src_path = os.path.join(labels_folder, filename)
            dst_path = os.path.join(output_folder, filename)
            if os.path.isfile(src_path):
                shutil.move(src_path, dst_path)
        else:
            print("Skipping file: ", filename)
    rich_logger(1,8,"Merging files folders successfully!")
from rich.panel import Panel
def head_rich_logger():
    MARKDOWN = """
    - `-p`, `--path_project_folder`: **(str)** : Absolute path project folder. If use this not need to use -p_image and -p_labels
    - `-p_image`, `--path_image_folder`: **(str)**   : Absolute path for the folder containing images. If use this not need to use -p
    - `-p_labels`, `--path_label_folder`: **(str)**  : Absolute path for the folder containing YOLO format labels. If use this not need to use -p
    - `--train`: **(bool)**  : If specified, output JSON file is named `train2017.json` and images are moved to `train2017` directory.
    - `--val`: **(bool)**    : If specified, output JSON file is named `val2017.json` and images are moved to `val2017` directory.
    - `--test`: **(bool)**   : If specified, output JSON file is named `test2017.json` and images are moved to `test2017` directory.
    - `--debug`: **(bool)**  : If specified, bounding boxes are visualized and annotation information is printed for debugging purposes.
    - `--output`: **(str)**  : Name of the output JSON file. Default is `train_coco.json`.
    - `--yolo-subdir`: **(bool)** : Indicates that annotations are stored in a subdirectory, not in the same directory as images.
    - `--box2seg`: **(bool)**  : Populates the COCO segmentation field with a polygon that replicates the bounding box data.
    - `--results`: **(bool)**  : Saves confidence scores from YOLO results to the COCO results format.
    """
    console = Console()
    md = Markdown(MARKDOWN, style="white",code_theme="manni")
    panel = Panel(md, title="YOLO to COCO Converter Parameters", expand=False, style="on grey93", border_style="blue")
    console.print(panel)
def trainvaltest_activtation(input_images,type):
    if type=="train":
        train_folder = "train2017"
        os.makedirs(train_folder, exist_ok=True)

        for filename in os.listdir(input_images):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                src_path = os.path.join(input_images, filename)
                dst_path = os.path.join(train_folder, filename)
                shutil.move(src_path, dst_path)

        for filename in os.listdir(input_images):
            file_path = os.path.join(input_images, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    if type=="val":
        val_folder = "val2017"
        os.makedirs(val_folder, exist_ok=True)

        for filename in os.listdir(input_images):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                src_path = os.path.join(input_images, filename)
                dst_path = os.path.join(val_folder, filename)
                shutil.move(src_path, dst_path)

        for filename in os.listdir(input_images):
            file_path = os.path.join(input_images, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    if type=="test":
        test_folder = "test2017"
        os.makedirs(test_folder, exist_ok=True)

        for filename in os.listdir(input_images):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                src_path = os.path.join(input_images, filename)
                dst_path = os.path.join(test_folder, filename)
                shutil.move(src_path, dst_path)

        for filename in os.listdir(input_images):
            file_path = os.path.join(input_images, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)


def main(opt):
    head_rich_logger()
    text = Text("Starting conversion!", style="bold blue")
    rprint(text)

    if opt.path_project_folder:
        opt.path_project_folder = Path(opt.path_project_folder)

        if opt.train:
            opt.path_image_folder = opt.path_project_folder / "train/images"
            opt.path_label_folder = opt.path_project_folder / "train/labels"
        elif opt.val:
            opt.path_image_folder = opt.path_project_folder / "val/images"
            opt.path_label_folder = opt.path_project_folder / "val/labels"
        elif opt.test:
            opt.path_image_folder = opt.path_project_folder / "test/images"
            opt.path_label_folder = opt.path_project_folder / "test/labels"
        else:
            raise ValueError("Please specify either --train, --val, or --test.")

    # Merging folders
    move_files(opt.path_image_folder, opt.path_label_folder, "temp_folder")
    opt.path = "temp_folder"


    if opt.train is True:
        rich_logger(2,8,"Creating train2017")

        os.makedirs("annotations", exist_ok=True)
        if os.path.exists("train2017"):
            shutil.rmtree("train2017")

        output_name = "instace_train2017.json"
        output_path = "annotations/" + output_name

        rich_logger(2,8,"Create successfully: " + output_path)
        rich_logger(3,8,"Creating annotations")

        (coco_format["images"],coco_format["annotations"],) = get_images_info_and_annotations(opt)
        for index, label in enumerate(classes):
            categories = {
                "supercategory": "Defect",
                "id": index + 1,  # ID starts with '1' .
                "name": label,
            }
            coco_format["categories"].append(categories)

        if opt.results == True:
            dict_list = []
            for l in coco_format["annotations"]:
                dict_list.append(l[0])
            with open(output_path, "w") as outfile:
                str = json.dump(dict_list, outfile, indent=4)

        else:
            with open(output_path, "w") as outfile:
                json.dump(coco_format, outfile, indent=4)

        rich_logger(3,8,"Annotations created successfully!")
        rich_logger(4,8,"Moving files to train2017")
        trainvaltest_activtation("temp_folder", "train")
        rich_logger(4,8,"Files moved successfully!")

        ### Checking if the images in the COCO JSON file exist in the specified directory. ###
        rich_logger(5,8,"Checking if the images in the COCO JSON file exist in the specified directory.")
        check_exist(output_path, "train2017")
        rich_logger(5,8,"Checking completed successfully!")

        ### Checking if the size of the images in the COCO JSON file matches the actual size of the images. ###
        rich_logger(6,8,"Checking if the size of the images in the COCO JSON file matches the actual size of the images.")
        checkingSize(output_path, "train2017")
        rich_logger(6,8,"Checking completed successfully!")

    if opt.val is True:

        rich_logger(2,8,"Creating val2017")

        os.makedirs("annotations", exist_ok=True)
        if os.path.exists("val2017"):
            shutil.rmtree("val2017")

        output_name = "instace_val2017.json"
        output_path = "annotations/" + output_name

        rich_logger(2,8,"Create successfully: " +output_path)
        rich_logger(3,8,"Creating annotations")

        (coco_format["images"], coco_format["annotations"],) = get_images_info_and_annotations(opt)
        for index, label in enumerate(classes):
            categories = {
                "supercategory": "Defect",
                "id": index + 1,  # ID starts with '1' .
                "name": label,
            }
            coco_format["categories"].append(categories)

        if opt.results == True:
            dict_list = []
            for l in coco_format["annotations"]:
                dict_list.append(l[0])
            with open(output_path, "w") as outfile:
                str = json.dump(dict_list, outfile, indent=4)

        else:
            with open(output_path, "w") as outfile:
                json.dump(coco_format, outfile, indent=4)

        rich_logger(3,8,"Annotations created successfully!")
        rich_logger(4,8,"Moving files to val2017")
        trainvaltest_activtation("temp_folder", "val")
        rich_logger(4,8,"Files moved successfully!")

        ### Checking if the images in the COCO JSON file exist in the specified directory. ###
        rich_logger(5, 8, "Checking if the images in the COCO JSON file exist in the specified directory.")
        check_exist(output_path, "val2017")
        rich_logger(5, 8, "Checking completed successfully!")

        ### Checking if the size of the images in the COCO JSON file matches the actual size of the images. ###
        rich_logger(6, 8,
                    "Checking if the size of the images in the COCO JSON file matches the actual size of the images.")
        checkingSize(output_path, "val2017")
        rich_logger(6, 8, "Checking completed successfully!")


    if opt.test is True:
        rich_logger(2,8,"Creating test2017")
        os.makedirs("annotations", exist_ok=True)
        if os.path.exists("test2017"):
            shutil.rmtree("test2017")

        output_name = "instace_test2017.json"
        output_path = "annotations/" + output_name

        rich_logger(2,8,"Create successfully: " + output_path)
        rich_logger(3,8,"Creating annotations")

        (coco_format["images"], coco_format["annotations"],) = get_images_info_and_annotations(opt)
        for index, label in enumerate(classes):
            categories = {
                "supercategory": "Defect",
                "id": index + 1,  # ID starts with '1' .
                "name": label,
            }
            coco_format["categories"].append(categories)

        if opt.results == True:
            dict_list = []
            for l in coco_format["annotations"]:
                dict_list.append(l[0])
            with open(output_path, "w") as outfile:
                str = json.dump(dict_list, outfile, indent=4)

        else:
            with open(output_path, "w") as outfile:
                json.dump(coco_format, outfile, indent=4)

        rich_logger(3,8,"Annotations created successfully!")
        rich_logger(4,8,"Moving files to test2017")
        trainvaltest_activtation("temp_folder", "test")
        rich_logger(4,8,"Files moved successfully!")

        ### Checking if the images in the COCO JSON file exist in the specified directory. ###
        rich_logger(5, 8, "Checking if the images in the COCO JSON file exist in the specified directory.")
        check_exist(output_path, "test2017")
        rich_logger(5, 8, "Checking completed successfully!")

        ### Checking if the size of the images in the COCO JSON file matches the actual size of the images. ###
        rich_logger(6, 8,
                    "Checking if the size of the images in the COCO JSON file matches the actual size of the images.")
        checkingSize(output_path, "test2017")
        rich_logger(6, 8, "Checking completed successfully!")

    text = Text("Finished!", style="bold blue")
    rprint(text)


if __name__ == "__main__":
    options = get_args()
    main(options)

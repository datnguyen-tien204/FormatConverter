"""Label Convert

Convert from YOLO -> VOC | VOC -> YOLO

"""
import argparse
import multiprocessing
import os
from xml.etree import ElementTree

from PIL import Image
from pascal_voc_writer import Writer

import config
from checking.checkSize import checkingSize
from checking.existCheck import check_exist
import logging
from rich.logging import RichHandler


def yolo2voc(txt_file: str) -> None:
    """Convert YOLO to VOC
    Args:
        txt_file: str
    """
    w, h = Image.open(os.path.join(config.IMAGE_DIR, f"{txt_file[:-4]}.jpg")).size
    writer = Writer(f"{txt_file[:-4]}.xml", w, h)
    with open(os.path.join(config.LABEL_DIR, txt_file)) as f:
        for line in f.readlines():
            label, x_center, y_center, width, height = line.rstrip().split(" ")
            x_min = int(w * max(float(x_center) - float(width) / 2, 0))
            x_max = int(w * min(float(x_center) + float(width) / 2, 1))
            y_min = int(h * max(float(y_center) - float(height) / 2, 0))
            y_max = int(h * min(float(y_center) + float(height) / 2, 1))
            writer.addObject(config.names[int(label)], x_min, y_min, x_max, y_max)
    writer.save(os.path.join(config.XML_DIR, f"{txt_file[:-4]}.xml"))


def voc2yolo(xml_file: str) -> None:
    """Convert VOC to YOLO
    Args:
        xml_file: str
    """
    with open(f"{config.XML_DIR}/{xml_file}") as in_file:
        tree = ElementTree.parse(in_file)
        size = tree.getroot().find("size")
        height, width = map(int, [size.find("height").text, size.find("width").text])

    class_exists = False
    for obj in tree.findall("object"):
        name = obj.find("name").text
        if name in config.names:
            class_exists = True

    if class_exists:
        with open(f"{config.LABEL_DIR}/{xml_file[:-4]}.txt", "w") as out_file:
            for obj in tree.findall("object"):
                difficult = obj.find("difficult").text
                if int(difficult) == 1:
                    continue
                xml_box = obj.find("bndbox")

                x_min = float(xml_box.find("xmin").text)
                y_min = float(xml_box.find("ymin").text)

                x_max = float(xml_box.find("xmax").text)
                y_max = float(xml_box.find("ymax").text)

                # according to darknet annotation
                box_x_center = (x_min + x_max) / 2.0 - 1
                box_y_center = (y_min + y_max) / 2.0 - 1

                box_w = x_max - x_min
                box_h = y_max - y_min

                box_x = box_x_center * 1.0 / width
                box_w = box_w * 1.0 / width

                box_y = box_y_center * 1.0 / height
                box_h = box_h * 1.0 / height

                b = [box_x, box_y, box_w, box_h]

                cls_id = config.names.index(obj.find("name").text)
                out_file.write(str(cls_id) + " " + " ".join([str(f"{i:.6f}") for i in b]) + "\n")


def rich_logger(num_first,num_last, notification):
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )
    log = logging.getLogger("rich")
    formatted_numbers = f"[{num_first}|{num_last}]"
    log.info(f"{formatted_numbers} - {notification}!")



def voc2yolo_a(xml_file: str) -> None:
    """Convert VOC to YOLO with absolute cordinates
    Args:
        xml_file: str
    """
    with open(f"{config.XML_DIR}/{xml_file}") as in_file:
        tree = ElementTree.parse(in_file)

    class_exists = False
    for obj in tree.findall("object"):
        name = obj.find("name").text
        if name in config.names:
            class_exists = True

    if class_exists:
        with open(f"{config.LABEL_DIR}/{xml_file[:-4]}.txt", "w") as out_file:
            for obj in tree.findall("object"):
                difficult = obj.find("difficult").text
                if int(difficult) == 1:
                    continue
                xml_box = obj.find("bndbox")
                x_min = round(float(xml_box.find("xmin").text))
                y_min = round(float(xml_box.find("ymin").text))

                x_max = round(float(xml_box.find("xmax").text))
                y_max = round(float(xml_box.find("ymax").text))

                b = [x_min, y_min, x_max, y_max]
                cls_id = config.names.index(obj.find("name").text)
                out_file.write(str(cls_id) + " " + " ".join([str(f"{i}") for i in b]) + "\n")


from rich.console import Console
from rich.markdown import Markdown
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


from rich.text import Text
import shutil


def trainvaltest_activtation(input_images,output_folder,type):
    if type=="train":
        train_folder = "train2017"
        try:
            train_folder = os.path.join(output_folder,train_folder)
            os.makedirs(train_folder, exist_ok=True)
        except:
            train_folder = os.path.join(output_folder, train_folder)
            shutil.rmtree(train_folder)
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
        try:
            val_folder = os.path.join(output_folder, val_folder)
            os.makedirs(val_folder, exist_ok=True)
        except:
            val_folder = os.path.join(output_folder, val_folder)
            shutil.rmtree(val_folder)
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
        try:
            test_folder = os.path.join(output_folder, test_folder)
            os.makedirs(test_folder, exist_ok=True)
        except:
            test_folder = os.path.join(output_folder, test_folder)
            shutil.rmtree(test_folder)
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--yolo2voc", action="store_true", help="YOLO to VOC")
    parser.add_argument("--voc2yolo", action="store_true", help="VOC to YOLO")
    parser.add_argument("--voc2yolo_a", action="store_true", help="VOC to YOLO absolute")
    args = parser.parse_args()

    head_rich_logger()
    text=Text("Starting conversion",style="bold blue")
    rich_logger(1, 6, "Finding all files *.json in folder")
    txt_files = [
        name for name in os.listdir(config.LABEL_DIR) if name.endswith(".txt")
    ]
    rich_logger(1,6,"Finding all file successfully")
    rich_logger(2,6, "Starting conversion!")

    with multiprocessing.Pool(os.cpu_count()) as pool:
        pool.map(yolo2voc, txt_files)
    pool.join()

    if args.voc2yolo_a:
        xml_files = [
            name for name in os.listdir(config.XML_DIR) if name.endswith(".xml")
        ]
        with multiprocessing.Pool(os.cpu_count()) as pool:
            pool.map(voc2yolo_a, xml_files)
        pool.close()

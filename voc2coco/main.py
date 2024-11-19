#!/usr/bin/python
import os
import json
import xml.etree.ElementTree as ET
import glob

from rich.logging import RichHandler
from rich.console import Console
from rich.markdown import Markdown
from rich import print as rprint
from rich.text import Text

START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = None
from rich.panel import Panel
import argparse
import logging
from checking import existCheck, checkSize
import shutil

# If necessary, pre-define category and its id
#  PRE_DEFINE_CATEGORIES = {"aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
#  "bottle":5, "bus": 6, "car": 7, "cat": 8, "chair": 9,
#  "cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
#  "motorbike": 14, "person": 15, "pottedplant": 16,
#  "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20}


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise ValueError("Can not find %s in %s." % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise ValueError(
            "The size of %s is supposed to be %d, but is %d."
            % (name, length, len(vars))
        )
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    try:
        filename = filename.replace("\\", "/")
        filename = os.path.splitext(os.path.basename(filename))[0]
        return int(filename)
    except:
        raise ValueError("Filename %s is supposed to be an integer." % (filename))


def get_categories(xml_files):
    """Generate category name to id mapping from a list of xml files.

    Arguments:
        xml_files {list} -- A list of xml file paths.

    Returns:
        dict -- category name to id mapping.
    """
    classes_names = []
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            classes_names.append(member[0].text)
    classes_names = list(set(classes_names))
    classes_names.sort()
    return {name: i for i, name in enumerate(classes_names)}


def convert(xml_files, json_file):
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    if PRE_DEFINE_CATEGORIES is not None:
        categories = PRE_DEFINE_CATEGORIES
    else:
        categories = get_categories(xml_files)
    bnd_id = START_BOUNDING_BOX_ID
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        path = get(root, "path")
        if len(path) == 1:
            filename = os.path.basename(path[0].text)
        elif len(path) == 0:
            filename = get_and_check(root, "filename", 1).text
        else:
            raise ValueError("%d paths found in %s" % (len(path), xml_file))
        ## The filename must be a number
        image_id = get_filename_as_int(filename)
        size = get_and_check(root, "size", 1)
        width = int(get_and_check(size, "width", 1).text)
        height = int(get_and_check(size, "height", 1).text)
        image = {
            "file_name": filename,
            "height": height,
            "width": width,
            "id": image_id,
        }
        json_dict["images"].append(image)
        ## Currently we do not support segmentation.
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        for obj in get(root, "object"):
            category = get_and_check(obj, "name", 1).text
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, "bndbox", 1)
            xmin = int(get_and_check(bndbox, "xmin", 1).text) - 1
            ymin = int(get_and_check(bndbox, "ymin", 1).text) - 1
            xmax = int(get_and_check(bndbox, "xmax", 1).text)
            ymax = int(get_and_check(bndbox, "ymax", 1).text)
            assert xmax > xmin
            assert ymax > ymin
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {
                "area": o_width * o_height,
                "iscrowd": 0,
                "image_id": image_id,
                "bbox": [xmin, ymin, o_width, o_height],
                "category_id": category_id,
                "id": bnd_id,
                "ignore": 0,
                "segmentation": [],
            }
            json_dict["annotations"].append(ann)
            bnd_id = bnd_id + 1

    for cate, cid in categories.items():
        cat = {"supercategory": "none", "id": cid, "name": cate}
        json_dict["categories"].append(cat)

    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    json_fp = open(json_file, "w")
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()

def rich_logger(num_first,num_last, notification):
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )
    log = logging.getLogger("rich")
    formatted_numbers = f"[{num_first}|{num_last}]"
    log.info(f"{formatted_numbers} - {notification}!")

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

# Can lam: Chuyen de nhan train, val, test
# Dung rich de ve

def get_args():
    parser = argparse.ArgumentParser(
        description="Convert Pascal VOC annotation to COCO format."
    )

    ## To do

    # Nhan 1 folder va chuyen sang COCO
    parser.add_argument("--train", action="store_true",
                        help="Set output to train2017.json and move images to train2017")

    parser.add_argument("--val", action="store_true",
                        help="Set output to train2017.json and move images to val2017")

    parser.add_argument("--test", action="store_true",
                        help="Set output to train2017.json and move images to test2017")

    parser.add_argument("--dataset-name", help="Name of the dataset.", type=str, default="COCO2017")
    parser.add_argument("--image_dir", help="Directory path to images.", type=str,required=True)
    parser.add_argument("xml_dir", help="Directory path to xml files.", type=str)
    parser.add_argument("json_file", help="Output COCO format json file.", type=str)


    args = parser.parse_args()
    return args

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


def main(opt):
    head_rich_logger()
    text = Text("Starting conversion!", style="bold blue")
    rprint(text)

    # Nhan full folder va doc tat ca file xml va Txt
    ## To do
    # Nhan 1 folder va chuyen sang COCO
    rich_logger(1, 4, "Finding all xml files in the directory. Please wait...")
    xml_files = glob.glob(os.path.join(opt.xml_dir, "*.xml"))
    rich_logger(1, 4, "Find successfully. Found {} xml files".format(len(xml_files)))
    rich_logger(2, 4, "Converting xml files to json. Please wait...")
    convert(xml_files, opt.json_file)
    rich_logger(3, 4, "Convert successfully. Start checking if the images in the COCO JSON file exist in the specified directory....")
    existCheck.check_exist(opt.json_file, opt.image_dir)
    rich_logger(3, 4, "Checking successfully.")
    rich_logger(4, 4, "Start checking if the size of the images in the COCO JSON file matches the actual size of the images...")
    checkSize.checkingSize(opt.json_file, opt.image_dir)
    rich_logger(4, 4, "Checking successfully.")


    if opt.train is True:
        rich_logger(1, 5, "Checking images in the directory. Moving...")
        trainvaltest_activtation(opt.image_dir,opt.dataset_name,"train")
        rich_logger(1, 5, "Check successfully. Images are moved to train2017 directory.")

        rich_logger(2, 5, "Finding all xml files in the directory. Please wait...")
        os.makedirs('annotations', exist_ok=True)

        xml_files = glob.glob(os.path.join(opt.xml_dir, "*.xml"))
        rich_logger(2, 5, "Find successfully. Found {} xml files".format(len(xml_files)))
        rich_logger(3, 5, "Converting xml files to json. Please wait...")

        json_file="train2017.json"
        json_files = os.path.join(opt.dataset_name,"annotations",json_file)
        convert(xml_files, json_files)
        rich_logger(3, 5,
                    "Convert successfully. Start checking if the images in the COCO JSON file exist in the specified directory....")
        existCheck.check_exist(opt.json_file, "train2017")
        rich_logger(4, 5, "Checking successfully.")
        rich_logger(5, 5,
                    "Start checking if the size of the images in the COCO JSON file matches the actual size of the images...")
        checkSize.checkingSize(opt.json_file, "train2017")
        rich_logger(5, 5, f"Checking successfully. You can find final output is in folder {opt.dataset_name}")

    if opt.val is True:
        rich_logger(1, 5, "Checking images in the directory. Moving...")
        trainvaltest_activtation(opt.image_dir,opt.dataset_name,"val")
        rich_logger(1, 5, "Check successfully. Images are moved to val2017 directory.")

        rich_logger(2, 5, "Finding all xml files in the directory. Please wait...")
        os.makedirs('annotations', exist_ok=True)

        xml_files = glob.glob(os.path.join(opt.xml_dir, "*.xml"))
        rich_logger(2, 5, "Find successfully. Found {} xml files".format(len(xml_files)))
        rich_logger(3, 5, "Converting xml files to json. Please wait...")

        json_file="train2017.json"
        json_files = os.path.join(opt.dataset_name,"annotations",json_file)
        convert(xml_files, json_files)
        rich_logger(3, 5,
                    "Convert successfully. Start checking if the images in the COCO JSON file exist in the specified directory....")
        existCheck.check_exist(opt.json_file, "val2017")
        rich_logger(4, 5, "Checking successfully.")
        rich_logger(5, 5,
                    "Start checking if the size of the images in the COCO JSON file matches the actual size of the images...")
        checkSize.checkingSize(opt.json_file, "val2017")
        rich_logger(5, 5, f"Checking successfully. You can find final output is in folder {opt.dataset_name}")

    if opt.test is True:
        rich_logger(1, 5, "Checking images in the directory. Moving...")
        trainvaltest_activtation(opt.image_dir,opt.dataset_name,"test")
        rich_logger(1, 5, "Check successfully. Images are moved to test2017 directory.")

        rich_logger(2, 5, "Finding all xml files in the directory. Please wait...")
        os.makedirs('annotations', exist_ok=True)

        xml_files = glob.glob(os.path.join(opt.xml_dir, "*.xml"))
        rich_logger(2, 5, "Find successfully. Found {} xml files".format(len(xml_files)))
        rich_logger(3, 5, "Converting xml files to json. Please wait...")

        json_file="train2017.json"
        json_files = os.path.join(opt.dataset_name,"annotations",json_file)
        convert(xml_files, json_files)
        rich_logger(3, 5,
                    "Convert successfully. Start checking if the images in the COCO JSON file exist in the specified directory....")
        existCheck.check_exist(opt.json_file, "test2017")
        rich_logger(4, 5, "Checking successfully.")
        rich_logger(5, 5,
                    "Start checking if the size of the images in the COCO JSON file matches the actual size of the images...")
        checkSize.checkingSize(opt.json_file, "test2017")
        rich_logger(5, 5, f"Checking successfully. You can find final output is in folder {opt.dataset_name}")

if __name__ == "__main__":
    options = get_args()
    main(options)
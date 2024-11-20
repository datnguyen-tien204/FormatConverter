###################### All code referenced from datnguyen-tien204 #########
####################  Github: https://github.com/datnguyen-tien204  #########
############ Profile: http://tien-datnguyen-blogs.me/
### This code is used to check if the images in the YOLO images and YOLO labels match exist in the specified directory. #######

import os
import xml.etree.ElementTree as ET
from PIL import Image
from .folder_create import rich_logger

def fix_mismatched_sizes(image_folder, annotation_folder,extensions=".jpg", type="voc"):
    if type == "voc":
        for xml_file in os.listdir(annotation_folder):
            if xml_file.endswith(".xml"):
                xml_path = os.path.join(annotation_folder, xml_file)
                tree = ET.parse(xml_path)
                root = tree.getroot()
                width = int(root.find("size/width").text)
                height = int(root.find("size/height").text)
                image_file = os.path.splitext(xml_file)[0] + extensions
                image_path = os.path.join(image_folder, image_file)

                if not os.path.exists(image_path):
                    rich_logger(5, 6, f"Missing image file for: {xml_file}")
                    continue
                with Image.open(image_path) as img:
                    actual_width, actual_height = img.size
                if width != actual_width or height != actual_height:
                    rich_logger(5, 6, f"Fixing size mismatch for: {xml_file}")
                    root.find("size/width").text = str(actual_width)
                    root.find("size/height").text = str(actual_height)
                    tree.write(xml_path)
    if type=="yolo":
        for xml_file in os.listdir(annotation_folder):
            if xml_file.endswith(".txt"):
                xml_path = os.path.join(annotation_folder, xml_file)
                tree = ET.parse(xml_path)
                root = tree.getroot()
                width = int(root.find("size/width").text)
                height = int(root.find("size/height").text)
                image_file = os.path.splitext(xml_file)[0] + extensions
                image_path = os.path.join(image_folder, image_file)

                if not os.path.exists(image_path):
                    rich_logger(5, 6, f"Missing image file for: {xml_file}")
                    continue
                with Image.open(image_path) as img:
                    actual_width, actual_height = img.size
                if width != actual_width or height != actual_height:
                    rich_logger(5, 6, f"Fixing size mismatch for: {xml_file}")
                    root.find("size/width").text = str(actual_width)
                    root.find("size/height").text = str(actual_height)
                    tree.write(xml_path)

def clean_invalid_files(image_folder, annotation_folder, trainval_file,extensions=".jpg"):
    with open(trainval_file, "r") as f:
        trainval_names = {line.strip() for line in f if line.strip()}

    image_names = {os.path.splitext(f)[0] for f in os.listdir(image_folder) if f.endswith(extensions)}
    annotation_names = {os.path.splitext(f)[0] for f in os.listdir(annotation_folder) if f.endswith(".xml")}

    valid_names = trainval_names & image_names & annotation_names
    invalid_images = image_names - valid_names
    invalid_annotations = annotation_names - valid_names
    invalid_in_trainval = trainval_names - valid_names

    for name in invalid_images:
        os.remove(os.path.join(image_folder, name + extensions))
        rich_logger(5, 6, f"Removed invalid image: {name}{extensions}")

    for name in invalid_annotations:
        os.remove(os.path.join(annotation_folder, name + ".xml"))
        rich_logger(5, 6, f"Removed invalid annotation: {name}.xml")

    if invalid_in_trainval:
        rich_logger(5, 6, "Updating trainval.txt to remove invalid entries...")
        with open(trainval_file, "w") as f:
            for name in sorted(valid_names):
                f.write(name + "\n")


def clean_annotations(images_folder, labels_folder, annotations_file, extensions_in=".jpg"):
    if not os.path.exists(images_folder):
        raise FileNotFoundError(f"Image folder not exist: {images_folder}")
    if not os.path.exists(labels_folder):
        raise FileNotFoundError(f"Label folder not exist: {labels_folder}")
    if not os.path.exists(annotations_file):
        raise FileNotFoundError(f"Annotations file not exist: {annotations_file}")

    # Lấy danh sách file ảnh và file nhãn (không bao gồm phần mở rộng)
    image_files = {os.path.splitext(f)[0] for f in os.listdir(images_folder) if f.endswith(extensions_in)}
    label_files = {os.path.splitext(f)[0] for f in os.listdir(labels_folder) if f.endswith(".xml")}

    valid_lines = []
    with open(annotations_file, "r") as f:
        for line in f:
            image_name = line.strip()
            if image_name in image_files and image_name in label_files:
                valid_lines.append(image_name)

    # Ghi lại các dòng hợp lệ vào file annotations
    with open(annotations_file, "w") as f:
        f.writelines(line + "\n" for line in valid_lines)

    rich_logger(5, 6, f"Cleaned annotations file: {annotations_file}. Valid lines: {len(valid_lines)}")


def validate_and_fix_voc_dataset(image_folder, annotation_folder, trainval_file,extensions_in=".jpg"):
    rich_logger(5, 6, "Checking and fixing mismatched sizes...")
    fix_mismatched_sizes(image_folder, annotation_folder,extensions=extensions_in)
    rich_logger(5, 6, "Cleaning up invalid files...")
    clean_invalid_files(image_folder, annotation_folder, trainval_file,extensions=extensions_in)
    rich_logger(5, 6, "Validation and fixes completed.")

def validate_and_fix_voc_dataset_voc_input(image_folder, annotation_folder,extensions_in=".jpg", type="voc"):
    rich_logger(2, 6, "Checking and fixing mismatched sizes...")
    fix_mismatched_sizes(image_folder, annotation_folder,extensions=extensions_in, type=type)

def validate_and_fix_voc_dataset_yolo_output(image_folder, annotation_folder,extensions_in=".jpg",type="yolo"):
    rich_logger(5, 6, "Checking and fixing mismatched sizes...")
    fix_mismatched_sizes(image_folder, annotation_folder,extensions=extensions_in,type=type)
def cleaning_files(images_folder, labels_folder, annotations_file,extensions_in=".jpg"):
    rich_logger(5, 6, f"Cleaning up invalid files {annotations_file}...")
    clean_annotations(images_folder, labels_folder, annotations_file,extensions_in=extensions_in)
    rich_logger(5, 6, f"Validation file {annotations_file} and fixes completed.")
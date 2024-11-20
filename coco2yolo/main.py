"""
Alright reserved belonging to datnguyen-tien204
Profile: http://tien-datnguyen-blogs.me/
Convert from COCO -> YOLO

"""

import json
import os
import argparse
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from utils import head_rich_logger, rich_logger
from rich.text import Text
from rich.console import Console
from rich import print as rprint
from checking.checkSize import checkingSize
from checking.existCheck import check_and_handle_files
from checking.folder_create import create_folder_structure, move_files_and_generate_yolo_v1

def get_args():
    parser = argparse.ArgumentParser(description='COCO format to YOLO format.')


    ## To do
    parser.add_argument("--extension_inp", choices=["jpg", "jpeg", "png"], help="Extension of image file",
                        default=".jpg")

    ## Config part 1. Input: Label directory and image directory | Output: Output Directory.
    parser.add_argument("--label_dir", type=str,
                        help="Path to COCO json label files. If use this not need to use --project_dir")
    parser.add_argument("--image_dir", type=str,
                        help="Path to COCO images directory. If use this not need to use --project_dir")
    parser.add_argument("--output_dir", type=str,
                        help="Path to folder output. If use this not need to use --project_dir", default="Output_YOLO")

    ## Config part 1. Input: Project directory | Output: Output Directory.

    parser.add_argument("--project_dir", type=str,
                        help="Path to project. If use this not need to use --label_dir, --image_dir and --output_dir")
    parser.add_argument("--train", action="store_true",
                        help="Set output to VOC train. If use this not need to use --label_dir, --image_dir and --output_dir ")
    parser.add_argument("--valid", action="store_true",
                        help="Set output to VOC val. If use this not need to use --label_dir, --image_dir and --output_dir ")
    parser.add_argument("--test", action="store_true",
                        help="Set output to VOC test. If use this not need to use --label_dir, --image_dir and --output_dir ")
    parser.add_argument("--dataset-name", help="Name of the dataset.", type=str, default="Output_VOC")
    args = parser.parse_args()
    return args

def check_file_and_dir(file_path, dir_path):
    if not os.path.exists(file_path):
        raise ValueError("file not found")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def load_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_categories(labels):
    categories = {}
    for cls in labels['categories']:
        categories[cls['id']] = cls['name']
    return categories

def load_images_info(labels):
    images_info = {}
    for image in labels['images']:
        img_id = image['id']
        file_name = image['file_name']
        if '\\' in file_name:
            file_name = file_name.split('\\')[-1]
        w = image['width']
        h = image['height']
        images_info[img_id] = (file_name, w, h)
    return images_info

def bbox_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox
    centerx = x + w / 2
    centery = y + h / 2
    dw = 1 / img_w
    dh = 1 / img_h
    return centerx * dw, centery * dh, w * dw, h * dh

def convert_annotations(labels, images_info, coco_id_name_map, coco_name_list):
    anno_dict = {}
    for anno in labels['annotations']:
        bbox = anno['bbox']
        image_id = anno['image_id']
        category_id = anno['category_id']

        image_info = images_info.get(image_id)
        if image_info:
            image_name, img_w, img_h = image_info
            yolo_box = bbox_to_yolo(bbox, img_w, img_h)

            anno_info = (image_name, category_id, yolo_box)
            if image_id not in anno_dict:
                anno_dict[image_id] = [anno_info]
            else:
                anno_dict[image_id].append(anno_info)
    return anno_dict

def save_classes(labels, output_file='classes.txt'):
    sorted_classes = sorted(labels['categories'], key=lambda x: x['id'])
    class_names = [cls['name'] for cls in sorted_classes]
    with open(output_file, 'w', encoding='utf-8') as f:
        for cls in class_names:
            f.write(cls + '\n')

def save_txt(anno_dict, coco_id_name_map, coco_name_list, output):
    for annotations in anno_dict.values():
        file_name = os.path.splitext(annotations[0][0])[0] + ".txt"
        with open(os.path.join(output, file_name), 'w', encoding='utf-8') as f:
            for obj in annotations:
                cat_name = coco_id_name_map[obj[1]]
                category_id = coco_name_list.index(cat_name)
                box = ' '.join(['{:.6f}'.format(x) for x in obj[2]])
                line = f"{category_id} {box}"
                f.write(line + '\n')

def main(opt):
    head_rich_logger()

    if opt.project_dir and any([opt.label_dir, opt.image_dir, opt.output_dir]):
        raise ValueError(
            "You must choose either `--project_dir` or the combination of `--label_dir`, `--image_dir`, and `--output_dir`. "
            "Both options cannot be used simultaneously."
        )

    if not opt.project_dir and not (opt.label_dir and opt.image_dir and opt.output_dir):
        raise ValueError(
            "You must specify either `--project_dir` or provide `--label_dir`, `--image_dir`, and `--output_dir` together."
        )

    if opt.extension_inp not in ["jpg","jpeg","png"]:
        raise ValueError(
            "Extension of image file must be in ['jpg','jpeg','png']"
        )

    if opt.extension_inp == "jpg":
        extension = ".jpg"
    if opt.extension_inp == "jpeg":
        extension = ".jpeg"
    if opt.extension_inp == "png":
        extension = ".png"

    text = Text("[Mode] Conversion COCO Format to YOLO format ", style="bold green")
    rprint(text)
    text = Text("Starting conversion...", style="bold blue")
    rprint(text)

    if opt.label_dir and opt.image_dir and opt.output_dir:
        label_dir = opt.label_dir
        image_dir = opt.image_dir
        output_dir = opt.output_dir
        os.makedirs("temp", exist_ok=True)
        rich_logger(1, 6, "Starting checking compability of file...")
        checkingSize(label_dir, image_dir,extensions=extension)
        rich_logger(1, 6, "Checking compability of file done!")
        coco_to_yolo(label_dir, "temp", image_dir, output_dir,extension=extension)
        os.rmdir("temp")


def coco_to_yolo(json_file, label_folder_out,image_folder,output_folder,extension=".jpg"):
    check_file_and_dir(json_file, label_folder_out)
    labels = load_json(json_file)
    coco_id_name_map = extract_categories(labels)
    coco_name_list = list(coco_id_name_map.values())

    rich_logger(2, 6, "Starting conversion...")
    rich_logger(2, 6, f"Total images {len(labels['images'])}")
    rich_logger(2, 6, f"Total categories {len(labels['categories'])}")
    rich_logger(2, 6, f"Total labels {len(labels['annotations'])}")

    rich_logger(2, 6, "Starting loading image info...")
    images_info = load_images_info(labels)
    anno_dict = convert_annotations(labels, images_info, coco_id_name_map, coco_name_list)
    rich_logger(2, 6, "Converting done. Total labels: {}".format(len(anno_dict)))
    rich_logger(3, 6, "Saving to classes txt...")
    save_txt(anno_dict, coco_id_name_map, coco_name_list, label_folder_out)
    save_classes(labels)
    rich_logger(3, 6, "Saving done.")
    rich_logger(4, 6, "Moving files...")
    annotations_dir, jpeg_images_dir=move_files_and_generate_yolo_v1(image_folder, label_folder_out, output_folder,extensions=extension)
    rich_logger(4, 6, "Moving done.")
    rich_logger(5, 6, "Final checking...")
    # checkingSize(annotations_dir, jpeg_images_dir, extensions=extension)
    has_discrepancy = check_and_handle_files(jpeg_images_dir, annotations_dir, extensions=extension)
    if not has_discrepancy:
        rich_logger(1, 6, "No extra or missing files found.")
    rich_logger(5, 6, "Checking done.")
    rich_logger(6, 6, "Conversion done.")

def coco2017toYOLO():
    pass

if __name__ == '__main__':
    args = get_args()
    main(args)

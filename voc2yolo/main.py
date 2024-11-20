"""
Alright reserved belonging to datnguyen-tien204
Profile: http://tien-datnguyen-blogs.me/
Convert from VOC -> YOLO
"""
import argparse
import multiprocessing
import os
from xml.etree import ElementTree
from PIL import Image
from pascal_voc_writer import Writer
from rich import print as rprint
from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text
import logging
from rich.panel import Panel
from rich.markdown import Markdown
from functools import partial
from checking.existCheck import check_and_handle_files
from checking.checkSize import validate_and_fix_voc_dataset_voc_input,validate_and_fix_voc_dataset_yolo_output
from checking.folder_create import move_files_and_generate_yolo_v1,create_folder_structure,move_files_and_generate_type2
from pathlib import Path

def rich_logger(num_first,num_last, notification):
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )
    log = logging.getLogger("rich")
    formatted_numbers = f"[{num_first}|{num_last}]"
    log.info(f"{formatted_numbers} - {notification}")


def read_classes_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            names = [line.strip() for line in file if line.strip()]
        return names
    except FileNotFoundError:
        raise FileNotFoundError(f"[Error] File not found: {file_path}. Please check again!")


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


def voc2yolo(xml_file: str,xml_dir:str,classes_dir:str) -> None:
    """Convert VOC to YOLO
    Args:
        xml_file: str
    """
    with open(f"{xml_dir}/{xml_file}") as in_file:
        tree = ElementTree.parse(in_file)
        size = tree.getroot().find("size")
        height, width = map(int, [size.find("height").text, size.find("width").text])

    names=read_classes_from_file(classes_dir)
    class_exists = False
    for obj in tree.findall("object"):
        name = obj.find("name").text
        if name in names:
            class_exists = True

    if class_exists:
        with open(f"{xml_dir}/{xml_file[:-4]}.txt", "w") as out_file:
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

                cls_id = names.index(obj.find("name").text)
                out_file.write(str(cls_id) + " " + " ".join([str(f"{i:.6f}") for i in b]) + "\n")


def voc2yolo_a(xml_file: str,xml_dir:str,classes_dir:str) -> None:
    """Convert VOC to YOLO with absolute cordinates
    Args:
        xml_file: str
    """
    with open(f"{xml_dir}/{xml_file}") as in_file:
        tree = ElementTree.parse(in_file)

    names=read_classes_from_file(classes_dir)
    class_exists = False
    for obj in tree.findall("object"):
        name = obj.find("name").text
        if name in names:
            class_exists = True

    if class_exists:
        with open(f"{xml_dir}/{xml_file[:-4]}.txt", "w") as out_file:
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
                cls_id = names.index(obj.find("name").text)
                out_file.write(str(cls_id) + " " + " ".join([str(f"{i}") for i in b]) + "\n")


def get_args():
    parser = argparse.ArgumentParser(
        description="Convert Pascal VOC annotation to YOLO format."
    )
    ## To do
    parser.add_argument("--voc2yolo", action="store_true", help="VOC to YOLO. Convert VOC to YOLO relative")
    parser.add_argument("--voc2yolo_a", action="store_true", help="VOC to YOLO absolute")

    parser.add_argument("--extension_inp",choices=["jpg","jpeg","png"],help="Extension of image file",default=".jpg")
    parser.add_argument("--list_classes", type=str, help="Path to list classes, must be in txt format.")

    ## Config part 1. Input: Label directory and image directory | Output: Output Directory.
    parser.add_argument("--label_dir",type=str,help="Path to YOLO label directory. If use this not need to use --project_dir")
    parser.add_argument("--image_dir", type=str, help="Path to YOLO images directory. If use this not need to use --project_dir")
    parser.add_argument("--output_dir",type=str,help="Path to folder output. If use this not need to use --project_dir")

    ## Config part 1. Input: Project directory | Output: Output Directory.
    # Nhan 1 folder va chuyen sang COCO
    parser.add_argument("--project_dir", type=str, help="Path to project. If use this not need to use --label_dir, --image_dir and --output_dir")
    parser.add_argument("--train", action="store_true", help="Set output to VOC train. If use this not need to use --label_dir, --image_dir and --output_dir ")
    parser.add_argument("--valid", action="store_true",help="Set output to VOC val. If use this not need to use --label_dir, --image_dir and --output_dir ")
    parser.add_argument("--test", action="store_true", help="Set output to VOC test. If use this not need to use --label_dir, --image_dir and --output_dir ")
    parser.add_argument("--dataset-name", help="Name of the dataset.", type=str, default="Output_YOLO")
    args = parser.parse_args()
    return args


def main(opt):
    head_rich_logger()
    #Check Logic
    if not opt.voc2yolo and not (opt.voc2yolo_a):
        raise ValueError(
            "You must specify either `--yolo2voc` or `--voc2yolo_a`."
        )
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

   # YOLO2VOC Relative
    if opt.voc2yolo:
        text = Text("[Mode] Conversion to YOLO Relative-BoundingBox for images and labels folder. ", style="bold green")
        rprint(text)
        text = Text("Starting conversion...", style="bold blue")
        rprint(text)

        if opt.label_dir and opt.image_dir and opt.output_dir:
            rich_logger(1,6, "Starting checking compability of file...")

            has_discrepancy = check_and_handle_files(opt.image_dir, opt.label_dir,extensions=extension)
            if not has_discrepancy:
                rich_logger(1,6, "No extra or missing files found.")
            rich_logger(1,6, "Checking compability of file successfully")
            validate_and_fix_voc_dataset_voc_input(opt.image_dir, opt.label_dir,extensions_in=extension)
            rich_logger(2,6, "Matching file successfully")

            rich_logger(3, 6, "Preparing all files *.xml in folder")
            txt_files = [
                name for name in os.listdir(opt.label_dir) if name.endswith(".xml")
            ]
            rich_logger(3, 6, "Preparing all file successfully")
            rich_logger(4, 6, "Starting conversion Boundingbox....")
            convert_partial = partial(
                voc2yolo,
                classes_dir=opt.list_classes,
                xml_dir=opt.label_dir,
            )
            with multiprocessing.Pool(os.cpu_count()) as pool:
                pool.map(convert_partial, txt_files)
            pool.close()
            pool.join()

            rich_logger(3, 6, "Successfully conversion YOLO-Bounding-Box to VOC-Bounding-box")
            rich_logger(4, 6, "Starting moving file to output folder...")
            annotations_dir, jpeg_images_dir=move_files_and_generate_yolo_v1(opt.image_dir, opt.label_dir, opt.output_dir,extensions=extension)
            rich_logger(4, 6, "Successfully moving file to output folder")
            rich_logger(5,6 ,"Starting final checking...")
            validate_and_fix_voc_dataset_yolo_output(jpeg_images_dir, annotations_dir,extensions_in=extension)
            rich_logger(5,6, "Final checking successfully")
            rich_logger(6, 6, "Conversion successfully completed!")

        #Project Directory
        if opt.project_dir and opt.voc2yolo:
            rich_logger(1, 6, "Starting checking compability of file...")

            ann_folder_path = Path(opt.project_dir) / "Annotations"
            img_folder_path = Path(opt.project_dir) / "JPEGImages"
            set_folder_path = Path(opt.project_dir) / "ImageSets/Main"

            if not ann_folder_path.exists():
                raise FileNotFoundError(f"[Error] Folder not found: {ann_folder_path}")
            if not img_folder_path.exists():
                raise FileNotFoundError(f"[Error] Folder not found: {img_folder_path}")
            if not set_folder_path.exists():
                raise FileNotFoundError(f"[Error] Folder not found: {set_folder_path}")

            ### Checking
            has_discrepancy = check_and_handle_files(img_folder_path, ann_folder_path, extensions=extension)
            if not has_discrepancy:
                rich_logger(1,6, "No extra or missing files found.")
            rich_logger(1,6, "Checking compability of file successfully")
            validate_and_fix_voc_dataset_voc_input(img_folder_path, ann_folder_path,extensions_in=extension)
            rich_logger(2,6, "Matching file successfully")

            rich_logger(3, 6, "Preparing all files *.xml in folder")
            txt_files = [
                name for name in os.listdir(ann_folder_path) if name.endswith(".xml")
            ]
            rich_logger(3, 6, "Preparing all file successfully")
            rich_logger(4, 6, "Starting conversion Boundingbox....")
            convert_partial = partial(
                voc2yolo,
                classes_dir=opt.list_classes,
                xml_dir=ann_folder_path,
            )
            with multiprocessing.Pool(os.cpu_count()) as pool:
                pool.map(convert_partial, txt_files)
            pool.close()
            pool.join()

            rich_logger(3, 6, "Successfully conversion YOLO-Bounding-Box to VOC-Bounding-box")
            rich_logger(4, 6, "Starting moving file to output folder...")

            if opt.train:
                set_type = "train"
                output_dir = Path(opt.dataset_name)
                output_images_dir, output_labels_dir=move_files_and_generate_type2(img_folder_path, ann_folder_path, set_folder_path, output_dir, set_type=set_type, extension=extension)
                rich_logger(4, 6, f"Successfully moving [train ]file to output folder {output_dir}")
                rich_logger(5, 6, "Starting final checking...")
                validate_and_fix_voc_dataset_voc_input(output_images_dir, output_labels_dir,extensions_in=extension)
                rich_logger(5, 6, "Final checking successfully")
                rich_logger(6, 6, "Conversion successfully completed!")

            if opt.valid:
                set_type = "valid"
                output_dir = Path(opt.dataset_name)
                output_images_dir, output_labels_dir=move_files_and_generate_type2(img_folder_path, ann_folder_path, set_folder_path, output_dir, set_type=set_type, extension=extension)
                rich_logger(4, 6, f"Successfully moving [valid ]file to output folder {output_dir}")
                rich_logger(5, 6, "Starting final checking...")
                validate_and_fix_voc_dataset_voc_input(output_images_dir, output_labels_dir,extensions_in=extension)
                rich_logger(5, 6, "Final checking successfully")
                rich_logger(6, 6, "Conversion successfully completed!")
            if opt.test:
                set_type = "test"
                output_dir = Path(opt.dataset_name)
                output_images_dir, output_labels_dir=move_files_and_generate_type2(img_folder_path, ann_folder_path, set_folder_path, output_dir, set_type=set_type, extension=extension)
                rich_logger(4, 6, f"Successfully moving [test ]file to output folder {output_dir}")
                rich_logger(5, 6, "Starting final checking...")
                validate_and_fix_voc_dataset_voc_input(output_images_dir, output_labels_dir,extensions_in=extension)
                rich_logger(5, 6, "Final checking successfully")
                rich_logger(6, 6, "Conversion successfully completed!")

    if opt.voc2yolo_a:
        text = Text("[Mode] Conversion to YOLO Absolute-BoundingBox for images and labels folder. ",
                    style="bold green")
        rprint(text)
        text = Text("Starting conversion...", style="bold blue")
        rprint(text)

        if opt.label_dir and opt.image_dir and opt.output_dir:
            rich_logger(1, 6, "Starting checking compability of file...")

            has_discrepancy = check_and_handle_files(opt.image_dir, opt.label_dir, extensions=extension)
            if not has_discrepancy:
                rich_logger(1, 6, "No extra or missing files found.")
            rich_logger(1, 6, "Checking compability of file successfully")
            validate_and_fix_voc_dataset_voc_input(opt.image_dir, opt.label_dir, extensions_in=extension)
            rich_logger(2, 6, "Matching file successfully")

            rich_logger(3, 6, "Preparing all files *.xml in folder")
            txt_files = [
                name for name in os.listdir(opt.label_dir) if name.endswith(".xml")
            ]
            rich_logger(3, 6, "Preparing all file successfully")
            rich_logger(4, 6, "Starting conversion Boundingbox....")
            convert_partial = partial(
                voc2yolo_a,
                classes_dir=opt.list_classes,
                xml_dir=opt.label_dir,
            )
            with multiprocessing.Pool(os.cpu_count()) as pool:
                pool.map(convert_partial, txt_files)
            pool.close()
            pool.join()

            rich_logger(3, 6, "Successfully conversion YOLO-Bounding-Box to VOC-Bounding-box")
            rich_logger(4, 6, "Starting moving file to output folder...")
            annotations_dir, jpeg_images_dir = move_files_and_generate_yolo_v1(opt.image_dir, opt.label_dir,
                                                                               opt.output_dir,
                                                                               extensions=extension)
            rich_logger(4, 6, "Successfully moving file to output folder")
            rich_logger(5, 6, "Starting final checking...")
            validate_and_fix_voc_dataset_yolo_output(jpeg_images_dir, annotations_dir, extensions_in=extension)
            rich_logger(5, 6, "Final checking successfully")
            rich_logger(6, 6, "Conversion successfully completed!")

        # Project Directory
        if opt.project_dir and opt.voc2yolo_a:
            rich_logger(1, 6, "Starting checking compability of file...")

            ann_folder_path = Path(opt.project_dir) / "Annotations"
            img_folder_path = Path(opt.project_dir) / "JPEGImages"
            set_folder_path = Path(opt.project_dir) / "ImageSets/Main"

            if not ann_folder_path.exists():
                raise FileNotFoundError(f"[Error] Folder not found: {ann_folder_path}")
            if not img_folder_path.exists():
                raise FileNotFoundError(f"[Error] Folder not found: {img_folder_path}")
            if not set_folder_path.exists():
                raise FileNotFoundError(f"[Error] Folder not found: {set_folder_path}")

            ### Checking
            has_discrepancy = check_and_handle_files(img_folder_path, ann_folder_path, extensions=extension)
            if not has_discrepancy:
                rich_logger(1, 6, "No extra or missing files found.")
            rich_logger(1, 6, "Checking compability of file successfully")
            validate_and_fix_voc_dataset_voc_input(img_folder_path, ann_folder_path, extensions_in=extension)
            rich_logger(2, 6, "Matching file successfully")

            rich_logger(3, 6, "Preparing all files *.xml in folder")
            txt_files = [
                name for name in os.listdir(ann_folder_path) if name.endswith(".xml")
            ]
            rich_logger(3, 6, "Preparing all file successfully")
            rich_logger(4, 6, "Starting conversion Boundingbox....")
            convert_partial = partial(
                voc2yolo_a,
                classes_dir=opt.list_classes,
                xml_dir=ann_folder_path,
            )
            with multiprocessing.Pool(os.cpu_count()) as pool:
                pool.map(convert_partial, txt_files)
            pool.close()
            pool.join()

            rich_logger(3, 6, "Successfully conversion YOLO-Bounding-Box to VOC-Bounding-box")
            rich_logger(4, 6, "Starting moving file to output folder...")

            if opt.train:
                set_type = "train"
                output_dir = Path(opt.dataset_name)
                output_images_dir, output_labels_dir = move_files_and_generate_type2(img_folder_path,
                                                                                     ann_folder_path,
                                                                                     set_folder_path,
                                                                                     output_dir,
                                                                                     set_type=set_type,
                                                                                     extension=extension)
                rich_logger(4, 6, f"Successfully moving [train ]file to output folder {output_dir}")
                rich_logger(5, 6, "Starting final checking...")
                validate_and_fix_voc_dataset_voc_input(output_images_dir, output_labels_dir,
                                                       extensions_in=extension)
                rich_logger(5, 6, "Final checking successfully")
                rich_logger(6, 6, "Conversion successfully completed!")

            if opt.valid:
                set_type = "valid"
                output_dir = Path(opt.dataset_name)
                output_images_dir, output_labels_dir = move_files_and_generate_type2(img_folder_path,
                                                                                     ann_folder_path,
                                                                                     set_folder_path,
                                                                                     output_dir,
                                                                                     set_type=set_type,
                                                                                     extension=extension)
                rich_logger(4, 6, f"Successfully moving [valid ]file to output folder {output_dir}")
                rich_logger(5, 6, "Starting final checking...")
                validate_and_fix_voc_dataset_voc_input(output_images_dir, output_labels_dir,
                                                       extensions_in=extension)
                rich_logger(5, 6, "Final checking successfully")
                rich_logger(6, 6, "Conversion successfully completed!")
            if opt.test:
                set_type = "test"
                output_dir = Path(opt.dataset_name)
                output_images_dir, output_labels_dir = move_files_and_generate_type2(img_folder_path,
                                                                                     ann_folder_path,
                                                                                     set_folder_path,
                                                                                     output_dir,
                                                                                     set_type=set_type,
                                                                                     extension=extension)
                rich_logger(4, 6, f"Successfully moving [test ]file to output folder {output_dir}")
                rich_logger(5, 6, "Starting final checking...")
                validate_and_fix_voc_dataset_voc_input(output_images_dir, output_labels_dir,
                                                       extensions_in=extension)
                rich_logger(5, 6, "Final checking successfully")
                rich_logger(6, 6, "Conversion successfully completed!")






            # has_discrepancy = check_and_handle_files(opt.image_dir, opt.label_dir, extensions=extension)
            # if not has_discrepancy:
            #     rich_logger(1, 6, "No extra or missing files found.")
            # rich_logger(1, 6, "Checking compability of file successfully")
            # validate_and_fix_voc_dataset_voc_input(opt.image_dir, opt.label_dir, extensions_in=extension)
            # rich_logger(2, 6, "Matching file successfully")
            #
            # rich_logger(3, 6, "Preparing all files *.xml in folder")
            # txt_files = [
            #     name for name in os.listdir(opt.label_dir) if name.endswith(".xml")
            # ]
            # rich_logger(3, 6, "Preparing all file successfully")
            # rich_logger(4, 6, "Starting conversion Boundingbox....")
            # convert_partial = partial(
            #     voc2yolo,
            #     classes_dir=opt.list_classes,
            #     xml_dir=opt.label_dir,
            # )
            # with multiprocessing.Pool(os.cpu_count()) as pool:
            #     pool.map(convert_partial, txt_files)
            # pool.close()
            # pool.join()
            #
            # rich_logger(3, 6, "Successfully conversion YOLO-Bounding-Box to VOC-Bounding-box")
            #
            # rich_logger(4, 6, "Starting moving file to output folder...")





if __name__ == "__main__":
    options = get_args()
    main(options)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument("--yolo2voc", action="store_true", help="YOLO to VOC")
#     parser.add_argument("--voc2yolo", action="store_true", help="VOC to YOLO")
#     parser.add_argument("--voc2yolo_a", action="store_true", help="VOC to YOLO absolute")
#     args = parser.parse_args()
#
#     if args.yolo2voc:
#         print("YOLO to VOC")
#         txt_files = [
#             name for name in os.listdir(config.LABEL_DIR) if name.endswith(".txt")
#         ]
#
#         with multiprocessing.Pool(os.cpu_count()) as pool:
#             pool.map(yolo2voc, txt_files)
#         pool.join()
#
#     if args.voc2yolo:
#         print("VOC to YOLO")
#         xml_files = [
#             name for name in os.listdir(config.XML_DIR) if name.endswith(".xml")
#         ]
#         with multiprocessing.Pool(os.cpu_count()) as pool:
#             pool.map(voc2yolo, xml_files)
#         pool.join()
#
#     if args.voc2yolo_a:
#         xml_files = [
#             name for name in os.listdir(config.XML_DIR) if name.endswith(".xml")
#         ]
#         with multiprocessing.Pool(os.cpu_count()) as pool:
#             pool.map(voc2yolo_a, xml_files)
#         pool.close()
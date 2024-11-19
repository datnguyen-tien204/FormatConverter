"""
Alright reserved by datnguyen-tien204
Profile: http://tien-datnguyen-blogs.me/
Convert from YOLO -> VOC

"""
import argparse
import multiprocessing
import os
from xml.etree import ElementTree
from PIL import Image
from pascal_voc_writer import Writer
import config
from checking.existCheck import check_and_handle_files
import logging
from rich.logging import RichHandler
from rich import print as rprint
from checking.folder_create import move_files_and_generate_trainval
from checking.checkSize import validate_and_fix_voc_dataset


def yolo2voc(txt_file: str,classes_path:str,img_dir:str,label_dir:str,extension:str=".jpg") -> None:
    lst_classes = read_classes_from_file(classes_path)
    w, h = Image.open(os.path.join(img_dir, f"{txt_file[:-4]}.{extension}")).size
    writer = Writer(f"{txt_file[:-4]}.xml", w, h)
    with open(os.path.join(label_dir, txt_file)) as f:
        for line in f.readlines():
            label, x_center, y_center, width, height = line.rstrip().split(" ")
            x_min = int(w * max(float(x_center) - float(width) / 2, 0))
            x_max = int(w * min(float(x_center) + float(width) / 2, 1))
            y_min = int(h * max(float(y_center) - float(height) / 2, 0))
            y_max = int(h * min(float(y_center) + float(height) / 2, 1))
            writer.addObject(lst_classes[int(label)], x_min, y_min, x_max, y_max)
    writer.save(os.path.join(config.XML_DIR, f"{txt_file[:-4]}.xml"))


def rich_logger(num_first,num_last, notification):
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )
    log = logging.getLogger("rich")
    formatted_numbers = f"[{num_first}|{num_last}]"
    log.info(f"{formatted_numbers} - {notification}!")


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

def read_classes_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            names = [line.strip() for line in file if line.strip()]
        return names
    except FileNotFoundError:
        raise FileNotFoundError(f"[Error] File not found: {file_path}. Please check again!")


def get_args(opt):
    parser = argparse.ArgumentParser(
        description="Convert Pascal YOLO annotation to VOC format."
    )

    ## To do
    parser.add_argument("--yolo2voc", action="store_true", help="YOLO to VOC")
    parser.add_argument("--voc2yolo_a", action="store_true", help="VOC to YOLO absolute")
    parser.add_argument("--extension_inp",choices=["jpg","jpeg","png"],help="Extension of image file",default=".jpg")
    parser.add_argument("list_classes", type=str, help="Path to list classes, must be in txt format.")

    ## Config part 1. Input: Label directory and image directory | Output: Output Directory.
    parser.add_argument("--label_dir",type=str,help="Path to YOLO label directory. If use this not need to use --project_dir")
    parser.add_argument("--image_dir", type=str, help="Path to YOLO images directory. If use this not need to use --project_dir")
    parser.add_argument("--output_dir",type=str,help="Path to folder output. If use this not need to use --project_dir")

    ## Config part 1. Input: Project directory | Output: Output Directory.
    # Nhan 1 folder va chuyen sang COCO

    parser.add_argument("--project_dir", type=str, help="Path to project. If use this not need to use --label_dir, --image_dir and --output_dir")
    parser.add_argument("--train", action="store_true", help="Set output to VOC train. If use this not need to use --label_dir, --image_dir and --output_dir ")
    parser.add_argument("--val", action="store_true",help="Set output to VOC val. If use this not need to use --label_dir, --image_dir and --output_dir ")
    parser.add_argument("--test", action="store_true", help="Set output to VOC test. If use this not need to use --label_dir, --image_dir and --output_dir ")
    parser.add_argument("--dataset-name", help="Name of the dataset.", type=str, default="Output_VOC")
    args = parser.parse_args()
    return args


def create_folder_structure(output_folder):
    """
    Tạo cấu trúc thư mục đầu ra nếu chưa tồn tại.
    """
    annotations_dir = os.path.join(output_folder, "Annotations")
    jpeg_images_dir = os.path.join(output_folder, "JPEGImages")
    image_sets_dir = os.path.join(output_folder, "ImageSets/Main")

    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(jpeg_images_dir, exist_ok=True)
    os.makedirs(image_sets_dir, exist_ok=True)

    return annotations_dir, jpeg_images_dir, image_sets_dir




def main(opt):
    head_rich_logger()

    #Check Logic
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
    if opt.yolo2voc:
        text = Text("[Mode] Conversion to YOLO Relative-BoundingBox. ", style="bold green")
        rprint(text)
        text = Text("Starting conversion...", style="bold blue")
        rprint(text)

        if opt.label_dir and opt.image_dir and opt.output_dir:
            rich_logger(1,6, "Starting checking compability of file...")

            has_discrepancy = check_and_handle_files(opt.image_dir, opt.label_dir,extensions=extension)
            if not has_discrepancy:
                print("[bold green]No extra or missing files found.[/bold green]")
            rich_logger(1,6, "Checking compability of file successfully")
            rich_logger(2, 6, "Preparing all files *.txt in folder")
            txt_files = [
                name for name in os.listdir(config.LABEL_DIR) if name.endswith(".txt")
            ]
            rich_logger(2, 6, "Preparing all file successfully")
            rich_logger(3, 6, "Starting conversion Boundingbox....")
            with multiprocessing.Pool(os.cpu_count()) as pool:
                #def yolo2voc(txt_file: str,classes_path:str,img_dir:str,label_dir:str,extension:str=".jpg") -> None:
                pool.map(yolo2voc, txt_files,classes_path=opt.list_classes,img_dir=opt.image_dir,label_dir=opt.label_dir,extension=extension)
            pool.join()
            rich_logger(3, 6, "Successfully conversion YOLO-Bounding-Box to VOC-Bounding-box")
            rich_logger(4, 6, "Starting moving file to output folder...")
            annotations_dir, jpeg_images_dir, trainval_path=move_files_and_generate_trainval(opt.image_dir, opt.label_dir, opt.output_dir,extensions=extension)
            rich_logger(4, 6, "Successfully moving file to output folder")
            rich_logger(5,6 ,"Starting final checking...")
            validate_and_fix_voc_dataset(jpeg_images_dir, annotations_dir, trainval_path,extensions_in=extension)
            rich_logger(5, 6, "Validation and fixes completed.")
            rich_logger(6, 6, "Conversion successfully completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--yolo2voc", action="store_true", help="YOLO to VOC")
    parser.add_argument("--voc2yolo_a", action="store_true", help="VOC to YOLO absolute")
    args = parser.parse_args()

    # head_rich_logger()
    # text=Text("Starting conversion",style="bold blue")
    # rich_logger(1, 6, "Finding all files *.json in folder")
    txt_files = [
        name for name in os.listdir(config.LABEL_DIR) if name.endswith(".txt")
    ]
    # rich_logger(1,6,"Finding all file successfully")
    # rich_logger(2,6, "Starting conversion!")
    with multiprocessing.Pool(os.cpu_count()) as pool:
        pool.map(yolo2voc, txt_files)
    pool.join()
    # rich_logger()

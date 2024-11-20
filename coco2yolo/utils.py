"""
Alright reserved belonging to datnguyen-tien204
Profile: http://tien-datnguyen-blogs.me/
Convert from COCO -> YOLO

"""

import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

def rich_logger(num_first, num_last, notification):
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )
    log = logging.getLogger("rich")
    formatted_numbers = f"[{num_first}|{num_last}]"
    log.info(f"{formatted_numbers} - {notification}")


def head_rich_logger():
    MARKDOWN = """
    For more information about the COCO to YOLO Converter, please visit the: https://github.com/datnguyen-tien204/FormatConverter

    **General Options:**
    - `--extension_inp`: **(str)** : Extension of image files (e.g., "jpg", "jpeg", "png"). Default is `.jpg`.

    **Input Configuration:**
    - `--label_dir`: **(str)** : Path to the folder containing COCO JSON label files. If specified, you don't need to use `--project_dir`.
    - `--image_dir`: **(str)** : Path to the folder containing COCO images. If specified, you don't need to use `--project_dir`.
    - `--output_dir`: **(str)** : Path to the folder where the output will be stored. If specified, you don't need to use `--project_dir`.
    **Project Directory Configuration (With a datasets):**
    - `--project_dir`: **(str)** : Path to the project folder containing `labels`, `images`, or equivalent. If specified, you don't need to use `--label_dir`, `--image_dir`, or `--output_dir`.
    - `--train`: **(bool)** : If specified, the output will be generated for the VOC train set.
    - `--valid`: **(bool)** : If specified, the output will be generated for the VOC validation set.
    - `--test`: **(bool)** : If specified, the output will be generated for the VOC test set.
    - `--dataset-name`: **(str)** : Specify the name of the dataset. Default is `Output_VOC`.
    """
    console = Console()
    md = Markdown(MARKDOWN, style="white", code_theme="manni")
    panel = Panel(md, title="YOLO to COCO Converter Parameters", expand=False, style="on grey93", border_style="blue")
    console.print(panel)
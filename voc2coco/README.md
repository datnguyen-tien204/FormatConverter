
# Installation
For installation, you follow the readme.md in the main project.

# Arguments

```
--path_project_folder: Absolute path path_project_folder. If use this not use --path_image_folder and --path_labels_folder.
--path_image_folder: Absolute path for the folder containing images. If use this not need use to use --path_project_folder.
--path_labels_folder: Absolute path for the folder containing labels. If use this not need use to use --path_project_folder.
--train: **(bool)** : If specified, output JSON file is named `train2017.json` and images are moved to `train2017` directory.
--val`: **(bool)**    : If specified, output JSON file is named `val2017.json` and images are moved to `val2017` directory.
--test: **(bool)**   : If specified, output JSON file is named `test2017.json` and images are moved to `test2017` directory.
--debug: **(bool)**  : If specified, bounding boxes are visualized and annotation information is printed for debugging purposes.
--output: **(str)**  : Name of the output JSON file. Default is `train_coco.json`.
--yolo-subdir: **(bool)** : Indicates that annotations are stored in a subdirectory, not in the same directory as images.
--box2seg: **(bool)**  : Populates the COCO segmentation field with a polygon that replicates the bounding box data.
--results: **(bool)**  : Saves confidence scores from YOLO results to the COCO results format.
```

# How to run:

## With main folder
With this mode, you only specified **--path_project_folder**. Folder need to have format as:

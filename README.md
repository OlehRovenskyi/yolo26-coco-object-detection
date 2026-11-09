# Lab 4: Object detection with YOLO26

Lab work 4 for "Information Technologies in Scientific Research": detecting objects of given classes on
images with a convolutional neural network. The assignment describes YOLOv4; by agreement with the teacher
this project uses the newest Ultralytics model, **YOLO26**.

Repository: https://github.com/OlehRovenskyi/yolo26-coco-object-detection

| Task | What the script does |
|---|---|
| 1 | Detects objects of all 80 COCO classes on 10 images |
| 2 | Computes precision (p) and recall (r) on 10 COCO val2017 images for confidence 0.3, 0.5, 0.7, 0.9, picks the best threshold by F1 and finds the 3 worst results |
| 3 | Detects only the classes of variant 21: `remote` and `cell phone` (model class ids 65 and 67) |

## Requirements

- Python 3.10 or newer (tested with Python 3.14, ultralytics 8.4.208, torch 2.15.0 on macOS)
- About 1 GB of free disk space
- Internet access on the first run (model weights and images are downloaded automatically)

## Setup

```bash
git clone https://github.com/OlehRovenskyi/yolo26-coco-object-detection.git
cd yolo26-coco-object-detection

# 1. Virtual environment and packages
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# 2. COCO val2017 annotations (241 MB archive, only one 20 MB file is kept)
mkdir -p data && cd data
curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -o annotations_trainval2017.zip annotations/instances_val2017.json
rm annotations_trainval2017.zip
cd ..
```

On Windows use `.venv\Scripts\pip` and `.venv\Scripts\python` instead of `.venv/bin/...`.

## Run

```bash
.venv/bin/python lab4.py
```

On the first run the script downloads `yolo26m.pt` (42 MB) and the 14 selected images. The script skips
files that are already on disk. One run takes about a minute on a laptop CPU.

The console prints p, r and F1 for each threshold and the best threshold.

## Results

| Path | Contents |
|---|---|
| `results/task1_report.md` | Model settings and the list of detected objects on each image |
| `results/task1/` | Images with detected objects (class and confidence) |
| `results/task2_report.md` | Selected images, p/r table, totals with F1, best threshold, 3 worst results with explanation |
| `results/task2/conf_<threshold>/` | Images for each threshold: blue is a correct detection (TP), red is a false detection (FP), yellow is a missed object from the COCO annotation (FN) |
| `results/task3_report.md` | Variant 21 results: classes on each image, detections with and without the class filter |
| `results/task3/` | `NN_all_classes.jpg` (all classes) and `NN_variant_only.jpg` (only remote and cell phone) |
| `results/ІТ_НД_лаб_4_Ровенський_Олег_G3.docx` | Lab report (in Ukrainian) |

Original images are in `data/val2017/` under their COCO file names, e.g. `results/task1/02_81394.jpg` is
`data/val2017/000000081394.jpg`.

## How it works

- **Image selection.** Images come from COCO val2017, because it has annotations; test2017 does not.
  For tasks 1-2 the script randomly takes 10 images with at least 3 object classes and at most 12 objects.
  For task 3 it takes 4 images that contain both a remote and a cell phone and at least 5 classes in total.
  Images with "crowd" annotations are skipped. The random seed is fixed (`SEED = 42`), so every run selects the same images.
- **Matching.** A detection is a true positive (TP) when its class matches an annotated object and their
  IoU is at least 0.5. Each annotated object can be matched once. Unmatched detections are FP, unmatched
  annotated objects are FN. Then p = TP / (TP + FP), r = TP / (TP + FN).
- **Class filter (task 3).** Ultralytics filters by class id: `model.predict(..., classes=[65, 67])`.
  Class ids start from 0, so they are one less than the numbers in appendix 2.1 of the assignment.

## Settings

All settings are constants at the top of `lab4.py`:

| Constant | Default | Meaning |
|---|---|---|
| `MODEL` | `yolo26m.pt` | Model size: `yolo26n/s/m/l/x.pt`, from fastest to most accurate |
| `IMAGE_SIZE` | `640` | Network input size, must be a multiple of 32 |
| `TASK1_CONF` | `0.5` | Confidence threshold for task 1 |
| `THRESHOLDS` | `[0.3, 0.5, 0.7, 0.9]` | Confidence thresholds for task 2 |
| `IOU_MATCH` | `0.5` | Minimum IoU for a correct detection |
| `NUM_IMAGES` | `10` | Number of images for tasks 1-2 |
| `SEED` | `42` | Random seed for image selection |
| `VARIANT`, `VARIANT_CLASSES` | `21`, `["remote", "cell phone"]` | Variant number and its COCO class names |

## Links

- Ultralytics YOLO26 documentation: https://docs.ultralytics.com/models/yolo26/
- Ultralytics predict mode (arguments `conf`, `classes`, `imgsz`): https://docs.ultralytics.com/modes/predict/
- Ultralytics source code: https://github.com/ultralytics/ultralytics
- YOLO26 announcement: https://www.ultralytics.com/blog/meet-ultralytics-yolo26-a-better-faster-smaller-yolo-model
- Model weights `yolo26m.pt`: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt
- MS COCO dataset: https://cocodataset.org/
- COCO annotations 2017: http://images.cocodataset.org/annotations/annotations_trainval2017.zip
- COCO val2017 images (all 5000, 1 GB; the script downloads only the 14 it needs): http://images.cocodataset.org/zips/val2017.zip
- COCO annotation format: https://cocodataset.org/#format-data
- YOLOv4 from the assignment, for comparison: https://github.com/AlexeyAB/darknet

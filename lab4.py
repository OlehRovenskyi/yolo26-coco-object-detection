"""Lab 4: object detection on MS COCO images with YOLO26.

Task 1: detect objects of all 80 COCO classes with the YOLO26 model (Ultralytics).
Task 2: precision (p) and recall (r) on 10 COCO val2017 images for confidence thresholds 0.3..0.9.
Task 3: detect only the classes of variant 21 (remote and cell phone).
"""

import json
import random
import urllib.request
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "data" / "val2017"
ANNOTATIONS = ROOT / "data" / "annotations" / "instances_val2017.json"
RESULTS = ROOT / "results"

MODEL = "yolo26m.pt"
IMAGE_SIZE = 640
TASK1_CONF = 0.5
THRESHOLDS = [0.3, 0.5, 0.7, 0.9]
IOU_MATCH = 0.5
NUM_IMAGES = 10
SEED = 42

VARIANT = 21
VARIANT_CLASSES = ["remote", "cell phone"]
VARIANT_IMAGES = 4

TP_COLOR, FP_COLOR, FN_COLOR = (255, 128, 0), (0, 0, 255), (0, 200, 255)


def load_coco():
    """Returns image info by id and, for each image, its objects as {name, box} plus a crowd flag."""
    with open(ANNOTATIONS) as f:
        coco = json.load(f)
    names = {c["id"]: c["name"] for c in coco["categories"]}
    images = {img["id"]: img for img in coco["images"]}
    objects, crowd = defaultdict(list), set()
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        objects[a["image_id"]].append({"name": names[a["category_id"]], "box": [x, y, x + w, y + h]})
        if a["iscrowd"]:
            crowd.add(a["image_id"])
    return images, objects, crowd


def select(objects, crowd, condition, count):
    # "crowd" annotations cover a group of objects with one box, so they can't be matched fairly
    candidates = sorted(i for i, objs in objects.items() if i not in crowd and condition(objs))
    return random.Random(SEED).sample(candidates, count)


def classes_of(objs):
    return {o["name"] for o in objs}


def download(images, image_ids):
    IMAGES.mkdir(parents=True, exist_ok=True)
    for image_id in image_ids:
        path = IMAGES / images[image_id]["file_name"]
        if not path.exists():
            urllib.request.urlretrieve(images[image_id]["coco_url"], path)
    return {i: IMAGES / images[i]["file_name"] for i in image_ids}


def predict(model, path, conf, classes=None):
    result = model.predict(str(path), conf=conf, classes=classes, imgsz=IMAGE_SIZE, verbose=False)[0]
    boxes = [
        {"name": model.names[int(c)], "conf": float(s), "box": b.tolist()}
        for b, c, s in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf)
    ]
    return result, boxes


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(predictions, truth):
    """Greedy matching: a box is TP if an unmatched ground-truth object of the same class has IoU >= 0.5."""
    used, tp, fp = set(), [], []
    for p in sorted(predictions, key=lambda p: -p["conf"]):
        candidates = [(iou(p["box"], t["box"]), i) for i, t in enumerate(truth)
                      if i not in used and t["name"] == p["name"]]
        best_iou, best = max(candidates, default=(0.0, None))
        if best_iou >= IOU_MATCH:
            used.add(best)
            tp.append(p)
        else:
            fp.append(p)
    return tp, fp, [t for i, t in enumerate(truth) if i not in used]


def metrics(tp, fp, fn):
    """p, r and F1; None where the denominator is zero."""
    def div(a, b):
        return a / b if b else None
    return div(tp, tp + fp), div(tp, tp + fn), div(2 * tp, 2 * tp + fp + fn)


def fmt(value):
    return "-" if value is None else f"{value:.2f}"


def table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
           ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]


def describe(items):
    return "; ".join(
        i["name"] + (f" ({i['conf']:.2f})" if "conf" in i else "")
        + f", {int(i['box'][2] - i['box'][0])}×{int(i['box'][3] - i['box'][1])} px"
        for i in items
    ) or "none"


def draw_matches(path, out, tp, fp, fn):
    image = cv2.imread(str(path))
    for items, color, prefix in [(tp, TP_COLOR, ""), (fp, FP_COLOR, "FP "), (fn, FN_COLOR, "FN ")]:
        for item in items:
            x1, y1, x2, y2 = (int(v) for v in item["box"])
            label = prefix + item["name"] + (f" {item['conf']:.2f}" if "conf" in item else "")
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imwrite(str(out), image)


def write(name, lines):
    (RESULTS / name).write_text("\n".join(lines) + "\n")


def task1(model, images, paths, image_ids):
    out_dir = RESULTS / "task1"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Task 1. Object detection with YOLO26", "",
             f"Model `{MODEL}` (Ultralytics, trained on MS COCO, {len(model.names)} classes), "
             f"input size {IMAGE_SIZE}×{IMAGE_SIZE}, confidence threshold {TASK1_CONF}.", "",
             "## Detected objects (class, confidence, box size)", ""]
    for n, image_id in enumerate(image_ids, start=1):
        result, found = predict(model, paths[image_id], TASK1_CONF)
        cv2.imwrite(str(out_dir / f"{n:02d}_{image_id}.jpg"), result.plot())
        lines.append(f"- Image {n} (`{images[image_id]['file_name']}`, {len(found)} objects): {describe(found)}")
    write("task1_report.md", lines)


def task2(model, images, objects, paths, image_ids):
    """Returns the best threshold by F1 over all images."""
    stats = {}  # (image number, threshold) -> (tp, fp, fn) lists
    for n, image_id in enumerate(image_ids, start=1):
        _, predictions = predict(model, paths[image_id], min(THRESHOLDS))
        for th in THRESHOLDS:
            out_dir = RESULTS / "task2" / f"conf_{th}"
            out_dir.mkdir(parents=True, exist_ok=True)
            stats[n, th] = match([p for p in predictions if p["conf"] >= th], objects[image_id])
            draw_matches(paths[image_id], out_dir / f"{n:02d}_{image_id}.jpg", *stats[n, th])

    def counts(n, th):
        return [len(x) for x in stats[n, th]]

    numbers = range(1, len(image_ids) + 1)
    totals = {th: [sum(c) for c in zip(*(counts(n, th) for n in numbers))] for th in THRESHOLDS}
    best = max(THRESHOLDS, key=lambda th: metrics(*totals[th])[2] or 0)
    worst = sorted(numbers, key=lambda n: metrics(*counts(n, best))[2] or 0)[:3]

    lines = ["# Task 2. Precision and recall of the YOLO26 detector", "", "## Images", "",
             *table(["#", "COCO file", "Objects", "Classes"],
                    [[n, images[i]["file_name"], len(objects[i]), ", ".join(sorted(classes_of(objects[i])))]
                     for n, i in zip(numbers, image_ids)]),
             "", "## p / r table", "",
             *table(["Confidence"] + [f"{n} {m}" for n in numbers for m in ("p", "r")],
                    [[th] + [fmt(v) for n in numbers for v in metrics(*counts(n, th))[:2]] for th in THRESHOLDS]),
             "", f"## Totals over all {len(image_ids)} images", "",
             *table(["Confidence", "TP", "FP", "FN", "p", "r", "F1"],
                    [[th, *totals[th], *map(fmt, metrics(*totals[th]))] for th in THRESHOLDS]),
             "", f"Best threshold by F1: **{best}**", "", f"## Three worst results (threshold {best})", ""]
    for n in worst:
        image_id = image_ids[n - 1]
        tp, fp, fn = stats[n, best]
        lines += [f"### Image {n} ({images[image_id]['file_name']})", "",
                  f"File: `results/task2/conf_{best}/{n:02d}_{image_id}.jpg`", "",
                  f"- TP: {len(tp)}, FP: {len(fp)}, FN: {len(fn)}",
                  f"- False detections (FP): {describe(fp)}",
                  f"- Missed objects (FN): {describe(fn)}", ""]
    lines.append("Colors on the images: blue is a correct detection (TP), red is a false detection (FP), "
                 "yellow is a missed object from the COCO annotation (FN).")
    write("task2_report.md", lines)

    for th in THRESHOLDS:
        p, r, f1 = metrics(*totals[th])
        print(f"conf={th}: p={fmt(p)} r={fmt(r)} F1={fmt(f1)}")
    return best


def task3(model, images, objects, paths, image_ids, conf):
    out_dir = RESULTS / "task3"
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = [k for k, v in model.names.items() if v in VARIANT_CLASSES]
    lines = [f"# Task 3. Variant {VARIANT}: {' and '.join(VARIANT_CLASSES)}", "",
             f"Model class ids (numbered from 0): {ids}. Confidence threshold: {conf}.", ""]
    for n, image_id in enumerate(image_ids, start=1):
        result_all, found_all = predict(model, paths[image_id], conf)
        result_variant, found_variant = predict(model, paths[image_id], conf, classes=ids)
        cv2.imwrite(str(out_dir / f"{n:02d}_all_classes.jpg"), result_all.plot())
        cv2.imwrite(str(out_dir / f"{n:02d}_variant_only.jpg"), result_variant.plot())
        truth = objects[image_id]
        lines += [f"## Image {n} ({images[image_id]['file_name']})", "",
                  f"- Classes in the COCO annotation ({len(classes_of(truth))}): {', '.join(sorted(classes_of(truth)))}",
                  f"- Annotated objects of the variant classes: {describe(o for o in truth if o['name'] in VARIANT_CLASSES)}",
                  f"- All detections without the filter ({len(found_all)}): {describe(found_all)}",
                  f"- Detections with the variant filter ({len(found_variant)}): {describe(found_variant)}",
                  f"- Files: `results/task3/{n:02d}_all_classes.jpg`, `results/task3/{n:02d}_variant_only.jpg`", ""]
    write("task3_report.md", lines)


def main():
    RESULTS.mkdir(exist_ok=True)
    images, objects, crowd = load_coco()
    task2_ids = select(objects, crowd, lambda objs: len(classes_of(objs)) >= 3 and len(objs) <= 12, NUM_IMAGES)
    task3_ids = select(objects, crowd, lambda objs: set(VARIANT_CLASSES) <= classes_of(objs)
                       and len(classes_of(objs)) >= 5, VARIANT_IMAGES)
    paths = download(images, task2_ids + task3_ids)

    model = YOLO(MODEL)
    task1(model, images, paths, task2_ids)
    best = task2(model, images, objects, paths, task2_ids)
    task3(model, images, objects, paths, task3_ids, best)
    print(f"Best threshold: {best}. Reports: {RESULTS}")


if __name__ == "__main__":
    main()

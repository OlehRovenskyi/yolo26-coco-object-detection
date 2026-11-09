# Task 2. Precision and recall of the YOLO26 detector

## Images

| # | COCO file | Objects | Classes |
|---|---|---|---|
| 1 | 000000476810.jpg | 9 | bed, book, cat, remote, tv |
| 2 | 000000081394.jpg | 6 | car, motorcycle, person, truck |
| 3 | 000000016598.jpg | 3 | cell phone, person, tie |
| 4 | 000000562059.jpg | 6 | apple, bowl, cup, dining table, spoon |
| 5 | 000000211120.jpg | 6 | bowl, teddy bear, truck |
| 6 | 000000182417.jpg | 8 | bowl, cake, cup, dining table, spoon |
| 7 | 000000170099.jpg | 3 | couch, person, tie |
| 8 | 000000099054.jpg | 6 | airplane, person, truck |
| 9 | 000000559513.jpg | 5 | bottle, dining table, fork, knife |
| 10 | 000000072852.jpg | 4 | bicycle, handbag, person |

## p / r table

| Confidence | 1 p | 1 r | 2 p | 2 r | 3 p | 3 r | 4 p | 4 r | 5 p | 5 r | 6 p | 6 r | 7 p | 7 r | 8 p | 8 r | 9 p | 9 r | 10 p | 10 r |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.3 | 0.75 | 0.33 | 0.80 | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 0.67 | 1.00 | 0.71 | 0.62 | 1.00 | 1.00 | 1.00 | 0.83 | 0.33 | 0.40 | 0.75 | 0.75 |
| 0.5 | 1.00 | 0.22 | 0.75 | 0.50 | 1.00 | 1.00 | 1.00 | 0.83 | 0.83 | 0.83 | 0.80 | 0.50 | 1.00 | 1.00 | 1.00 | 0.67 | 0.67 | 0.40 | 1.00 | 0.75 |
| 0.7 | 1.00 | 0.22 | 1.00 | 0.17 | 1.00 | 0.67 | 1.00 | 0.83 | 1.00 | 0.83 | 1.00 | 0.38 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 0.40 | 1.00 | 0.75 |
| 0.9 | 1.00 | 0.22 | 1.00 | 0.17 | 1.00 | 0.33 | 1.00 | 0.67 | 1.00 | 0.50 | 1.00 | 0.12 | 1.00 | 1.00 | – | 0.00 | – | 0.00 | 1.00 | 0.25 |

## Totals over all 10 images

| Confidence | TP | FP | FN | p | r | F1 |
|---|---|---|---|---|---|---|
| 0.3 | 40 | 12 | 16 | 0.77 | 0.71 | 0.74 |
| 0.5 | 34 | 4 | 22 | 0.89 | 0.61 | 0.72 |
| 0.7 | 29 | 0 | 27 | 1.00 | 0.52 | 0.68 |
| 0.9 | 16 | 0 | 40 | 1.00 | 0.29 | 0.44 |

Best threshold by F1: **0.3**

## Three worst results (threshold 0.3)

### Image 9 (000000559513.jpg)

File: `results/task2/conf_0.3/09_559513.jpg`

- TP: 2, FP: 4, FN: 3
- False detections (FP): cup (0.54), 78×173 px; potted plant (0.48), 186×183 px; cup (0.45), 98×107 px; dining table (0.37), 637×421 px
- Missed objects (FN): bottle, 77×162 px; knife, 209×311 px; dining table, 80×174 px

### Image 1 (000000476810.jpg)

File: `results/task2/conf_0.3/01_476810.jpg`

- TP: 3, FP: 1, FN: 6
- False detections (FP): book (0.39), 18×135 px
- Missed objects (FN): book, 63×140 px; tv, 67×71 px; book, 10×108 px; book, 7×124 px; book, 34×139 px; book, 12×107 px

### Image 6 (000000182417.jpg)

File: `results/task2/conf_0.3/06_182417.jpg`

- TP: 5, FP: 2, FN: 3
- False detections (FP): bottle (0.62), 122×169 px; potted plant (0.36), 251×351 px
- Missed objects (FN): cake, 156×93 px; cup, 122×179 px; bowl, 48×83 px

Colors on the images: blue is a correct detection (TP), red is a false detection (FP), yellow is a missed object from the COCO annotation (FN).

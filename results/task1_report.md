# Task 1. Object detection with YOLO26

- Model: `yolo26m.pt` (Ultralytics, trained on MS COCO, 80 classes)
- Input size: 640×640 (a multiple of 32)
- Device: cpu
- Confidence threshold: 0.5

| # | COCO file | Size | Objects | Time, ms (preprocess / inference / postprocess) |
|---|---|---|---|---|
| 1 | 000000476810.jpg | 640×480 | 2 | 0.5 / 136.2 / 0.6 |
| 2 | 000000081394.jpg | 640×480 | 4 | 0.6 / 142.7 / 0.5 |
| 3 | 000000016598.jpg | 478×640 | 3 | 0.5 / 133.2 / 0.6 |
| 4 | 000000562059.jpg | 427×640 | 5 | 0.5 / 143.2 / 0.6 |
| 5 | 000000211120.jpg | 640×379 | 6 | 0.8 / 130.8 / 0.6 |
| 6 | 000000182417.jpg | 640×480 | 5 | 0.5 / 134.1 / 0.7 |
| 7 | 000000170099.jpg | 640×478 | 3 | 0.6 / 126.8 / 0.5 |
| 8 | 000000099054.jpg | 427×640 | 4 | 0.5 / 149.4 / 0.6 |
| 9 | 000000559513.jpg | 640×421 | 3 | 0.8 / 189.3 / 0.5 |
| 10 | 000000072852.jpg | 424×640 | 3 | 0.5 / 170.5 / 0.6 |

## Detected objects (class, confidence, box size)

### Image 1 (000000476810.jpg)

File: `results/task1/01_476810.jpg`

remote (0.95), 298×71 px; cat (0.93), 509×301 px

### Image 2 (000000081394.jpg)

File: `results/task1/02_81394.jpg`

truck (0.97), 428×415 px; motorcycle (0.66), 50×46 px; motorcycle (0.58), 22×47 px; person (0.56), 19×31 px

### Image 3 (000000016598.jpg)

File: `results/task1/03_16598.jpg`

person (0.95), 477×624 px; cell phone (0.86), 73×116 px; tie (0.52), 115×335 px

### Image 4 (000000562059.jpg)

File: `results/task1/04_562059.jpg`

spoon (0.98), 224×170 px; cup (0.95), 194×237 px; apple (0.94), 146×150 px; bowl (0.91), 307×300 px; apple (0.90), 152×118 px

### Image 5 (000000211120.jpg)

File: `results/task1/05_211120.jpg`

teddy bear (0.95), 177×223 px; teddy bear (0.94), 156×194 px; teddy bear (0.94), 154×215 px; teddy bear (0.82), 103×70 px; bowl (0.74), 56×39 px; car (0.68), 628×146 px

### Image 6 (000000182417.jpg)

File: `results/task1/06_182417.jpg`

spoon (0.94), 187×182 px; dining table (0.87), 627×394 px; spoon (0.82), 29×38 px; bottle (0.62), 122×169 px; bowl (0.54), 101×76 px

### Image 7 (000000170099.jpg)

File: `results/task1/07_170099.jpg`

tie (0.95), 93×273 px; person (0.94), 350×442 px; couch (0.91), 300×233 px

### Image 8 (000000099054.jpg)

File: `results/task1/08_99054.jpg`

person (0.84), 28×76 px; airplane (0.84), 427×497 px; person (0.72), 16×26 px; person (0.66), 44×55 px

### Image 9 (000000559513.jpg)

File: `results/task1/09_559513.jpg`

fork (0.88), 282×72 px; bottle (0.72), 112×145 px; cup (0.54), 78×173 px

### Image 10 (000000072852.jpg)

File: `results/task1/10_72852.jpg`

bicycle (0.91), 156×151 px; person (0.89), 36×77 px; person (0.86), 28×78 px


# Dataset

Licence-plate data for training/fine-tuning. There are two independent sources,
each with its own `data.yaml`.

## Layout

```
data/
├── custom/                  self-collected images (fine-tuning source)
│   ├── images/
│   │   ├── train/   518 .png
│   │   ├── val/     148 .png
│   │   └── test/     74 .png
│   ├── labels/
│   │   ├── train/   (empty)
│   │   ├── val/     (empty)
│   │   └── test/    (empty)
│   └── data.yaml            nc=1, names=[license_plate]
├── kaggle/                  dataset downloaded from Kaggle
│   ├── images/
│   │   ├── train/  3433 .png
│   │   └── val/    1145 .png
│   ├── labels/              YOLO bbox, SINGLE class 0 = license_plate
│   │   ├── train/  3433 .txt   (3887 objects)
│   │   └── val/    1145 .txt   (1313 objects)
│   ├── labels_polygon/      ORIGINAL polygon labels (backup, source of truth)
│   │   ├── train/  3433 .txt
│   │   └── val/    1145 .txt
│   ├── labels_2class_backup/  bbox labels BEFORE the 1-class merge
│   │   ├── train/  3433 .txt
│   │   └── val/    1145 .txt
│   └── data.yaml            nc=1, names=[license_plate]
└── classes.txt
```

## Dataset configs

| File | task | train | val | test | nc | classes |
|---|---|---|---|---|---|---|
| `custom/data.yaml` | detection | 518 | 148 | 74 | 1 | license_plate |
| `kaggle/data.yaml` | detection | 3433 | 1145 | — | 1 | 0=license_plate (both source classes merged) |

Paths in each `data.yaml` are relative to that file's directory (Ultralytics
resolves them).

## Label format

Detection (5-value bbox), one line per object:

```
<class_id> <x_center> <y_center> <width> <height>   # normalised 0-1
```

- **`kaggle/labels/`** holds bbox labels, **converted from the original polygon
  labels** via axis-aligned bounding box of each polygon, then **merged to a
  single class** (`0 = license_plate`). Original polygons remain untouched in
  `kaggle/labels_polygon/`; the pre-merge bbox labels are in
  `kaggle/labels_2class_backup/`.
- **`custom/labels/`** is empty (images are not annotated yet).

## Polygon -> bbox conversion

The Kaggle labels were originally YOLO segmentation polygons
(`<class> <x1> <y1> ... <xn> <yn>`, 4-6 points). They were converted with:

```
xmin = min(xs), xmax = max(xs), ymin = min(ys), ymax = max(ys)
cx = (xmin+xmax)/2, cy = (ymin+ymax)/2, w = xmax-xmin, h = ymax-ymin
```

- Coordinates were clamped to `[0, 1]` (6 polygons were slightly out of range).
- No degenerate boxes were produced (min w = 0.0059, min h = 0.0089).
- Class ids were preserved during conversion, then **merged to one class**:
  visualisation showed the original split was inconsistent (`0` = car plate
  everywhere, but `1` = motorcycle plate in some sources and 2-line car plate in
  others). All object ids were remapped to `0` (`license_plate`).
- Pre-merge bbox labels are kept in `labels_2class_backup/`.
- To re-run the conversion, the source dir is always `labels_polygon/`.

## Naming

Ultralytics matches a label to an image by replacing the *last* extension of
the image with `.txt`, e.g. `carlong_0001.png` -> `carlong_0001.txt`.

## Known issues (NOT fixed here)

- **`custom/labels/*` are empty** — the `custom` split cannot be trained until
  annotations are added (fine-tuning deferred).
- **`kaggle` has no `test` split.**
- Ultralytics `*.cache` files are regenerated automatically on the next
  training run and are git-ignored.
- The old 2-class `weights/best.pt` must be retrained against this single-class
  dataset (`nc=1`); the current two-class checkpoint no longer matches.

## Status

This is the *current* data, reorganised as-is. It is **not** declared the final
dataset of the project.

# Báo cáo quét & phân tích source code — `D:\CCNLTHD-main`

> Tài liệu lịch sử: phản ánh trạng thái **trước khi refactor**. Đường dẫn
> tuyệt đối trong báo cáo là của trạng thái cũ và đã được thay bằng đường dẫn
> tương đối trong cấu trúc mới (xem `docs/architecture.md`).

> Phạm vi: chỉ đọc/phân tích. **Không sửa, không xóa, không đổi tên, không di chuyển, không refactor, không commit.** Không có `.git` nên không có Git metadata.

---

## 1. Current Structure

`D:\CCNLTHD-main` **không phải git repo** (không có `.git`). Có 2 cây nội dung: app đang chạy ở root và một bundle artifact train **trùng tên** nằm lồng bên trong.

```
D:\CCNLTHD-main\
├─ be\main.py                    2.6 KB   Backend FastAPI (YOLO + EasyOCR)
├─ fe\index.html                 7.1 KB   Frontend (Tailwind CDN, gọi API)
├─ notebooks\test123.ipynb       937 KB   Notebook train + inference prototype (4 cells)
├─ data\                         162 MB   Dataset YOLO + config
│  ├─ dataset.yaml               143 B    train/val trỏ path cũ (chết)
│  ├─ classes.txt                13 B     "license_plate"
│  ├─ images\{train=1526, val=169}\  (chỉ .jpg)
│  └─ labels\{train=1526, val=169}\  (.txt) + train.cache, val.cache  <- cache sinh tự động
├─ requirements_1.txt            78 B     Deps runtime (unpinned, 7 dòng)
├─ how_to_run.txt                162 B    Hướng dẫn chạy 3 bước
├─ src\{inference, training, utils}\      EMPTY (chỉ có thư mục rỗng)
├─ tests\                                  EMPTY
├─ assets\                                 EMPTY
├─ weights\                                EMPTY
├─ .kilo\                                 .gitignore + worktrees\ (rỗng)
└─ CCNLTHD-main\                 53.9 MB  Bundle gốc (trùng tên)
   ├─ requirements.txt           6.6 KB   Env frozen 176 dòng (có dòng lặp lại)
   ├─ .gitignore                 "archive.rar"
   ├─ 1.jpg, Biển_số_xe_đồng_tháp.jpg, OIP.webp, z7659940505816_....jpg   Ảnh mẫu test
   └─ runs\detect\LPR_Project\
      ├─ train_bien_so\    args.yaml (không weights/results)
      ├─ train_bien_so2..4\ args.yaml (không weights/results)
      ├─ train_bien_so5\   args, results.csv, labels.jpg, train_batch0-2.jpg, weights\best.pt+last.pt (18.4 MB each)
      └─ train_bien_so6\   args, results.csv, results.png, curves, confusion_matrix, batches, val_*, weights\best.pt+last.pt (6.2 MB each)
```

---

## 2. Core Files & Dependencies

Chuỗi phụ thuộc thực tế: `fe/index.html` -> `POST http://127.0.0.1:8000/detect` -> `be/main.py`.

| File | Vai trò | Import / phụ thuộc | Hard-coded path |
| --- | --- | --- | --- |
| `be/main.py` | Backend API duy nhất | `fastapi`, `fastapi.middleware.cors`, `ultralytics.YOLO`, `cv2`, `numpy`, `easyocr`, `uvicorn` | `be/main.py:19` model path; `main.py:65` host/port |
| `fe/index.html:83` | UI upload + gọi API | Tailwind CDN, `fetch` | `http://127.0.0.1:8000/detect` |
| `notebooks/test123.ipynb` | Train + inference + class `LicensePlateReader` | ultralytics, cv2, easyocr, re, numpy, matplotlib | `D:/CCNLTHD/...` (nhiều nơi, đã chết) |
| `data/dataset.yaml:1-2` | Config train YOLO | — | `D:/CCNLTHD/archive/archive/images/...` (chết) |
| `how_to_run.txt` | Hướng dẫn | dùng `requirements_1.txt`, `uvicorn main:app --reload` trong `be\`, Live Server cho `fe\` | — |
| `requirements_1.txt` | Deps runtime (7, không pin) | fastapi, uvicorn, python-multipart, ultralytics, easyocr, opencv-python, numpy | — |
| `CCNLTHD-main/requirements.txt` | Env frozen đầy đủ (176 dòng, **nhiều dòng trùng**) | torch 2.10 cu130, paddleocr, vietocr, easyocr… | — |

**Phát hiện quan trọng:**

- `be/main.py:19` trỏ `D:\DO_AN_CHUYEN_NGANH\CCNLTHD-main\...\train_bien_so6\weights\last.pt` — **thư mục `D:\DO_AN_CHUYEN_NGANH` không tồn tại** -> app sẽ crash khi import. Weight thật đang ở `D:\CCNLTHD-main\CCNLTHD-main\runs\detect\LPR_Project\train_bien_so6\weights\last.pt`.
- `D:\CCNLTHD` (root mà mọi path trong notebook/`args.yaml`/`dataset.yaml` trỏ tới) **không tồn tại**.
- `src/`, `tests/`, `weights/`, `assets/` **rỗng** -> cấu trúc dự kiến nhưng chưa dùng (**NEEDS MANUAL REVIEW**).

---

## 3. Dataset / YOLO / Model Analysis

**Dataset (`data\`)**

- 1526 ảnh train + 1526 label, 169 ảnh val + 169 label, chỉ `.jpg`.
- **Mapping 1:1 hoàn hảo, 0 ảnh mồ côi, 0 label mồ côi** ở cả train và val.
- Format label: `0 cx cy w h` (1 class, id = 0 = `license_plate`).
- 2 quy ước tên label — **cả hai đều hợp lệ với Ultralytics** (`rsplit(".",1)[0] + ".txt"`): dạng `file.jpg.txt` (ảnh `file.jpg.jpg`) và dạng `video4_3860.txt` (ảnh `video4_3860.jpg`).
- **12 cặp ảnh+label trùng** trong train: `... - Copy`, `- Copy (2)`, `- Copy (3)` -> mẫu trùng lặp.
- `data\labels\train.cache`, `val.cache` -> **cache Ultralytics tự sinh**, tái tạo được.

**YOLO training (6 runs trong `CCNLTHD-main\runs\detect\LPR_Project\`)**

| Run | data path | Trạng thái |
| --- | --- | --- |
| `train_bien_so` | `D:/CCNLTHD/thu_muc_dataset/data.yaml` | chỉ `args.yaml` |
| `train_bien_so2/3/4` | `D:/CCNLTHD/archive/archive/dataset.yaml` | chỉ `args.yaml` (dở dang) |
| `train_bien_so5` | như trên | `results.csv` mới 1 epoch; best/last.pt **18.4 MB (chưa strip optimizer)** -> **run bị ngắt** |
| `train_bien_so6` | như trên | **Hoàn tất 50/50 epoch** |

- Cấu hình chung: `yolov8n.pt`, `nc=1`, `imgsz=640`, `epochs=50`, `batch=8`, `device=cpu`, optimizer auto (AdamW).
- **Kết quả `train_bien_so6` (best):** P=0.982, R=0.994, **mAP50=0.994, mAP50-95=0.881**; `best.pt`/`last.pt` = 6.2 MB. Đây là model đang được `be/main.py` và notebook dùng.
- `args.yaml` ghi `save_dir = D:\CCNLTHD\runs\...` (root khác hiện tại).

**Preprocessing / Inference — 2 pipeline khác nhau (divergent):**

- `be/main.py:37-56`: crop -> BGR->Gray -> resize x2 -> **Otsu threshold** -> EasyOCR `allowlist='0-9A-Z-'` -> join text, trả `confidence`.
- `notebooks/test123.ipynb` (cell 3, class `LicensePlateReader`): crop -> RGB->Gray -> resize x2 -> **bilateralFilter -> CLAHE** -> EasyOCR + **hậu xử lý biển số VN** (`format_plate_vietnam`) -> trả `(text, avg_conf)`. Kết quả mẫu trong notebook: `63-B 35682` @ 76.43%.
- Cell 2 chỉ visualize bounding box.
- Backend chạy `easyocr.Reader(['en'], gpu=False)`; notebook yêu cầu model path `D:/CCNLTHD/...` (chết).

---

## 4. Files to KEEP

- `be/main.py`, `fe/index.html`, `notebooks/test123.ipynb`, `how_to_run.txt`, `requirements_1.txt`.
- `data/dataset.yaml`, `data/classes.txt`, toàn bộ `data/images/**`, `data/labels/**` (trừ 2 file `.cache`).
- `CCNLTHD-main/runs/detect/LPR_Project/train_bien_so6/**` (đặc biệt `weights/best.pt`, `weights/last.pt`, `results.csv`, `args.yaml`, plots).
- `CCNLTHD-main/requirements.txt` — giữ như **tham chiếu môi trường** (nên dedupe).
- Ảnh mẫu test: `CCNLTHD-main/1.jpg`, `Biển_số_xe_đồng_tháp.jpg`, `OIP.webp`, `z7659940505816_....jpg`.

---

## 5. Files to MOVE + proposed location

Không di chuyển ở bước này; đây là **đề xuất** cho refactor sau:

| Nguồn | Đích đề xuất | Ghi chú |
| --- | --- | --- |
| `CCNLTHD-main\runs\` | `runs\` (root) | Flatten bundle trùng tên |
| `CCNLTHD-main\runs\...\train_bien_so6\weights\best.pt` | `weights\best.pt` (root `weights\` đang rỗng) | Cập nhật `be/main.py:19` |
| Ảnh mẫu trong `CCNLTHD-main\*.jpg/webp` | `assets\` (đang rỗng) | Hoặc `data\samples\` |
| `CCNLTHD-main\requirements.txt` | `docs\` hoặc `requirements-full.txt` | Hoặc gộp vào `requirements_1.txt` |
| `be\main.py` -> `src\inference\` | `src\` (đang rỗng) | **NEEDS MANUAL REVIEW** — `how_to_run.txt` hướng dẫn chạy `uvicorn main:app` trong `be\`, di chuyển sẽ phá hướng dẫn |

---

## 6. Files to REMOVE CANDIDATES + reason

| Ứng viên | Lý do |
| --- | --- |
| `data\labels\train.cache`, `data\labels\val.cache` | Cache Ultralytics sinh tự động, tái tạo khi train |
| `runs\...\train_bien_so`, `so2`, `so3`, `so4` | Chỉ có `args.yaml`, không weights/results -> run dở dang |
| `runs\...\train_bien_so5\` | Bị `train_bien_so6` thay thế hoàn toàn; 1-epoch csv + ~37 MB weights chưa strip |
| 12 cặp `... - Copy*` (ảnh + label) trong `data\train` | Mẫu huấn luyện trùng lặp |
| Các dòng lặp trong `CCNLTHD-main\requirements.txt` | Trùng lặp vô nghĩa khi giữ file |
| `CCNLTHD-main\` (vỏ bundle sau khi hợp nhất) | Chỉ còn artifact trùng tên — **NEEDS MANUAL REVIEW** trước khi xóa |

---

## 7. Risks / Unknowns

- **Path chết (blocking runtime):** `be/main.py:19` và `data/dataset.yaml:1-2` đều trỏ đường dẫn không tồn tại -> app không chạy / không train lại được nếu không sửa.
- **Không có git** -> xóa artifact là không thể hoàn tác; nên backup trước.
- **Trùng tên thư mục** `CCNLTHD-main\CCNLTHD-main` dễ gây nhầm lẫn tooling/build.
- **`src/`, `tests/`, `weights/`, `assets/` rỗng** — không rõ có phải cấu trúc dự kiến hay tàn dư -> **NEEDS MANUAL REVIEW**.
- **Hai requirements khác nhau** (`requirements_1` dùng easyocr; bản full dùng paddleocr/vietocr/torch cu130) -> cần chốt một nguồn.
- **12 cặp `- Copy`**: không xác nhận được có chủ đích (augmentation/tăng mẫu) hay lỗi copy -> **NEEDS MANUAL REVIEW**.
- Không thể xác nhận `train_bien_so5` có phải bản dự phòng quan trọng không -> **NEEDS MANUAL REVIEW** trước khi xóa.

---

## 8. Recommended Refactor Order

1. **Backup** toàn bộ `D:\CCNLTHD-main` (trước mọi thao tác, vì không có git).
2. **Sửa path** (ưu tiên cao nhất): chuyển `be/main.py:19` và `data/dataset.yaml:1-2` sang path tương đối; chạy `uvicorn main:app --reload` trong `be\` + mở `fe\index.html` để xác nhận end-to-end.
3. **Hợp nhất weight:** đưa model `train_bien_so6` vào `weights\` ở root, cập nhật tham chiếu.
4. **Flatten** `CCNLTHD-main\runs\` -> `runs\`; loại vỏ thư mục trùng tên sau khi xác nhận đủ artifact.
5. **Dọn remove candidates:** xóa `.cache`; xóa `train_bien_so/so2/so3/so4`; quyết định `so5` và các cặp `- Copy`.
6. **Thống nhất dependencies:** dedupe `CCNLTHD-main\requirements.txt`, chọn một file requirements duy nhất.
7. **Chỉ sau khi chốt thủ công:** tổ chức lại code vào `src\inference|training|utils` (hiện rỗng).
8. **Kiểm thử lại** toàn bộ luồng sau refactor (backend + frontend + ảnh mẫu).

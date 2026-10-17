"""Package the existing two-image traces; does not run inference."""
import csv
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
from compare_alpr_cases import OUT,ROOT,load,dump,sha
import cv2
import numpy as np

rows=[]
variants=[]
for case in ['A','B']:
    trace=load(OUT/(case+'-lazy')/'trace.json')
    for det in trace['traces']:
        details=det.get('details',{})
        variants.append((case+' production det-'+str(det['index']),details))
    for folder in [OUT/f'{case}-geometry-0',OUT/f'{case}-geometry-1',OUT/f'{case}-panel']:
        if not (folder/'trace.json').exists():continue
        study=load(folder/'trace.json')
        for name,value in study['variants'].items():
            details=value.get('details',{})
            variants.append((folder.name+'/'+name,details))
            pixels=cv2.imread(str(folder/value['input']),0)
            split_row=details.get('split_row')
            if split_row is not None:
                cv2.imwrite(str(folder/(name+'-top.png')),pixels[:split_row])
                cv2.imwrite(str(folder/(name+'-bottom.png')),pixels[split_row:])
            for i,call in enumerate(details.get('ocr_calls',[])):
                source=pixels
                if call['label']=='top':source=pixels[:split_row]
                if call['label']=='bottom':source=pixels[split_row:]
                canvas=cv2.cvtColor(source,cv2.COLOR_GRAY2BGR)
                for j,item in enumerate(call['results']):
                    pts=np.int32(item['bbox']);cv2.polylines(canvas,[pts],True,(0,0,255),2)
                    x,y=pts.min(axis=0)
                    cv2.putText(canvas,f"{j}: {item['text']} {item['confidence']:.4f}",(max(0,int(x)),max(20,int(y))),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)
                cv2.imwrite(str(folder/f'{name}-{call["label"]}-boxes.jpg'),canvas)
    # Readable stage overview, exported only; full resolution PNGs remain the evidence.
    for det in trace['traces']:
        tiles=[]
        for stage,ref in det['stages'].items():
            im=cv2.imread(str(OUT/ref['path']));h,w=im.shape[:2]
            im=cv2.resize(im,(400,round(h*400/w)))
            tile=np.full((460,420,3),245,np.uint8);im=im[:410]
            tile[40:40+im.shape[0],10:410]=im
            cv2.putText(tile,stage,(10,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),2);tiles.append(tile)
        cv2.imwrite(str(OUT/(case+'-lazy')/f'det-{det["index"]}/stage-overview.jpg'),np.hstack(tiles))

for name,details in variants:
    for call in details.get('ocr_calls',[]):
        for item in call['results']:
            rows.append({'variant':name,'call':call['label'],'raw_text':item['text'],
                         'confidence':item['confidence'],'bbox':json.dumps(item['bbox'])})
with (OUT/'raw_ocr.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['variant','call','raw_text','confidence','bbox']);w.writeheader();w.writerows(rows)

before=load(OUT/'protected_before.json')
changed=[p for p,digest in before.items() if not (ROOT/p).exists() or sha(ROOT/p)!=digest]
dump(OUT/'integrity.json',{'protected_file_count':len(before),'changed_or_missing':changed,
                         'weights_sha256':sha(ROOT/'weights/best.pt'),'production_patch_applied':False})

table=[]
for name,d in variants:
    raw='; '.join(f"{x['text']} ({x['confidence']:.6f})" for c in d.get('ocr_calls',[]) for x in c['results'])
    table.append('| '+ ' | '.join([name,raw or ('**runtime error**' if d.get('error') else 'not completed'),
        ' / '.join(d.get('raw_lines',[])) or '—',d.get('format_status','—'),d.get('plate_number','') or 'rỗng'])+' |')

report='''# Debug đối chứng A / B — đúng hai ảnh

## Kết luận và giới hạn bằng chứng

**B tái lập chính xác response đã cung cấp. A tái lập chính xác cả hai YOLO detection, nhưng request mới bị native crash tại CRAFT trên detection thứ hai; không tái lập được HTTP 200 có hai chuỗi rỗng.** Trên detection biển thật của A, trước crash, trace ghi lỗi cấp phát CPU 1.656.750.080 bytes ở CRAFT; `read_text_details()` bắt exception rồi tạo `plate_number=""`. Chạy riêng đúng pixels đó, không nạp YOLO, vẫn gặp cùng lỗi cấp phát. Đây là nguyên nhân trả rỗng đã xác minh trong môi trường hiện tại, **chưa đủ để khẳng định request trước của người dùng cũng rỗng vì thiếu RAM**. Muốn kết luận lịch sử cần log server đúng request đó (exception/CRAFT/raw OCR), không cần thêm ground truth cho hai ảnh này.

Khi giảm kích thước để OCR hoàn tất, A còn lỗi nhận dạng và ghép hàng: `B1→B7`, một fragment nền `B` confidence 0.008686, các fragment hai hàng bị gom vào cùng hàng. Xoay hay warp thử nghiệm cũng chưa đọc đúng serial; còn làm hỏng B đang đúng. **Không áp dụng patch production trong task này.** Không có bằng chứng cho việc formatter làm mất ký tự đúng ở baseline B hoặc tự sửa A thành đáp án.

Ground truth chỉ lấy từ người dùng: A `29-B1 542.70`, B `17-AK 044.70`. JSON user reference tách riêng, không truyền vào engine/formatter. Dấu nối/chấm khác nhau không phải lỗi recognition; serial sai là lỗi.

## 1. Đường chạy và cấu hình

`POST /detect` qua FastAPI TestClient → `DetectionService.detect_bytes` → `PlateDetector.detect` → crop padding → `preprocess_plate` → `PlateOCR.read_text/read_text_details` → EasyOCR CRAFT/recognizer → formatter. Wrapper chỉ ghi ảnh/thông tin và trả nguyên kết quả; dùng ảnh JPEG gốc, không dùng crop diagnostic cũ làm input API.

Lượt chính `A-lazy`, `B-lazy`: Reader khởi tạo lazy ngay trong thread xử lý request, sau YOLO, như production. Chỉ tắt download và verbose trong harness vì models đã cache; không đổi model, decoder, pixels hay threshold. Lượt đầu `A`, `B` nạp Reader trước request, cả hai native crash; giữ riêng, không dùng làm kết quả đối chứng chính.

YOLO effective: conf 0.25, IoU 0.7, imgsz 640, max_det 300, rect True, class 0. Service confidence 0.25, padding 0.1. OCR CPU/en, allowlist `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.`, deskew=True, bilateral=False, CLAHE=True (một lần), upscale=2, aspect candidate<1.9, safe-gap split. Không đổi bất kỳ threshold nào. [Trace A](A-lazy/trace.json), [trace B](B-lazy/trace.json), [runtime statuses](run_status.json).

## 2. Detection và padded crop

Coordinates xyxy trong ảnh gốc 3072×4096; kích thước ảnh ở bảng là W×H.

| Case / det | Confidence | YOLO xyxy | Padded xyxy | Crop → upscale | Nhận xét |
|---|---:|---|---|---|---|
| A / 0 | 0.7889413237571716 | [1354.252,1277.446,2922.956,2806.671] | [1197,1125,3072,2960] | 1875×1835 → 3750×3670 | Biển thật, chứa đủ ký tự |
| A / 1 | 0.5282004475593567 | [303.964,0.966,2495.852,1223.272] | [85,0,2715,1346] | 2630×1346 → 5260×2692 | False positive bảng quảng cáo/nền |
| B / 0 | 0.7126506567001343 | [1219.358,1576.238,1820.469,2059.521] | [1159,1528,1881,2108] | 722×580 → 1444×1160 | Biển thật, chứa đủ ký tự |

A có hai box class 0, confidence đều vượt 0.25. Hai YOLO box không giao nhau theo trục Y (1223.272 < 1277.446), IoU=0, nên NMS giữ cả hai đúng cơ chế; đây là model false positive, không phải hai box trùng biển hay lỗi NMS. B chỉ có một box sau cùng detector/config. Detection confidence không đo chất lượng OCR. Không tăng threshold để che false positive.

[Overlay A](A-lazy/detections.jpg) · [Overlay B](B-lazy/detections.jpg) · [A biển thật](A-lazy/det-0/00-crop.png) · [A false positive](A-lazy/det-1/00-crop.png) · [B crop](B-lazy/det-0/00-crop.png).

## 3. Preprocessing và split

Mỗi detection có `00-crop.png`, `01-deskew.png`, `02-grayscale.png`, `03-clahe.png`, `04-upscale.png`; SHA256 và shape nằm trong trace. [A/0 overview](A-lazy/det-0/stage-overview.jpg) · [A/1 overview](A-lazy/det-1/stage-overview.jpg) · [B overview](B-lazy/det-0/stage-overview.jpg). Overview chỉ để xem; PNG gốc là pixels thật.

- Cả ba `_find_plate_quad()` trả None: deskew image **giống nguyên pixels crop**, không có warp âm thầm cắt ký tự. Pipeline grayscale → CLAHE clip2/tile8 → cubic upscale2. Không CLAHE lần hai trong OCR.
- A/0 aspect=1.02180, B/0=1.24483 đều là split candidate; `find_split_row()` trả None, `split_status=rejected_no_reliable_gap`, đọc whole crop. **Không có top/bottom production để xuất vì split bị từ chối**, không phải thiếu artefact. A/1 aspect=1.95394, không yêu cầu split.
- Quad detector lấy contour lớn nhất, đòi area≥50% padded crop (`preprocessing.py:77–95`). Ở A/0 bright contour lớn nhất chỉ 36.41%; B/0 40.58%. Cả hai bị gate loại. Contour còn dính vùng nền, vì vậy hạ area gate đơn thuần không chứng minh sẽ chọn đúng khung biển. Nhánh đảo polarity theo mean cũng không kích hoạt ở hai crop này (binary mean≈94.0 và96.8).
- Ngưỡng diện tích 50% khiến deskew không hoạt động ở đây; **không đồng nghĩa đây là nguyên nhân duy nhất gây lỗi**, vì B vẫn đúng dù cũng skip deskew.

## 4. Stage đầu tiên thất bại và phân loại

| Stage | A biển thật | A bảng quảng cáo | B |
|---|---|---|---|
| Detection | Có đúng vùng biển | **Detection false positive** | Có đúng vùng biển |
| Crop / preprocessing | Không mất ký tự qua crop; deskew bỏ qua; input OCR rất lớn | Crop hợp box YOLO nhưng không phải biển | Crop đầy đủ; deskew bỏ qua |
| CRAFT production | **CPU allocation error**, chưa trả boxes; không được gọi là 0 boxes | **Native access violation**, chưa trả boxes | Hoàn tất: 1 horizontal +1 free box |
| EasyOCR recognition production | Chưa chạy thành công; không có raw text/conf hợp lệ | Chưa chạy thành công | `17-AK` 0.9058670595759083; `044.70` 0.9998050357717924 |
| Normalization / formatter | Catch exception tạo empty details, đây là **runtime failure**, không phải format rejection của text đã nhận | Không có final vì process chết | `17-AK→17AK`, `044.70→04470`, corrections=[], status=structured, `17AK-044.70` |

**Điểm phân kỳ có bằng chứng ở lượt production hiện tại là CRAFT/runtime**, sau khi YOLO và preprocessing đã xong. Chưa thể kết luận CRAFT bỏ hàng hay recognizer đọc sai trên exact-size A vì call chưa trả thành công. B CRAFT boxes và raw confidence nằm đầy đủ trong JSON; [overlay](B-lazy/det-0/ocr-0-boxes.jpg). A exact-size boxes không tồn tại vì inference thất bại, không bịa boxes từ run khác.

Input A/0 3750×3670 được EasyOCR default canvas_size2560/mag_ratio1 resize/pad thành khoảng 2560×2528; B giữ1444×1160 rồi pad1472×1184. Diện tích CRAFT A lớn khoảng3.71 lần B. Trace stack chỉ đến convolution/VGG của CRAFT; thiếu RAM là exception cụ thể, native AV không tự được quy thành thiếu RAM. Code liên quan: `easyocr.py:311,440` canvas mặc định; `imgproc.py:37` resize/pad; `detection.py:24–49` forward. Upscale2 không có giới hạn OCR riêng trong source ứng dụng.

## 5. Đối chứng tilt / perspective — chỉ diagnostic

Không đổi detector/crop production. Các input diagnostic giới hạn cạnh dài1444 để khảo sát hình học mà không nhầm resource failure thành recognition. Đây **không phải rerun cùng cấu hình pixels production A** và không thay thế trace baseline.

1. `bounded_whole`: crop cố định → grayscale+CLAHE như hiện tại, không deskew → resize1444.
2. `rotation_only`: xoay toàn crop bằng góc cạnh trên minAreaRect contour sáng, giữ canvas mở rộng, border replicate → grayscale+CLAHE → resize1444. Góc A≈22.35°, B≈9.80°. Giữ cả nền; không chọn theo text/confidence.
3. `panel_axis_aligned` và `panel_projective`: dùng bốn góc **khung biển được chọn thủ công chỉ để chẩn đoán**, xuất overlay và tọa độ. Axis lấy enclosing rectangle; projective warp khung rồi cùng grayscale/CLAHE/resize1444. Thí nghiệm này đổi ROI/resampling và đôi khi route whole/split, nên **không cô lập được riêng ảnh hưởng phối cảnh**, không chứng minh warp tự động tốt hơn. Không đưa góc thủ công vào production.

[A rotation inputs/trace](A-geometry-0/trace.json) · [B rotation](B-geometry-0/trace.json) · [A manual frame](A-panel/manual-frame.jpg) · [B manual frame](B-panel/manual-frame.jpg) · [A panel trace](A-panel/trace.json) · [B panel trace](B-panel/trace.json) · [false-positive diagnostic](A-geometry-1/trace.json).

Raw/confidence dưới đây là trước formatter; CSV và JSON giữ full precision/boxes. `raw_lines` đã qua grouping/join, chưa normalization. Final hợp cấu trúc vẫn có thể sai ground truth.

| Variant | Raw EasyOCR fragments (confidence) | raw_lines sau grouping/join | format_status | plate_number |
|---|---|---|---|---|
'''+ '\n'.join(table)+'''

Các phát hiện riêng:

- A bounded: CRAFT có2 horizontal+3 free boxes, có cả vùng nền sát mép trên (`B`, confidence0.008686). Nhận sai `B1→B7`, serial bị tách `542..` và `70`. `_group_lines()` dùng center-Y/average box height rồi so với **fragment liền trước**, không mô hình slope; hai hàng nghiêng bị chain vào `542..2970B7`. Đây là lỗi grouping thực tế trên input diagnostic; nhận dạng đã sai trước grouping nữa. Không chọn highest-confidence fragment (29/70) làm toàn biển.
- A rotation: raw `29`, `B1`, `642.70` → final `29-B1 642.70`, vẫn sai chữ số5→6. B rotation: raw `044.20` thay cho `044.70`, regression đã xác minh với text người dùng. Không thể bật rotation mặc định dựa vào confidence/top line.
- A projective: safe split được chấp nhận row561 (input1444×1133); top `29-B1`, bottom `54`,`70`,`7` ghép trái→phải thành54770 → final `29-B1 547.70`, vẫn sai2→7. [Top](A-panel/panel_projective-top.png) / [bottom](A-panel/panel_projective-bottom.png).
- B projective: safe split row525 (input1444×1029); top17+AK; bottom02+L+70 → raw_lines `17AK / 02L70`. Heuristic hiện tại sửa L→1 ở slot serial thành02170, final `17AK-021.70`, sai dù status=heuristic_corrected. [Top](B-panel/panel_projective-top.png) / [bottom](B-panel/panel_projective-bottom.png). Đây là ví dụ cần giữ `text_verified=False`; cấu trúc đủ không chứng minh text đúng.
- False positive bounded đọc BE/ER20K/PUJ/C4./LAVE./A0/45K/CQLD/HONDA. Formatter status=unrecognized, không correction, plate_number rỗng là rejection hợp lý của vùng không phải biển. **Chỉ kết quả diagnostic này**, không gán cho request A bị crash.

## 6. Minimal fix đề xuất, chưa áp dụng

Không thay production vì chưa có phương án hình học nào khắc phục A và giữ B đúng. Không hardcode case/plate/GT, không sửa formatter để ép kết quả.

- **Resource handling:** điểm sửa nhỏ nhất ở `PlateOCR.read_text_details()` (`postprocessing.py:291–318`): giữ `error` và phân biệt runtime failure trong log/status với empty OCR; hiện catch tạo `format_status=empty` rồi `read_text()` chỉ trả string nên client không phân biệt được. Cần giữ API compatibility khi bổ sung observability. Không có Python catch nào xử lý được native access violation.
- **Resource budget candidate:** bổ sung giới hạn kích thước đầu vào CRAFT có cấu hình tại lệnh `reader.readtext`, chỉ tác động crop quá lớn; không đổi threshold. Bounded diagnostic chứng minh inference có thể hoàn tất khi nhỏ hơn, **chưa chứng minh đây là recognition fix**. Không tự chọn1536/1444 làm production default từ hai ảnh; cần budget môi trường và kiểm chứng exact production option trước khi merge.
- **Geometry/grouping:** giữ fallback whole khi không có gap; chưa hạ area gate hay ép split. Contour sai có thể chứa nền; minAreaRect chỉ cho hình chữ nhật xoay, không đo được keystone thật. Candidate slope-aware grouping/quadrilateral localization cần kiểm chứng riêng, không trộn với formatter và không xem final đúng regex là improvement.

Do không áp dụng production patch, Before/After production là **giữ nguyên**; bảng trên là đối chứng input diagnostic, không quảng cáo là Before/After bản vá. Các snapshot trước task nằm trong `before/`; không có diff production cần rollback.

## 7. Regression, crash status và file bàn giao

- `python -B -m pytest tests -q -p no:cacheprovider`: **62 passed, 7 warnings, 16.09s**, có real inference smoke. [Log](pytest.log).
- Lệnh pytest không chỉ định `tests` ban đầu bị3 lỗi collection do snapshot `experiments/.../before/tests` trùng module name; đây không phải assertion regression. Đã chạy lại đúng current tests; không xóa snapshot hay sửa cấu hình test.
- 4 lượt full API worker: first A/B native AV; lazy A native AV ở det1, lazy B HTTP200/exit0. Native exit3221225477=0xC0000005, log giữ nguyên. Exact A/0 replay bắt được RuntimeError allocation; hoàn tất details có error, không có boxes/raw. Tất cả9 diagnostic variants (A4+B4+falsepositive1) có details hoàn tất, không native crash. Hai lượt geometry đầu có lỗi serialize NumPy ở dòng **print cuối sau khi JSON/ảnh đã lưu đủ**; đã sửa harness print, không thay kết quả inference. PowerShell redirect stderr có thể báo exit1 vì EasyOCR warning; đánh giá completion dựa checkpoint và log, không coi warning là native crash.
- B baseline khớp ground truth; rotation và projective đều gây regression. Không tuyên bố accuracy dataset tăng hay đã sửa hoàn toàn A.
- Hash đối chiếu file bảo vệ: '''+str(len(before))+''' files; changed/missing='''+json.dumps(changed)+'''. [Integrity](integrity.json). Model/dataset/labels/source/tests không bị sửa. Không sửa environment/dependencies/threshold, không annotation workflow, không mở benchmark dataset.

File mới: `scripts/compare_alpr_cases.py`, `scripts/finalize_alpr_comparison.py`, toàn bộ output chỉ dưới `experiments/compare-alpr-A-B/`. [Raw CSV](raw_ocr.csv), [user references](user_references.json), [source/config/box traces](A-lazy/trace.json), ảnh stage/raw boxes, logs, snapshots và integrity. Không ghi đè report cũ hoặc `ketqua.md` vì đây là task debug hai ảnh mới.

## Chạy lại bằng PowerShell

```powershell
Set-Location D:\\CCNLTHD
# Từng image worker, tag mới để không ghi đè bằng chứng:
.\\.venv\\Scripts\\python.exe -B scripts/compare_alpr_cases.py worker --case B --tag B-repeat
.\\.venv\\Scripts\\python.exe -B scripts/compare_alpr_cases.py worker --case A --tag A-repeat
# A có thể tiếp tục thiếu RAM/native crash với cấu hình production hiện tại.
.\\.venv\\Scripts\\python.exe -B -m pytest tests -q -p no:cacheprovider
```

Không dùng `prepare` lần nữa trên workspace đã khởi tạo. `geometry`/`panel` chỉ là diagnostic và sẽ ghi đè thư mục diagnostic cùng tên nếu chạy lại; không cần rerun để xem ảnh và raw đã xuất.
'''
(OUT/'report.md').write_text(report,encoding='utf-8')
print(json.dumps({'raw_fragments':len(rows),'variants':len(variants),'protected_changed':changed,'report':str(OUT/'report.md')}))

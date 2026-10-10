"""Build a reviewable runtime report from saved A/B evidence, without model runs."""
import ast
import csv
import json
import logging
from pathlib import Path
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from check_ocr_runtime import OUT,package
from compare_alpr_cases import load,dump,sha
import cv2
import numpy as np
from src.inference.postprocessing import PlateOCR,_craft_resize_plan
from src.inference.preprocessing import crop_plate


def shape_text(shape):
    return '×'.join(str(x) for x in shape)


def build():
    package()
    before_dir=ROOT/'experiments/compare-alpr-A-B'
    input_rows=[];raw_rows=[];tensor_rows=[];comparison_checks=[]
    for case in ['A','B']:
        before=load(before_dir/(case+'-lazy')/'trace.json')
        after=load(OUT/(case+'-after')/'trace.json')
        tensors=load(OUT/(case+'-after')/'tensors.json')
        if case=='B':
            assert before['traces'][0]['details']['ocr_calls'][0]['results']==after['traces'][0]['details']['ocr_calls'][0]['results']
            assert before['api_response']==after['api_response']
        original=cv2.imread(str(ROOT/'data/custom/images/test'/before['filename']))
        for i,det in enumerate(after['traces']):
            source_crop=crop_plate(original,det['xyxy'])
            path=OUT/(case+'-after')/f'det-{i}/unpadded.png'
            cv2.imwrite(str(path),source_crop)
            same_pixels=all(det['stages'][name]['sha256']==before['traces'][i]['stages'][name]['sha256'] for name in det['stages'])
            comparison_checks.append({'case':case,'detection':i,'same_preprocessing_pixels':same_pixels,
                'same_bbox':det['xyxy']==before['traces'][i]['xyxy']})
            padded=det['stages']['crop']['shape'][:2]
            up=det['stages']['upscale']['shape'][:2]
            input_rows.append(f'| {case}/{i} | {shape_text(source_crop.shape[1::-1])} | {shape_text(padded[::-1])} | '+
                ' | '.join(shape_text(det['stages'][name]['shape'][1::-1]) for name in ['deskew','grayscale','clahe','upscale'])+' |')
            old_plan=_craft_resize_plan(up,2560*2560)
            new_plan=_craft_resize_plan(up,2*1024*1024)
            actual=tensors['craft_forwards'][i]
            assert actual['shape']==new_plan['craft_tensor_shape']
            tensor_rows.append(f'| {case}/{i} | {old_plan["canvas_size"]} | {shape_text(old_plan["craft_tensor_shape"])} | '+
                f'{old_plan["full_resolution_64ch_float32_bytes"]:,} | {new_plan["canvas_size"]} | {shape_text(actual["shape"])} | '+
                f'{new_plan["full_resolution_64ch_float32_bytes"]:,} | {actual["status"]} |')
            for call in det.get('details',{}).get('ocr_calls',[]):
                for item in call['results']:
                    raw_rows.append({'case':case,'detection':i,'mode':'production API trace','raw_text':item['text'],
                        'confidence':item['confidence'],'bbox':json.dumps(item['bbox'])})
    isolated=load(OUT/'A-isolated-1/trace.json')
    for call in isolated['details']['ocr_calls']:
        for item in call['results']:
            raw_rows.append({'case':'A','detection':1,'mode':'isolated exact-pixel replay; not API response',
                'raw_text':item['text'],'confidence':item['confidence'],'bbox':json.dumps(item['bbox'])})
    with (OUT/'raw_ocr.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['case','detection','mode','raw_text','confidence','bbox'])
        writer.writeheader();writer.writerows(raw_rows)
    def formatter_ast(path):
        tree=ast.parse(path.read_text(encoding='utf-8'))
        return {n.name:ast.dump(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name!='_craft_resize_plan'}
    assert formatter_ast(OUT/'before/src/inference/postprocessing.py')==formatter_ast(ROOT/'src/inference/postprocessing.py')
    dump(OUT/'pixel-and-formatter-checks.json',{'preprocessing':comparison_checks,'formatter_functions_ast_unchanged':True,
        'case_B_raw_boxes_text_confidence_exact_match':True,'case_B_api_exact_match':True})
    # Deliberate fake-reader examples exercise exception handling only, not recognition.
    handler=logging.FileHandler(OUT/'error-handling.log',mode='w',encoding='utf-8')
    logger=logging.getLogger('src.inference.postprocessing');logger.addHandler(handler)
    previous=logger.level;logger.setLevel(logging.INFO)
    examples={}
    class Reader:
        def __init__(self,mode):self.mode=mode
        def readtext(self,image,**kwargs):
            if self.mode=='runtime':raise RuntimeError('INJECTED allocator failure (not a model run)')
            if self.mode=='empty':return []
            return [([[0,0],[90,0],[90,20],[0,20]],'L8953',.4)]
    try:
        for mode in ['runtime','empty','partial']:
            ocr=PlateOCR();ocr._reader=Reader(mode)
            examples[mode]=ocr.read_text_details(np.zeros((50,200),np.uint8))
    finally:
        logger.removeHandler(handler);handler.close();logger.setLevel(previous)
    dump(OUT/'injected-error-examples.json',{'simulated_reader_only':True,'examples':examples})
    test_log=(OUT/'pytest.log').read_text(encoding='utf-8').strip().splitlines()[-1]
    report='''# ALPR OCR runtime và error handling — minimal patch

## 1. Kết quả và giới hạn

**Đã sửa việc phân loại lỗi runtime thành OCR empty và thêm ngân sách tensor CRAFT. B giữ nguyên `17AK-044.70`. CRAFT biển thật A chạy hoàn tất, không còn lỗi cấp phát 1,66 GB ở lượt sau patch. Native crash trong full request A vẫn còn ở detection bảng quảng cáo; chưa thể tuyên bố request A ổn định.**

Không tối ưu recognition trong task này. A vẫn bị nhận/ghép chữ sai và bị formatter từ chối; không ép đáp án `29-B1 542.70`. Không thay YOLO checkpoint, confidence threshold, dataset, labels, EasyOCR engine, góc xoay, split algorithm hay formatter.

Baseline dùng artefact cũ [A](../compare-alpr-A-B/A-lazy/trace.json), [B](../compare-alpr-A-B/B-lazy/trace.json), không chạy lại nhánh có nhu cầu1,66GB. Sau patch chạy đúng hai JPEG gốc qua FastAPI `/detect` → production service với cùng detector/config, ngoài budget CRAFT mới. Reader vẫn lazy trong request thread. [Run status](run_status.json).

## 2. Vì sao cấp phát 1.656.750.080 bytes?

Ảnh gốc cả hai case3072×4096. Bảng W×H dưới đây tách crop không padding và crop service padding0.1. `unpadded.png` xuất bổ sung để xem, không đưa vào production OCR.

| Case/det | Core crop không padding | Padded crop | Deskew | Grayscale | CLAHE | Upscale / input readtext |
|---|---|---|---|---|---|---|
'''+ '\n'.join(input_rows)+'''

Deskew trả nguyên pixels cả ba crop. Grayscale đổi3channel→1channel; CLAHE clip2/tile8 không đổi shape; cubic upscale2 nhân4 số pixels. [Pixel/hash checks](pixel-and-formatter-checks.json) xác nhận tất cả intermediate pixels trước/sau patch giống nhau. Việc resize giới hạn chỉ xảy ra **bên trong text detector CRAFT**, không resize ảnh gốc, crop, hay ảnh chuyển cho recognizer. EasyOCR map boxes ngược về cùng input readtext, nên raw bbox dùng tọa độ ảnh upscale gốc.

Trước patch `reader.readtext(part, allowlist=...)` để mặc định canvas_size2560, mag_ratio1, batch1. EasyOCR `imgproc.resize_aspect_ratio()` scale cạnh dài về tối đa canvas rồi **pad lên bội32**. `detection.test_net()` chuyển RGBfloat32→NCHW và gọi CRAFT. A/0 tạo `1×3×2528×2560`; các lớp đầu VGG có64channel. Một buffer float32 cùng độ phân giải cần:

```
1 × 64 × 2528 × 2560 × 4 = 1,656,750,080 bytes = 1,580 MiB
```

Con số khớp lỗi allocator. Trace cũ dừng ở `F.conv2d` trong `vgg16_bn.slice1`; chưa có module-name hook cũ để xác định chính xác convolution nào hoặc workspace nội bộ. Cùng vùng full-resolution có nhiều buffer/workspace cùng sống, nên peak RAM vượt kích thước một activation. Không có shape bị corrupt hay chỉ số bbox phi lý; **đây là nhu cầu tensor lớn nhưng hợp lệ, không phù hợp resource budget hiện tại**. EasyOCR đã có cap2560, không phải tăng kích thước vô hạn.

Vị trí cũ gây lỗi: `postprocessing.py:292` gọi readtext không có budget riêng; `preprocessing.py:171` upscale; service dòng103/106. Stack dependency: `easyocr.py:456` → detect321 → `detection.py:46` net → `craft.py:60` basenet → `model/modules.py:68` slice1 → `torch/nn/modules/conv.py:560` F.conv2d. [Log trước patch](../compare-alpr-A-B/A-lazy/runtime.log).

### RAM vật lý và commit

[Windows snapshot đầu task](memory_before.json): RAM vật lý tổng16.832GB, available4.214GB; available commit/pagefile headroom theo GlobalMemoryStatusEx khoảng2.364GB. Ngay trước CRAFT A/0 sau khi models đã nạp, snapshot mới ghi physical available4.240GB, commit headroom1.556GB — không thể đồng nhất “còn RAM vật lý” với “còn khả năng cấp phát/commit”. Snapshot này thuộc lượt mới, không phải đo lại lịch sử trước patch.

Không kết luận máy đã hết sạch RAM vật lý hoặc lỗi ảnh là nguyên nhân duy nhất. Allocation failure cũ là bằng chứng runtime thiếu khả năng cấp phát block yêu cầu; kích thước tensor được code tạo ra giải thích vì sao nhu cầu lớn. Không có profile cũ để quy chính xác phần thiếu cho physical RAM, commit limit, allocator cache hay workspace. Không đổi pagefile, environment hoặc dừng tiến trình người dùng để che lỗi.

## 3. Patch thực tế

**`src/utils/config.py`**: thêm `OCR_CRAFT_MAX_PIXELS=2097152`, có thể cấu hình bằng environment khi deploy; không sửa `.env` hiện tại. Đây là **ngân sách2Mi pixels sau padding**, tương ứng tối đa512MiB cho một activation64channelfloat32; không phải giới hạn toàn bộ working set. Default dựa trên kích thước activation gây lỗi và control B có tensor1.742.848 pixels, không dựa nội dung/case_id/GT. Ngân sách này là giới hạn tài nguyên, không phải chọn winner recognition.

**`postprocessing.py::_craft_resize_plan()` dòng213, `__init__()` dòng242, lệnh readtext dòng340**: tính đúng phép scale/pad của EasyOCR, chọn canvas lớn nhất theo bước32 thỏa budget. Giữ mặc định2560 khi đã vừa; chỉ truyền `canvas_size` cho crop quá lớn. mag_ratio vẫn1. Recognition tiếp tục nhận pixels upscale gốc; không đổi decoder/threshold. Plan được ghi trước call để còn bằng chứng nếu call lỗi. Constructor kiểm tra budget>=1024, không âm thầm tắt limit khi cấu hình sai.

| Case/det | Canvas trước | Tensor trước (suy ra từ source) | Buffer64ch trước / bytes | Canvas sau | Tensor sau (hook đo thật) | Buffer64ch sau / bytes | CRAFT status |
|---|---:|---|---:|---:|---|---:|---|
'''+ '\n'.join(tensor_rows)+'''

[A actual tensors/memory](A-after/tensors.json) · [B actual tensors/memory](B-after/tensors.json). Cả ba plan≤2Mi padded pixels; B không thêm canvas override, không đổi input pixels/CRAFT tensor/recognition boxes.

**`PlateOCR.read_text_details()` dòng317 và exception handler dòng369**: thêm `ocr_status` riêng, giữ formatter nguyên logic:

| ocr_status | Nghĩa | format_status |
|---|---|---|
| empty_ocr | EasyOCR call hoàn tất, không có text | empty |
| unrecognized_format | Có OCR text, không có plate được formatter chấp nhận | unrecognized; trường hợp chỉ punctuation có thể empty sau normalization |
| ocr_runtime_error | Exception trong khởi tạo, resize-plan, readtext hoặc xử lý OCR | not_completed |
| ok | Pipeline xuất được plate; không chứng nhận text đúng GT | structured / heuristic_corrected |

Call được append **trước inference** với `status=started`, shape, budget/resize plan; sau đó completed hoặc ocr_runtime_error. Exception giữ `error`, `error_type`, `error_stage`, traceback; `logger.exception` ghi full stack. Nếu một nửa trước đã đọc được, giữ fragments đó và failed call, không format partial text thành plate. Runtime handler không gọi `format_vn_plate_details([])` nữa. [Injected examples](injected-error-examples.json) / [exception log](error-handling.log) chỉ dùng fake Reader để kiểm tra xử lý lỗi, không phải kết quả OCR ảnh A.

**`PlateOCR.read_text()` và service logging**: log ocr_status cùng format_status. Service bỏ thông điệp sai “OCR returned EMPTY” cho mọi chuỗi rỗng, thay bằng “no plate_number emitted; see OCR status/exception logs”. Các try/except decode, YOLO và preprocess/service vẫn ghi logger.exception; ngoại lệ Python không làm bỏ các detection tiếp theo trong luồng OCR đã được xử lý. Native AV nằm ngoài try/except Python.

### Public API

Giữ contract `/detect`, HTTP status và hai field `confidence`, `plate_number`; `read_text()` vẫn trả string. Runtime/unrecognized/empty có thể cùng public string rỗng **để tương thích**, nhưng trace/debug/log phân biệt rõ. Không thêm traceback ra public response. Client chỉ dùng response cũ sẽ chưa phân biệt ba lý do; nếu cần sau này có thể thêm endpoint/debug contract riêng sau khi review, task này chưa thay contract.

## 4. Before/After runtime và raw OCR

| Detection | Trước patch | Sau patch trong production request | API thực tế sau patch |
|---|---|---|---|
| A/0 biển thật, conf0.7889413237571716 | CRAFT RuntimeError allocation1656750080; bị ghi format_status=empty; raw không tồn tại | CRAFT/recognition hoàn tất; ocr_status=unrecognized_format; raw_lines `C / 542.29-B170`; no corrections; plate rỗng | Request sau đó crash ở A/1, **không có HTTP response hoàn tất** |
| A/1 bảng quảng cáo, conf0.5282004475593567 | CRAFT native AV, không có raw hợp lệ | Vẫn native AV trong CRAFT dù tensor đã nhỏ; không có raw từ call này | Không bịa response có hai empty plates |
| B/0 biển thật, conf0.7126506567001343 | raw `17-AK` (0.9058670595759083), `044.70` (0.9998050357717924); API `17AK-044.70` | Raw, confidence, boxes và plate_number **giữ nguyên**; ocr_status=ok | HTTP200 `{"results":[{"confidence":0.7126506567001343,"plate_number":"17AK-044.70"}]}` |

[A trace](A-after/trace.json) · [A runtime log](A-after/runtime.log) · [B trace/API](B-after/trace.json) · [B runtime log](B-after/runtime.log) · [Raw OCR full precision CSV](raw_ocr.csv). A/0 vẫn rỗng vì **format rejection sau OCR có text**, không phải runtime error giả làm empty OCR. Raw A/0 đang sai/ghép sai; chưa tuyên bố recognition cải thiện.

## 5. False positive và native crash

Bbox A/1 không đổi `[303.963867,0.966016,2495.852295,1223.271729]`; padded `[85,0,2715,1346]`, crop2630×1346, readtext5260×2692. Crop có bảng quảng cáo/nền. Cùng allowlist/threshold như biển thật; không loại theo confidence hoặc hardcode case.

Stage cuối sau patch: Reader.readtext → detect → CRAFT forward → `craft.py:76 upconv4` → double_conv `craft.py:26` → F.conv2d; native exit3221225477 (`0xC0000005`). CRAFT chưa trả boxes, recognizer chưa chạy. Đây là **native runtime failure**, không phải CRAFT empty boxes hay format rejection. Log sau patch cho thấy crash ở upconv4; stack baseline cũ dừng trong slice1, nên không đánh đồng hai vị trí.

Ngay trước CRAFT detection thứ hai trong request mới: physical available≈3.011GB, commit headroom≈0.634GB. Nhu cầu một activation~0.520GB chưa gồm các buffers/workspace khác; áp lực commit là yếu tố đáng điều tra, **chưa phải root cause native đã chứng minh**.

Đối chứng riêng **đúng pixels và cùng canvas1984** trong tiến trình mới không nạp YOLO: [isolated trace](A-isolated-1/trace.json), [tensors/last conv](A-isolated-1/tensors.json), [exit0](A-isolated-1/exit.json). CRAFT+recognizer hoàn tất; raw_lines `BEER20KJUJYC4.LAVE.145K / COL / HONDA`, status unrecognized_format, plate rỗng. Đây không phải response API. Đối chứng còn khác resident YOLO, thread và lịch sử các allocations/calls; không thể xác định một yếu tố duy nhất từ nó. Nó bác bỏ việc mặc định “cứ ảnh bảng quảng cáo này là crash”.

**Native crash chưa được sửa**. Pixel budget giảm activation lớn nhưng không ràng buộc toàn bộ model/workspace/cache. Muốn bảo vệ API trước mọi AV cần isolation bằng worker process và xử lý worker death; đây là thay đổi kiến trúc lớn hơn patch hiện tại, chưa tự thêm. Chưa đổi torch/backend/dependencies hoặc thử hàng loạt cấu hình.

## 6. Regression và review/rollback

Test suite current `tests/` (có real inference smoke): **'''+test_log+'''**. [Full log](pytest.log). Test mới bao gồm phân loại3status, exception+traceback, init failure, partial split failure, lỗi grouping đúng stage, ngân sách có tính padding, giữ pixels recognizer/B defaultcanvas, mọi call split+fallback, budget sai, API schema/HTTP compatibility và tiếp tục detection sau simulated allocation failure. Unit fault injection là test Python exception, không giả lập rằng try/except bắt được native AV.

Chỉ3 production files thay đổi: config, postprocessing, service log. Thêm regression tests vào3 test files. Hàm formatter được so AST với snapshot và giữ nguyên; không thay preprocessing. [Integrity](integrity.json) kiểm tra source/weights/test images/labels, chỉ các file đã nêu được sửa. [Unified diff](production-and-tests.patch), snapshots đầy đủ trong `before/` để review/rollback. Script mới `scripts/check_ocr_runtime.py`, `scripts/report_ocr_runtime.py`; artefact chỉ ở thư mục task này, không ghi đè baseline.

Không lấy prediction làm ground truth. Ground truth A/B do người dùng cung cấp chỉ dùng đánh giá và không đi vào engine. Recognition còn tồn tại: text nền trong padded crop, ký tự sai/fragmentation và grouping trên biển nghiêng; false positive detector. Chưa sửa formatter/geometry để làm chuỗi khớp GT.

```powershell
Set-Location D:\\CCNLTHD
.\\.venv\\Scripts\\python.exe -B -m pytest tests -q -p no:cacheprovider
# Tag mới để giữ bằng chứng cũ. A vẫn có nguy cơ native AV đã ghi nhận:
.\\.venv\\Scripts\\python.exe -B scripts/check_ocr_runtime.py worker --case B --tag B-repeat
.\\.venv\\Scripts\\python.exe -B scripts/check_ocr_runtime.py worker --case A --tag A-repeat
```

Native AV có thể làm chết tiến trình chạy lệnh; test harness dùng subprocess để ghi nhận riêng và không giết tiến trình API đang có của người dùng.
'''
    (OUT/'report.md').write_text(report,encoding='utf-8')
    print(json.dumps({'report':str(OUT/'report.md'),'raw_fragments':len(raw_rows),'checks':comparison_checks}))


if __name__=='__main__':build()

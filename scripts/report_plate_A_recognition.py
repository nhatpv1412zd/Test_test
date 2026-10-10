"""Package admitted historical trace and one offline layout proposal; no OCR."""
import difflib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compare_alpr_cases import dump, load, sha
OUT = ROOT/'experiments/plate-A-recognition-fix'


def main():
    import cv2
    import numpy as np
    from src.inference.postprocessing import PlateOCR, format_vn_plate_details
    proposed = OUT/'proposed'
    spec = importlib.util.spec_from_file_location('proposed_text_layout', proposed/'src/inference/text_layout.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    replay = {};raw_rows=[];legacy_rows=[]
    for case in ['A','B']:
        baseline=load(OUT/('before-'+case)/'trace.json')
        details=baseline['details'];call=details['ocr_calls'][0]
        assert call['status']=='completed' and call['memory_admission']['allowed']
        raw=call['results'];values=[(r['bbox'],r['text'],r['confidence']) for r in raw]
        layout=module.layout_fragments(values)
        result=format_vn_plate_details(layout['line_texts'])
        replay[case]={'source':'before-'+case+'/trace.json','engine_rerun':False,'layout':layout,'formatted':result}
        # Metrics from the existing production grouping, for root-cause tracing.
        centers=[{'index':i,'cx':float(np.mean(np.asarray(r['bbox'])[:,0])),
            'cy':float(np.mean(np.asarray(r['bbox'])[:,1])),
            'h':float(np.ptp(np.asarray(r['bbox'])[:,1]))} for i,r in enumerate(raw)]
        threshold=max(float(np.mean([r['h'] for r in centers]))*.6,10.)
        ordered=sorted(centers,key=lambda r:r['cy']);groups=[]
        for item in ordered:
            if groups and abs(item['cy']-groups[-1][-1]['cy'])<=threshold:groups[-1].append(item)
            else:groups.append([item])
        indices=[[r['index'] for r in sorted(g,key=lambda r:r['cx'])] for g in groups]
        assert PlateOCR._group_lines(values)==details['raw_lines']
        replay[case]['legacy_layout']={'center_metrics':centers,'threshold':threshold,'indices':indices}
        for i,item in enumerate(raw):
            raw_rows.append(f'| {case}/{i} | {item["text"]} | {item["confidence"]:.16g} | '+
                ('background/outlier' if case=='A' and i==0 else 'plate text')+' |')
        if case=='A':
            legacy_rows=[f'| {r["index"]} | {raw[r["index"]]["text"]} | {r["cx"]:.2f} | {r["cy"]:.2f} | {r["h"]:.2f} |' for r in centers]
        canvas=cv2.imread(str(OUT/('before-'+case)/'03-easyocr.png'))
        for i,item in enumerate(raw):
            row=next((k for k,g in enumerate(layout['line_indices']) if i in g),None)
            color=(0,0,255) if row is None else [(0,180,0),(255,80,0)][row % 2]
            pts=np.int32(item['bbox']);cv2.polylines(canvas,[pts],True,color,4)
            x,y=pts.min(axis=0)
            label=f"raw {i}: {item['text']} | "+('excluded (geometry)' if row is None else f'row {row+1}')
            cv2.putText(canvas,label,(max(0,int(x)),max(35,int(y))),cv2.FONT_HERSHEY_SIMPLEX,1,color,3)
        cv2.imwrite(str(OUT/f'{case}-offline-layout-overlay.jpg'),canvas)
    dump(OUT/'offline-layout-replay.json',replay)
    # Archive production-ready test files alongside the offline runner, without
    # modifying tests/ or applying the proposal to production.
    fixture_path=proposed/'fixtures/raw_layout_references.json'
    target=proposed/'tests/fixtures/raw_layout_references.json';target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(fixture_path.read_bytes())
    offline=(proposed/'test_text_layout.py').read_text(encoding='utf-8')
    body=offline[offline.index('def raw(case):'):]
    header='''"""Geometry regression using saved raw OCR and user-confirmed assertions."""
import copy
import itertools
import json
from pathlib import Path
import numpy as np
import pytest
from src.inference.postprocessing import format_vn_plate_details
from src.inference.text_layout import layout_fragments
REFERENCES = json.loads((Path(__file__).parent/'fixtures/raw_layout_references.json').read_text(encoding='utf-8'))


'''
    (proposed/'tests/test_text_layout.py').write_text(header+body,encoding='utf-8')
    names=['src/inference/postprocessing.py','src/inference/text_layout.py','tests/test_postprocessing.py',
           'tests/test_text_layout.py','tests/fixtures/raw_layout_references.json']
    patch=[]
    for name in names:
        before=OUT/'before'/name
        patch+=difflib.unified_diff(before.read_text(encoding='utf-8').splitlines(True) if before.exists() else [],
            (proposed/name).read_text(encoding='utf-8').splitlines(True),
            fromfile='a/'+name if before.exists() else '/dev/null',tofile='b/'+name)
    (OUT/'proposed.patch').write_text(''.join(patch),encoding='utf-8')
    hashes=load(OUT/'before_hashes.json')
    changed=[p for p,digest in hashes.items() if not (ROOT/p).exists() or sha(ROOT/p)!=digest]
    assert changed==[] and not (ROOT/'src/inference/text_layout.py').exists()
    dump(OUT/'integrity.json',{'production_proposal_applied':False,'protected_file_count':len(hashes),
        'changed_or_missing':changed,'weights_sha256':sha(ROOT/'weights/best.pt')})
    after=load(OUT/'after-A/trace.json');failed=after['details'];memory=failed['ocr_calls'][0]['memory_admission']
    assert failed['ocr_status']=='ocr_runtime_error' and not memory['allowed']
    test_result=(OUT/'offline-tests.log').read_text(encoding='utf-8').strip().splitlines()[-1]
    report='''# Plate A recognition — trace và một candidate, chưa áp dụng production

## Trạng thái bàn giao

**Lượt A mới bị memory admission từ chối trước CRAFT: available1.733.201.920 bytes, required2.257.584.128 bytes. Theo yêu cầu, đã dừng thử nghiệm OCR, không chạy tiếp B và không bypass guard. Không có raw OCR mới để kết luận recognition sau patch.**

Candidate grouping được lưu riêng, **production đã khôi phục byte-for-byte**, gồm source và test hiện có. [Proposed diff](proposed.patch), [integrity](integrity.json). Không tuyên bố API hiện tại đã đọc đúng A. Không tiếp tục điều tra tài nguyên/native crash.

Task dừng ở bước kiểm chứng OCR thực tế vì thiếu tài nguyên; phần dưới phân biệt rõ **trace cũ đã hoàn tất** với **offline replay**. Điều còn thiếu để áp dụng patch là một lượt A admitted/completed sau patch và B regression thực, không phải thêm ground truth (cả hai text đã do người dùng xác nhận).

## 1. Baseline có thể dùng để phân tích

Workspace đã có script, before snapshots, trace A/B và `single-candidate.json` chưa hoàn tất từ lượt trước. Đã kiểm tra SHA256: mọi source/test/weights/test-image/label trong baseline vẫn trùng hiện tại trước khi thử patch. Tận dụng đúng một candidate sẵn có, không mở thêm phương án.

Trong [before-A/trace.json](before-A/trace.json), A/0 đã `status=completed`, OCR call `completed`, admission.allowed=True, available2.463.064.064 ≥required2.257.584.128 bytes. Đây là **OCR thực có chạy và hoàn tất**, không phải chuỗi rỗng do runtime. B baseline cũng admitted/completed. Mỗi baseline chỉ dùng detection index0; không đưa bảng quảng cáo detection1 vào thí nghiệm recognition.

Đường dữ liệu giữ nguyên production:

| Stage | A/0 | B/0 |
|---|---|---|
| Frozen YOLO xyxy | [1354.2515869,1277.4456787,2922.9558105,2806.6713867] | [1219.3576660,1576.2379150,1820.4691162,2059.5207520] |
| Detection confidence | 0.7889413237571716 | 0.7126506567001343 |
| Unpadded crop W×H | 1569×1530 | 601×484 |
| Padded crop, padding0.1 | 1875×1835 | 722×580 |
| Deskew | Skip, nguyên crop | Skip, nguyên crop |
| Grayscale→CLAHE→upscale2 | 3750×3670 | 1444×1160 |
| EasyOCR input | 3750×3670 gray | 1444×1160 gray |
| CRAFT canvas / actual tensor | 1440 /1×3×1440×1440 | 2560 /1×3×1184×1472 |
| Split | rejected_no_reliable_gap→whole | rejected_no_reliable_gap→whole |

Không rerun detector để chọn một bbox thuận lợi khác. Bbox lấy từ trace production trước, crop được dựng lại từ JPEG gốc và kiểm tra **same_pixels_as_production=True**. Checkpoint, confidence threshold, padding và preprocessing không đổi.

[A original detector crop](before-A/01-unpadded.png) · [A padded crop](before-A/02-padded.png) · [A EasyOCR input](before-A/03-easyocr.png) · [A text-box overlay](before-A/04-boxes-0.jpg).

[B detector crop](before-B/01-unpadded.png) · [B padded crop](before-B/02-padded.png) · [B EasyOCR input](before-B/03-easyocr.png) · [B boxes](before-B/04-boxes-0.jpg). Trace giữ CRAFT horizontal/free polygons, raw boxes và full-precision confidence; không dùng OCR prediction làm ground truth.

## 2. Stage đầu tiên sai trong admitted baseline A

CRAFT trả một horizontal box ngoài biển sát mép trên và3 free polygons chứa đủ hai hàng biển. Nó **không bỏ sót hàng chữ** trong trace này. Recognition trên các vùng biển trả đúng toàn bộ chuỗi cần thiết; lỗi không phải raw5→6 hoặc2→7 ở baseline đang xét.

| Case/raw index | Raw EasyOCR text | Confidence | Vùng |
|---|---|---:|---|
'''+ '\n'.join(raw_rows)+'''

Box0 của A nằm x2423..2654,y0..102 trong input3750×3670, ở ngoài biển thật. **Sai lệch đầu tiên là text detection thu thêm vùng nền**, sau đó recognizer đọc thành `C`; confidence thấp không tự chứng minh text sai, vị trí trên ảnh mới là bằng chứng đây không phải ký tự biển. Không chọn highest-confidence fragment làm toàn biển.

## 3. Lỗi grouping làm mất thứ tự hai hàng

`src/inference/postprocessing.py::PlateOCR._group_lines()` dùng cy từ image axes, height=maxY−minY, threshold=0.6×average height; xét khoảng cách cy với **fragment ngay trước**, rồi sort cả group theo cx. Với biển nghiêng, hai hàng có cy chồng nhau và axis-aligned bbox làm height phình lớn.

| Raw index | Text | cx | cy | Axis-aligned height |
|---|---|---:|---:|---:|
'''+ '\n'.join(legacy_rows)+'''

Ngưỡng legacy≈652.05px. cy(top→542.) chênh552.5px và cy(542.→70) chênh461px, đều dưới threshold, nên ba vùng biển bị chain chung một line. Sort cx khiến thứ tự thành **[2,1,3] = `542.` + `29-B1` + `70`**. Box nền thành line riêng[0].

- Raw engine fragments: `C`, `29-B1`, `542.`, `70`.
- Sau grouping legacy: `C / 542.29-B170`.
- Normalized: `C / 54229B170`.
- Corrections: [] (không tự sửa ký tự).
- Format status: unrecognized; final plate_number="".

Formatter từ chối đúng đầu vào sai cấu trúc. Chỉ bỏ `C` vẫn chưa sửa được hàng bị trộn; chỉ sửa formatter sẽ che lỗi thay vì khôi phục reading order.

## 4. Đúng một phương án: geometric line layout

Tận dụng candidate `scripts/plate_line_layout_candidate.py`, đã có offline output trước task. Phiên bản để review tại [proposed/src/inference/text_layout.py](proposed/src/inference/text_layout.py).

1. Ước lượng hướng baseline bằng weighted median góc cạnh trên/dưới OCR quads, trọng số là chiều rộng hình học.
2. Chiếu center/height sang trục dọc baseline và pháp tuyến, group theo khoảng cách pháp tuyến với center của group; sort theo baseline trong từng hàng.
3. Box nhỏ dưới25% reference height chỉ bị loại khi **nằm tách khỏi mọi dominant row**; box nhỏ inline vẫn được giữ. Cần ít nhất hai main boxes để xác lập block. Exclusion giữ nguyên raw text/confidence trong trace, ghi index/reason/khoảng cách, không xóa evidence.

Không nhìn text, regex, GT hay confidence để chọn/ghép hàng. Không thêm/cắt/đổi ký tự. Đây là **một spatial-layout candidate**, không thêm OCR engine, crops, angle candidates hoặc formatter heuristic.

A offline angle26.18°, projected normal heights các vùng biển≈833–897px; box nền≈91.5px, cách row gần nhất1555.8px so tolerance518.8px. Group index[[1],[2,3]], excluded[0]. Các hàng độc lập trở thành `29-B1 / 542.70`.

Không gọi lại recognizer trên hai half-crops: trace đã có đủ ký tự đúng, nên sửa trực tiếp grouping là thay đổi nhỏ hơn; split hiện tại từ chối vì không có horizontal gap đáng tin cậy. Không ép split khi biển nghiêng. **Chưa thử/thừa nhận split OCR tốt hơn whole OCR.**

[A offline layout overlay](A-offline-layout-overlay.jpg) · [B offline layout overlay](B-offline-layout-overlay.jpg) · [Offline replay đầy đủ](offline-layout-replay.json). Overlay minh họa lại boxes cũ, không phải output một lượt OCR mới.

## 5. Before / candidate replay / fresh validation

| Evidence | A raw engine | A grouped lines / normalization | A final | B final |
|---|---|---|---|---|
| Admitted/completed baseline | C;29-B1;542.;70 | C /542.29-B170→C /54229B170 | rỗng, unrecognized | 17AK-044.70 |
| Offline replay duy nhất | **Dùng nguyên fragments cũ**, engine_rerun=False | 29-B1 /542.70→29B1 /54270 | Candidate string29-B1 542.70, structured, corrections=[] | Candidate string17AK-044.70, corrections=[] |
| Fresh after-A attempt | **Không có: admission từ chối** | Không chạy grouping/formatter recognition | rỗng, ocr_runtime_error, not_completed | Không chạy tiếp B theo stop condition |

Candidate string khớp text người dùng là bằng chứng sửa layout trên **raw trace cố định**, không phải bằng chứng recognizer mới đọc đúng. Không tuyên bố accuracy tăng hoặc A đã sửa xong trong production.

[after-A/trace.json](after-A/trace.json) ghi stage memory_admission, allowed=False, available1.733.201.920 <required2.257.584.128 bytes; [runtime traceback](after-A/runtime.log) và [run status](runs.json). Process exit0 vì exception được phân loại đúng. Đây không phải empty_ocr và không phải formatting rejection. Không đổi guard để chạy tiếp; không thử OCR thêm lần nào sau refusal.

## 6. Production state, tests và file bàn giao

Candidate đã được áp dụng tạm để gọi after-A, nhưng không vào recognition vì guard. Đã archive source+tests và diff, rồi **khôi phục postprocessing.py và test_postprocessing.py byte-for-byte**, gỡ module text_layout.py khỏi production. Mọi protected baseline hash đều trùng; weights SHA256 giữ nguyên. Không để bản sửa thiếu live regression trong source đang chạy.

**'''+test_result+'''**: 95 current unit/API tests +10 offline candidate tests. Lệnh cố ý **exclude `tests/test_inference_smoke.py`** để không khởi tạo thêm model/chạy OCR sau stop condition; không báo là full model regression. [Log](offline-tests.log).

Candidate tests: A/B raw fixture→user-confirmed expected text, không sửa characters, reading order độc lập input permutation/scale/translation, confidence không chi phối layout, giữ small inline glyph, không loại row khi chỉ có một main box, không tự sinh text từ empty input. Expected text chỉ nằm trong test assertions, không đi vào production layout/formatter. [Offline test runner](proposed/test_text_layout.py); patch còn kèm production-ready `tests/test_text_layout.py` và fixture nếu được áp dụng sau này.

File mới trong task: `scripts/report_plate_A_recognition.py`, `report.md`, `proposed.patch`, `proposed/`, `after-A/`, `offline-layout-replay.json`, hai offline overlay JPG, `offline-tests.log`, `integrity.json`. Các script/baseline/candidate đã có trước được giữ lại; `runs.json` chỉ append lượt after-A. Không ghi đè baseline report khác hoặc annotation.

**Rủi ro candidate chưa được đánh giá rộng:** nhiều box nền có kích thước tương đương chữ biển hoặc row có chênh font quá lớn có thể vẫn group/exclude sai; chưa có đủ bằng chứng để áp dụng rộng. Ngưỡng geometry là heuristic có trace, không phải chứng nhận text. Cần live admitted A/B regression trước khi apply proposed.patch; task này không mở rộng dataset hoặc thử thêm phương án.
'''
    (OUT/'report.md').write_text(report,encoding='utf-8')
    print(json.dumps({'report':str(OUT/'report.md'),'protected_changes':changed,'production_patch_applied':False,
        'live_A_status':failed['ocr_status'],'candidate_only':{k:v['formatted']['plate_number'] for k,v in replay.items()},'tests':test_result}))


if __name__=='__main__':main()

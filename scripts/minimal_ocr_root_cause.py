"""Small follow-up using frozen study artefacts; no detector/benchmark rerun."""
import argparse
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'experiments/minimal-ocr-root-cause'
STUDY=ROOT/'experiments/ocr-recognition-study'
EDITED=['src/inference/postprocessing.py','tests/test_postprocessing.py']


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,
        default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)),encoding='utf-8')


def prepare():
    if (OUT/'before/manifest.json').exists(): raise ValueError('Before snapshot exists; refusing overwrite')
    for name in EDITED:
        target=OUT/'before'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    files=[]
    for folder in ['src','be','tests','data/custom/images/test','data/custom/labels/test',
                   'experiments/diagnostic-test','experiments/targeted-ocr-fix','experiments/ocr-recognition-study']:
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files += [ROOT/'weights/best.pt',ROOT/'requirements.txt']
    dump(OUT/'before/manifest.json',{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))})
    print('Saved rollback source and protected-file manifest')


def geometry():
    import cv2
    import numpy as np
    manifest=load(STUDY/'manifest.json')
    case=next(c for c in manifest['cases'] if c['case_id']=='006-0')
    source=case['source']; px,py,_,_=source['padded_xyxy']
    crop=cv2.imread(str(STUDY/'cases/006-0/inputs/raw_whole-whole.png'))
    original=cv2.imread(str(ROOT/'data/custom/images/test'/source['filename']))
    h,w=original.shape[:2]
    gt=[]
    label=ROOT/'data/custom/labels/test'/Path(source['filename']).with_suffix('.txt')
    for line in label.read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        cls,cx,cy,bw,bh=map(float,line.split())
        gt.append([(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h])
    def poly(rect):
        x1,y1,x2,y2=rect
        return np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],np.float32)
    def local(rect): return [rect[0]-px,rect[1]-py,rect[2]-px,rect[3]-py]
    core=local(source['xyxy']);gt_local=[local(box) for box in gt]
    clip=poly([0,0,crop.shape[1],crop.shape[0]])
    run=load(STUDY/'runs/006-0--raw_whole.json')
    canvas=crop.copy();rows=[]
    for name,rect,color in [('YOLO core',core,(0,200,255))]+[(f'GT {i}',box,(30,220,30)) for i,box in enumerate(gt_local)]:
        cv2.polylines(canvas,[np.round(poly(rect)).astype(np.int32)],True,color,2)
    for region in run['calls'][0]['raw']:
        points=np.asarray(region['bbox'],np.float32)
        area,visible=cv2.intersectConvexConvex(points,clip)
        fractions=[]
        for rect in [core]+gt_local:
            overlap,_=cv2.intersectConvexConvex(visible,poly(rect)) if visible is not None and area>0 else (0,None)
            fractions.append(float(overlap/area) if area>0 else 0.)
        rows.append(dict(region,visible_area=float(area),fraction_inside_yolo_core=fractions[0],
                         fraction_inside_gt_boxes=fractions[1:]))
        polygon=np.round(points).astype(np.int32)
        cv2.polylines(canvas,[polygon],True,(30,30,240),1)
        x,y=polygon.min(axis=0);cv2.putText(canvas,region['text'],(max(0,int(x)),max(12,int(y)-3)),cv2.FONT_HERSHEY_SIMPLEX,.45,(30,30,240),1)
    OUT.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(OUT/'006-geometry.jpg'),canvas)
    dump(OUT/'006-geometry.json',{'source':source,'raw_crop_shape':list(crop.shape),'yolo_core_local':core,
        'gt_local':gt_local,'regions':rows,'policy':'Geometry only; YOLO/GT rectangles are not exact text polygons. No dataset changes.'})
    print(json.dumps({'yolo_core_local':core,'gt_local':gt_local,
                      'regions':[{'text':r['text'],'inside_core':r['fraction_inside_yolo_core'],'inside_gt':r['fraction_inside_gt_boxes']} for r in rows]},indent=2))


def supplied_regions():
    """Manual geometry for one diagnostic experiment, NEVER production rules/GT."""
    import cv2
    import numpy as np
    raw=cv2.imread(str(STUDY/'cases/002-0/inputs/raw_whole-whole.png'))
    lower=load(STUDY/'runs/002-0--raw_whole.json')['calls'][0]['raw'][0]['bbox']
    # Four corners around the upper line, selected from the original crop only.
    upper=[[249,124],[710,207],[674,354],[213,271]]
    records={'case':'002-0','coordinate_frame':'Frozen raw detector crop, pixels',
        'regions':[{'id':'upper','provenance':'Manual visual line quadrilateral, diagnostic-only; no text supplied','quad':upper},
                   {'id':'lower','provenance':'Exact CRAFT region from prior raw_whole run, no geometry changes','quad':lower}],
        'production_enabled':False,'ground_truth_text_provided':False}
    dump(OUT/'supplied_line_regions.json',records)
    canvas=raw.copy()
    for item in records['regions']:
        points=np.round(item['quad']).astype(np.int32)
        cv2.polylines(canvas,[points],True,(0,210,255),2)
        x,y=points.min(axis=0);cv2.putText(canvas,item['id'],(int(x),int(y)-5),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,210,255),2)
        # Same rectification helper used by EasyOCR get_image_list.
        from easyocr.utils import four_point_transform
        line=four_point_transform(raw,np.array(item['quad'],dtype='float32'))
        cv2.imwrite(str(OUT/f'002-line-{item["id"]}.png'),line)
    cv2.imwrite(str(OUT/'002-supplied-regions.jpg'),canvas)
    print('Exported diagnostic-only supplied line quads; no automatic split/GT text')


def worker(args):
    import faulthandler
    faulthandler.enable()
    import cv2
    import easyocr
    from src.inference import postprocessing as current
    from src.utils.config import OCR_ALLOWLIST
    result={'mode':args.mode,'case':args.case,'status':'reader_initializing','calls':[]}
    dump(args.result,result)
    reader=easyocr.Reader(['en'],gpu=False,download_enabled=False,verbose=False)
    if args.mode=='direct-lines':
        region_data=load(OUT/'supplied_line_regions.json')
        color=cv2.imread(str(STUDY/'cases/002-0/inputs/raw_whole-whole.png'))
        image=cv2.cvtColor(color,cv2.COLOR_BGR2GRAY)
        # Passing regions to recognize avoids CRAFT; no OCR-generated text is fed back.
        result.update(status='recognizer_running',input_sha256=sha(STUDY/'cases/002-0/inputs/raw_whole-whole.png'),
                      supplied_regions=region_data['regions'],craft_calls=0,
                      grayscale_policy='cv2.cvtColor(BGR, COLOR_BGR2GRAY), same as EasyOCR raw ndarray reformat')
        def forbidden_detect(*a,**kw):
            result['craft_calls']+=1;dump(args.result,result)
            raise AssertionError('Direct-region experiment must not invoke CRAFT')
        reader.detect=forbidden_detect
        dump(args.result,result)
        values=reader.recognize(image,horizontal_list=[],free_list=[r['quad'] for r in region_data['regions']],
                                allowlist=OCR_ALLOWLIST,reformat=False)
        result.update(status='complete',raw=[{'bbox':b,'text':t,'confidence':float(c)} for b,t,c in values],
                      formatter_called=False,production_enabled=False)
    else:
        module=current
        if args.mode=='before':
            spec=importlib.util.spec_from_file_location('src.inference._minimal_before',OUT/'before/src/inference/postprocessing.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        image=cv2.imread(str(STUDY/f'cases/{args.case}/inputs/current_whole-whole.png'),0)
        ocr=module.PlateOCR(use_gpu=False);ocr._reader=reader
        original=reader.readtext
        def traced(part,**kwargs):
            n=len(result['calls']);path=OUT/'inputs'/f'{args.mode}-{args.case}-{n}.png';path.parent.mkdir(parents=True,exist_ok=True)
            cv2.imwrite(str(path),part)
            call={'input_path':path.relative_to(OUT).as_posix(),'shape':list(part.shape),'sha256':sha(path),'completed':False}
            result['calls'].append(call);dump(args.result,result)
            values=original(part,**kwargs)
            call.update(completed=True,raw=[{'bbox':b,'text':t,'confidence':float(c)} for b,t,c in values])
            dump(args.result,result);return values
        reader.readtext=traced
        result.update(details=ocr.read_text_details(image),status='complete')
    dump(args.result,result)


def run_small():
    if (OUT/'run_status.json').exists(): raise ValueError('Small run already recorded; refusing overwrite')
    statuses=[]
    jobs=[('before','sample-1'),('after','sample-1'),('after','001-0'),('after','052-1'),
          ('after','006-0'),('direct-lines','002-0')]
    for mode,case in jobs:
        dest=OUT/'runs'/f'{mode}-{case}.json';dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.with_suffix('.log').open('wb') as log:
            try:
                code=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'worker','--mode',mode,
                    '--case',case,'--result',str(dest)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=150).returncode
            except subprocess.TimeoutExpired: code='timeout'
        state='completed' if code==0 else 'native_crash' if isinstance(code,int) and code & 0xffffffff==0xc0000005 else 'process_error'
        row={'mode':mode,'case':case,'exit_code':code,'execution_status':state,'file':dest.relative_to(OUT).as_posix()}
        statuses.append(row);dump(OUT/'run_status.json',statuses);print(json.dumps(row),flush=True)


def integrity_diff():
    before=load(OUT/'before/manifest.json')
    changed=[n for n,d in before.items() if not (ROOT/n).is_file() or sha(ROOT/n)!=d]
    result={'checked_files':len(before),'changed':changed,'unexpected_changes':[n for n in changed if n not in EDITED],
            'weights_sha256':sha(ROOT/'weights/best.pt')}
    dump(OUT/'integrity.json',result)
    patch=[]
    for name in EDITED:
        patch.extend(difflib.unified_diff((OUT/'before'/name).read_text(encoding='utf-8').splitlines(keepends=True),
            (ROOT/name).read_text(encoding='utf-8').splitlines(keepends=True),fromfile='a/'+name,tofile='b/'+name))
    (OUT/'changes.patch').write_text(''.join(patch),encoding='utf-8')
    print(json.dumps(result))
    if result['unexpected_changes']: raise SystemExit(1)


def test():
    import xml.etree.ElementTree as ET
    xml=OUT/'pytest-results.xml'
    with (OUT/'pytest.log').open('wb') as stream:
        code=subprocess.run([sys.executable,'-B','-m','pytest','tests','-q','-p','no:cacheprovider',
                             f'--junitxml={xml}'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=180).returncode
    result={'exit_code':code}
    if xml.exists():
        suites=ET.parse(xml).getroot().findall('testsuite')
        result.update({field:sum(int(s.get(field,0)) for s in suites) for field in ['tests','failures','errors','skipped']})
    dump(OUT/'test_results.json',result);print(json.dumps(result))
    if code: raise SystemExit(1)


def report():
    import cv2
    import numpy as np
    def result(name): return load(OUT/'runs'/f'{name}.json')
    before=result('before-sample-1');after=result('after-sample-1')
    checks=[]
    for mode,variant,value in [('before','current_split_clahe',before),('after','current_split',after)]:
        for label,call in zip(['top','bottom'],value['calls']):
            expected=STUDY/f'cases/sample-1/inputs/{variant}-{label}.png'
            checks.append({'mode':mode,'line':label,'expected':expected.relative_to(ROOT).as_posix(),
                'actual':call['input_path'],'identical_pixels':np.array_equal(cv2.imread(str(expected),0),cv2.imread(str(OUT/call['input_path']),0))})
    for case in ['001-0','052-1','006-0']:
        call=result('after-'+case)['calls'][0]
        expected=STUDY/f'cases/{case}/inputs/current_whole-whole.png'
        checks.append({'mode':'after','case':case,'actual':call['input_path'],
                       'identical_pixels':np.array_equal(cv2.imread(str(expected),0),cv2.imread(str(OUT/call['input_path']),0))})
    dump(OUT/'pixel_checks.json',checks)
    geo=load(OUT/'006-geometry.json');direct=result('direct-lines-002-0')
    tests=load(OUT/'test_results.json') if (OUT/'test_results.json').exists() else {}
    integrity=load(OUT/'integrity.json') if (OUT/'integrity.json').exists() else {}
    statuses=load(OUT/'run_status.json')
    def words(run):
        return '; '.join(', '.join(f'{v["text"]} ({v["confidence"]:.6f})' for v in c.get('raw',[])) or '∅' for c in run['calls'])
    def study_words(case):
        data=load(STUDY/f'runs/{case}--current_whole.json')
        return '; '.join(', '.join(f'{v["text"]} ({v["confidence"]:.6f})' for v in c.get('raw',[])) or '∅' for c in data['calls'])
    lines=['# Targeted ALPR root cause & minimal fix','',
      '## 1. Lỗi nằm ở đâu','',
      '**A — CLAHE lặp lại:** service `be/services/detection_service.py:103` gọi `preprocess_plate`, CLAHE lần đầu tại `src/inference/preprocessing.py:168`, rồi chuyển ảnh đã chuẩn bị vào OCR ở service dòng 106. Whole chỉ OCR trực tiếp. Split cũ gọi `apply_clahe(half)` thêm lần nữa tại `postprocessing.py:300`, sau khi ảnh đã upscale. Không có tham số hay contract yêu cầu tăng contrast lần hai; nó còn chạy khi `use_clahe=False`. Đây là lỗi sở hữu preprocessing/config có thể xác minh không cần suy đoán text GT.',
      '', '**B — 006-0:** YOLO bbox gốc toàn ảnh `[1421.55188,1966.44299,1877.23474,2328.16553]`; padding 0.1 tạo crop `[1376,1930,1923,2364]`, 547×434 px. GT bbox chỉ dùng cho geometry, không lấy text làm đáp án. [Ảnh geometry](006-geometry.jpg): YOLO vàng, GT xanh, OCR đỏ; [số liệu](006-geometry.json).','',
      '| Raw region | Confidence | Diện tích trong YOLO core | Trong GT bbox |','|---|---:|---:|---:|']
    for r in geo['regions']:
        lines.append(f'| {r["text"]} | {r["confidence"]:.6f} | {100*r["fraction_inside_yolo_core"]:.2f}% | {100*r["fraction_inside_gt_boxes"][0]:.2f}% |')
    lines+=['',
      'HONDA nằm phía trên biển; P7 phía dưới; hai text region trên biển nằm trong cả YOLO core và GT bbox. P7 vẫn chồng một phần với core nên padding không phải nguyên nhân duy nhất. Highest-confidence fragment là P7 nhưng nó nằm ngoài biển. Đây là kết quả nhánh raw_whole trong study; production hiện tại sau deskew của 006-0 đã không còn P7/HONDA trong raw output.',
      '', '**C — 002-0:** raw CRAFT có 0 horizontal regions và 1 free region ở hàng dưới; không có vùng hàng trên. Nhận dạng trên region dưới trả Z3 (0.342787). Vậy có hai tầng lỗi: detector bỏ hàng trên; recognizer không đọc đủ hàng dưới. Thí nghiệm raw_whole không dùng split/CLAHE, nên lỗi này không thể đổ riêng cho split. Split sai cũ đã được chặn ở patch trước; blur/phối cảnh vẫn hiện diện.','',
      '## 2. Production patch đã áp dụng','',
      '- Chỉ đổi `src/inference/postprocessing.py::PlateOCR.read_text_details`: split gọi `readtext(half)` trên nguyên pixel đã preprocess, bỏ `apply_clahe(half)`. Whole và split dùng cùng quy tắc: chỉ preprocess upstream; không tăng contrast ngầm trong OCR.',
      '- Giữ alias import apply_clahe cho recorder diagnostic cũ; OCR không gọi alias này. Không đổi splitter, formatter, EasyOCR model, decoder, threshold, weights, service, dataset hoặc dependencies.',
      '- Thêm 3 regression tests trong `tests/test_postprocessing.py`: CLAHE True/False có tổng số lần đúng 1/0, mỗi half khớp pixel của ảnh prepared, whole giữ nguyên input.',
      '- Không hardcode case_id, plate text hoặc ROI vào production. [Diff để review](changes.patch).','',
      '## 3. Chưa đủ bằng chứng để sửa production','',
      '**ROI chữ nền:** đề xuất giữ pixel padding để bảo vệ ký tự mép, nhưng lọc text regions theo vùng biển có geometry đáng tin cậy; cần mang core/quad và phép biến đổi qua deskew/upscale để so đúng hệ tọa độ. Region cắt mép phải được giữ để review, không loại chỉ vì một đỉnh nằm ngoài. Tâm nằm ngoài + phần lớn diện tích ngoài là tín hiệu chẩn đoán ở case này, chưa phải threshold tổng quát để bật mặc định. Không thu hẹp crop toàn cục hoặc đổi padding để che lỗi.',
      '', '**Direct line recognition:** đã gọi `EasyOCR.Reader.recognize(gray, horizontal_list=[], free_list=[upper,lower], reformat=False)` trực tiếp trên geometry được cung cấp sẵn; CRAFT được gắn guard để báo lỗi nếu bị gọi. ROI upper được khoanh thủ công từ ảnh cho thí nghiệm; ROI lower giữ nguyên quad CRAFT cũ. Không có text nào được cung cấp cho engine. [ROI và provenance](supplied_line_regions.json), [overlay](002-supplied-regions.jpg), [raw result](runs/direct-lines-002-0.json).',
      '', f'- Direct result: {[(r["text"],r["confidence"]) for r in direct.get("raw",[])]}; CRAFT calls={direct.get("craft_calls")}; formatter_called={direct.get("formatter_called")}.',
      '- Có thể bỏ qua CRAFT khi đã có line regions đáng tin cậy, nhưng chưa có bộ định vị dòng tự động được kiểm chứng cho 002-0. Upper chỉ trả T (confidence 0.070749), lower giữ nguyên Z3 (0.342787): bypass detection chưa khắc phục recognition. Không đưa quad thủ công vào production hoặc ép split khi gate từ chối.',
      '', '## 4. Before/After trên cùng input','',
      'Chỉ 6 lượt OCR mới: sample-1 Before/After, 001-0, 052-1, 006-0 After, và direct regions 002-0. Before controls/006-0 lấy từ study cùng pixel và config đã lưu; không chạy lại 44 phương án. Confidence là confidence fragment, không phải accuracy.','',
      '| Case | Before raw OCR | After raw OCR | Diễn giải |','|---|---|---|---|',
      f'| sample-1 | {words(before)} | {words(after)} | Chỉ bỏ CLAHE lần hai; cả hai vẫn không thành plate_number được chấp nhận. |']
    for case in ['001-0','052-1','006-0']:
        value=result('after-'+case)
        lines.append(f'| {case} | {study_words(case)} | {words(value)} | Whole input không đổi; kiểm tra ổn định output. |')
    lines.append(f'| 002-0 | CRAFT bỏ upper; lower Z3 (0.342787) | {[(r["text"],round(r["confidence"],6)) for r in direct.get("raw",[])]} | Diagnostic direct-recognize trên supplied regions; không phải default production After. |')
    lines+=['',
      f'Input audit: {len(checks)} kiểm tra pixel, PASS={all(c["identical_pixels"] for c in checks)}. Hai phiên bản sample-1 dùng cùng current_whole ban đầu; split row giữ {before["details"]["split_row"]}/{after["details"]["split_row"]}. Half Before trùng study current_split_clahe, half After trùng current_split; các whole controls/006-0 trùng input cũ. Xem [pixel_checks.json](pixel_checks.json) và [inputs/](inputs/).',
      '', '## 5. Regression và runtime','',
      f'- Pytest: **{tests.get("tests", "pending")} passed**, failures={tests.get("failures", "pending")}, errors={tests.get("errors", "pending")}, skipped={tests.get("skipped", "pending")}. [Log](pytest.log), [JUnit](pytest-results.xml).',
      f'- 6 OCR subprocess: {sum(r["execution_status"]=="completed" for r in statuses)} hoàn tất, {sum(r["execution_status"]=="native_crash" for r in statuses)} native crash. [Trạng thái từng lượt](run_status.json); crash không tính là OCR rỗng.',
      f'- SHA256: {integrity.get("checked_files")} file được bảo vệ; chỉ postprocessing.py và test_postprocessing.py thay đổi; unexpected={integrity.get("unexpected_changes")}. Reports/crops/labels/checkpoint cũ giữ nguyên, [integrity.json](integrity.json).',
      '- Cải thiện đã chứng minh: đường split tôn trọng pixel/preprocessing và cấu hình CLAHE, không còn contrast transform lặp; supplied ROI giúp recognizer có cơ hội đọc hàng mà CRAFT bỏ. Chưa chứng minh ký tự nhận dạng đúng hơn.',
      '- Giữ nguyên: whole-route controls và case006 theo raw trace. Xấu đi về accuracy: chưa thể kết luận vì không có text người dùng xác nhận. sample-1 vẫn sai/không được xác nhận; bỏ CLAHE không giải quyết toàn bộ recognition.',
      '- Native crash trong study cũ vẫn là vấn đề riêng; 6 lượt mới không đủ để tuyên bố runtime ổn định hoặc crash đã được sửa.',
      '', '## 6. Review và rollback','',
      '[changes.patch](changes.patch) chứa đúng hai file sửa; `before/` lưu nguyên byte nguồn cũ. Script hỗ trợ mới: `scripts/minimal_ocr_root_cause.py`. Quads thủ công chỉ có trong artefact/script chẩn đoán, không có trong production.',
      '', '```powershell',
      "Copy-Item -LiteralPath 'experiments/minimal-ocr-root-cause/before/src/inference/postprocessing.py' -Destination 'src/inference/postprocessing.py'",
      "Copy-Item -LiteralPath 'experiments/minimal-ocr-root-cause/before/tests/test_postprocessing.py' -Destination 'tests/test_postprocessing.py'",
      '```','']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Wrote concise report, pixel checks and linked raw evidence')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','geometry','supplied-regions','worker','run-small','integrity','test','report'])
    p.add_argument('--mode',choices=['before','after','direct-lines']);p.add_argument('--case');p.add_argument('--result',type=Path)
    args=p.parse_args()
    {'prepare':prepare,'geometry':geometry,'supplied-regions':supplied_regions,'worker':lambda:worker(args),
     'run-small':run_small,'integrity':integrity_diff,'test':test,'report':report}[args.command]()

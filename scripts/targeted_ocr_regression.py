"""Small targeted OCR regression; never scores accuracy or writes baseline/data.

Uses saved diagnostic detector crops to hold detection/crop/preprocessing fixed.
Each EasyOCR run is isolated because earlier diagnostic runs had native crashes.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'experiments/targeted-ocr-fix'
BASE=ROOT/'experiments/diagnostic-test'
EDITED=['src/inference/preprocessing.py','src/inference/postprocessing.py',
        'tests/test_preprocessing.py','tests/test_postprocessing.py']
CASES=[('002-0','img_0026','split failure'),('052-0','img_0559','partial OCR mapped to plausible text'),
       ('024-0','img_0204','unbounded series'),('006-0','img_0045','serial truncation'),
       ('014-0','img_0100','previous nonempty two-line control'),
       ('001-0','img_0007','previous nonempty one-line control'),
       ('003-0','img_0028','previous empty control'),
       ('027-0','img_0226','partial characters coerced by mappings'),
       ('057-0','img_0605','crop coverage and empty OCR'),
       ('005-0','img_0033','previous native operational failure'),
       ('052-1','img_0559','additional previously nonempty two-line control')]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)),encoding='utf-8')


def snapshot():
    if (OUT/'before/manifest.json').exists(): raise RuntimeError('Before snapshot already exists; refusing overwrite')
    OUT.mkdir(parents=True,exist_ok=True)
    for name in EDITED:
        dest=OUT/'before'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    protected=[]
    for name in ['data/custom/images/test','data/custom/labels/test','experiments/diagnostic-test','src','be','tests']:
        protected.extend(p for p in (ROOT/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    protected.extend(p for p in [ROOT/'weights/best.pt',ROOT/'src/utils/config.py',ROOT/'requirements.txt',ROOT/'assets/samples/1.jpg'] if p.exists())
    dump(OUT/'before/manifest.json',{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted(set(protected))})
    print('Snapshot saved before production edits',flush=True)


def old_modules():
    modules=[]
    for name in ['preprocessing','postprocessing']:
        spec=importlib.util.spec_from_file_location('src.inference._before_'+name,OUT/'before/src/inference'/f'{name}.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);modules.append(module)
    # The old formatter normally imports the production split function: pin old helpers.
    modules[1].split_two_lines=modules[0].split_two_lines
    modules[1].is_two_line=modules[0].is_two_line
    return modules


def analyze():
    import cv2
    import numpy as np
    old,_=old_modules()
    rows=[]
    for identifier,filename,purpose in CASES:
        image=cv2.imread(str(BASE/f'visualizations/preprocessed/{identifier}.png'),0)
        h,w=image.shape;ink=cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
        profile=(ink>0).mean(axis=1);row=old.split_two_lines(image)[0].shape[0]
        rows.append({'id':identifier,'filename':filename,'purpose':purpose,'shape':[h,w],
          'old_split_row':row,'old_row_ink_fraction':float(profile[row]),
          'profile_quantiles':np.quantile(profile,[0,.1,.25,.5,.75,.9,1]).tolist(),
          'middle_min_fraction':float(profile[int(.3*h):int(.7*h)].min())})
        dump(OUT/'profiles'/f'{identifier}.json',{'profile':profile.tolist(),'old_split_row':row})
    dump(OUT/'split_analysis_before.json',rows);print(json.dumps(rows,indent=2))


def worker(args):
    import faulthandler
    faulthandler.enable()
    import cv2
    from src.inference.preprocessing import preprocess_plate
    from src.inference import postprocessing as current
    pre,post=old_modules() if args.version=='before' else (None,current)
    import easyocr
    reader=easyocr.Reader(['en'],gpu=False,download_enabled=False,verbose=False)
    ocr=post.PlateOCR(languages=['en'],use_gpu=False);ocr._reader=reader
    result={'version':args.version,'case':args.case,'calls':[]}
    original=reader.readtext
    def traced(image,**kwargs):
        call={'input_shape':list(image.shape),'started':True};result['calls'].append(call)
        dump(args.result,result)
        values=original(image,**kwargs)
        call.update(completed=True,raw=[{'bbox':box,'text':text,'confidence':float(conf)} for box,text,conf in values])
        dump(args.result,result);return values
    reader.readtext=traced
    if args.case=='sample-1':
        # Fixed detector box from ketqua.md, same padding/config; no detector rerun.
        image=cv2.imread(str(ROOT/'assets/samples/1.jpg'))
        from src.inference.preprocessing import crop_plate
        processor=pre.preprocess_plate if pre else preprocess_plate
        image=processor(crop_plate(image,[109.10308837890625,0.,597.3599243164062,392.8330078125],padding=.1))
    else: image=cv2.imread(str(BASE/f'visualizations/preprocessed/{args.case}.png'),0)
    if image is None: raise RuntimeError('Missing image')
    if args.version=='after':
        details=ocr.read_text_details(image);result.update(details=details,final=details['plate_number'])
    else: result['final']=ocr.read_text(image)
    result['completed']=True;dump(args.result,result)


def run_cases(tasks,statuses):
    OUT.mkdir(parents=True,exist_ok=True)
    for version,case,suffix in tasks:
            result=OUT/'runs'/f'{version}-{case}{suffix}.json';log=result.with_suffix('.log');log.parent.mkdir(parents=True,exist_ok=True)
            start=time.monotonic()
            with log.open('w',encoding='utf-8') as handle:
                command=[sys.executable,'-B',str(Path(__file__).resolve()),'worker','--case',case,'--version',version,'--result',str(result)]
                try:
                    completed=subprocess.run(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,timeout=180)
                    code=completed.returncode
                except subprocess.TimeoutExpired: code='timeout'
            row={'version':version,'case':case,'attempt':'retry1' if suffix else 'initial',
                 'result_file':str(result.relative_to(OUT)).replace('\\','/'),
                 'exit_code':code,'seconds':round(time.monotonic()-start,2)}
            statuses.append(row);dump(OUT/'run_status.json',statuses);print(json.dumps(row),flush=True)


def run():
    run_cases([(version,case,'') for version in ['before','after'] for case in [c[0] for c in CASES]+['sample-1']],[])


def followup():
    statuses=json.loads((OUT/'run_status.json').read_text(encoding='utf-8'))
    # One sequential retry only; preserve first native/memory failure logs.
    tasks=[(version,case,'-retry1') for version,case in [('before','003-0'),('after','027-0')]]
    tasks += [(version,'052-1','') for version in ['before','after']]
    if any(r.get('attempt')=='retry1' for r in statuses): raise RuntimeError('Followup already attempted')
    run_cases(tasks,statuses)


def integrity():
    before=json.loads((OUT/'before/manifest.json').read_text(encoding='utf-8'))
    changed=[name for name,digest in before.items() if not (ROOT/name).exists() or sha(ROOT/name)!=digest]
    dump(OUT/'integrity.json',{'checked_files':len(before),'authorized_changes':EDITED,
         'changed':changed,'unexpected_changes':[name for name in changed if name not in EDITED],
         'weights_sha256':sha(ROOT/'weights/best.pt')})
    print(json.dumps({'changed':changed,'unexpected_changes':[n for n in changed if n not in EDITED]}))


def report():
    import ast
    import difflib
    from src.inference.postprocessing import format_vn_plate_details
    _,old=old_modules()
    replays=[]
    for lines in [['L8953'],['68-61','G6Q'],['3BA','531032'],['29-PI','239.641'],
                  ['54K697','02L52'],['68-G1','668.86'],['30A53103'],['59-CL','05478'],
                  ['68-61','668.86'],['68-G1','66S86']]:
        replays.append({'raw_lines':lines,'before':old.format_vn_plate(lines),'after':format_vn_plate_details(lines)})
    dump(OUT/'formatter_replay.json',replays)
    patch=[]; locations={}
    for name in EDITED:
        before=(OUT/'before'/name).read_text(encoding='utf-8');after=(ROOT/name).read_text(encoding='utf-8')
        patch.extend(difflib.unified_diff(before.splitlines(keepends=True),after.splitlines(keepends=True),fromfile='a/'+name,tofile='b/'+name))
        locations[name]={phase:{node.name:node.lineno for node in ast.walk(ast.parse(source)) if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
                         for phase,source in [('before',before),('after',after)]}
    (OUT/'changes.patch').write_text(''.join(patch),encoding='utf-8')
    dump(OUT/'function_locations.json',locations)
    import xml.etree.ElementTree as ET
    test_results={}
    if (OUT/'pytest-results.xml').exists():
        suites=ET.parse(OUT/'pytest-results.xml').getroot().findall('testsuite')
        test_results={name:sum(int(s.get(name,0)) for s in suites) for name in ['tests','failures','errors','skipped']}
        test_results['full_suite_includes_real_model_smoke']=True
        test_results['prior_attempt_notes']=[
            'An earlier full-suite attempt aborted in CRAFT/max_pool2d while another OCR process ran.',
            'before-003-0 initial model allocation failed due insufficient RAM; isolated retry succeeded.',
            'No production threading/dependencies changed. Final full suite was run alone.']
        dump(OUT/'test_results.json',test_results)
    checks=json.loads((OUT/'integrity.json').read_text(encoding='utf-8')) if (OUT/'integrity.json').exists() else {}
    statuses=json.loads((OUT/'run_status.json').read_text(encoding='utf-8')) if (OUT/'run_status.json').exists() else []
    def get_run(version,case):
        state=next((s for s in reversed(statuses) if s['version']==version and s['case']==case),{})
        path=OUT/state.get('result_file',f'runs/{version}-{case}.json')
        value=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        return dict(value,execution=state)
    comparisons=[]
    for case,prefix,purpose in CASES+[('sample-1','assets/samples/1.jpg','observed audit failure')]:
        comparisons.append({'case':case,'image':prefix,'purpose':purpose,
                            'before':get_run('before',case),'after':get_run('after',case)})
    dump(OUT/'before_after.json',comparisons)
    # Annotate a COPY of the processed image, never change diagnostic pixels.
    import cv2
    image=cv2.imread(str(BASE/'visualizations/preprocessed/002-0.png'))
    if image is not None:
        image=cv2.resize(image,(806,629),interpolation=cv2.INTER_AREA)
        cv2.line(image,(0,401),(805,401),(0,0,255),2)
        cv2.putText(image,'OLD split 802/1258 - NEW: whole-crop OCR',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,255),2)
        cv2.imwrite(str(OUT/'split-0026-review.png'),image)
    def raw(run):
        return [[v['text'] for v in call.get('raw',[])] for call in run.get('calls',[])]
    def final(run):
        return repr(run.get('final','')) if run.get('completed') else f'NOT COMPLETED (exit={run.get("execution",{}).get("exit_code")})'
    lines=['# Targeted ALPR OCR bug fix','',
      '## 1. Root causes và vị trí','',
      '- Trước patch, `preprocessing.py:190–211` lấy argmin Otsu row projection trong 30–70% chiều cao, luôn trả hai nửa; aspect ratio <1.9 được dùng như quyết định bắt buộc split (`postprocessing.py:273`). Không có chứng cứ gap, độ rộng gap hoặc kiểm tra nét chữ băng qua đường cắt.',
      '- Với detector crop img_0026 đã lưu: 1612×1258, argmin ở row 802; 659/1612 pixel (40.8809%) còn foreground. Cực tiểu này không phải khoảng trắng giữa hai dòng. Visual inspection trong diagnostic xác nhận nó cắt hàng dưới. Otsu projection còn bị ảnh hưởng bởi padding/background và phối cảnh; không kết luận đây là nguyên nhân duy nhất OCR rỗng.',
      '- `L8953`: `_clean` giữ L8953 → `_repair_province` L→1 thành 18953 → `_repair_series_single` ký tự index 2 là 9→P → 18P53. Không kiểm tra serial đầy đủ trước correction.',
      '- `68-61` + `G6Q`: `_clean` → 6861 + G6Q; `_repair_series_top` index 2, 6→G → 68G1; `_format_number` G→6, giữ 6, Q→0 → 660; formatter trả 68-G1 660 dù serial chỉ 3 ký tự.',
      '- `_format_number` trước patch còn bỏ ký tự không map được và cắt serial trên 5 chữ số; `_cap_length` cắt tổng alphanumeric ở một số nhánh. Regex cũ chấp nhận province 2–3 số, series top không giới hạn và serial >=4 nên không kiểm tra completeness nhất quán.','',
      '## 2. Thay đổi production','',
      '- `src/inference/preprocessing.py`: aspect chỉ là candidate; `find_split_row` yêu cầu một dải ít foreground liên tục giữa hai vùng ink đủ cao, bỏ biên ngang, không chấp nhận component lớn băng qua đường cắt. Không đủ bằng chứng → None. `split_two_lines` có thể trả None; caller production duy nhất đã được cập nhật.',
      '- `src/inference/postprocessing.py`: OCR toàn crop khi split bị từ chối; thiếu một half OCR cũng fallback toàn crop. Giữ nguyên deskew, CLAHE và việc áp CLAHE cho half đã chấp nhận; không thay detector/config.',
      '- `_clean` chỉ uppercase/ASCII normalization. `_format_number` chỉ thêm dấu chấm cho đúng 5 số; `_cap_length` giữ nguyên text. Không thêm/xóa alphanumeric để ép format.',
      '- Regex giới hạn family hiện hỗ trợ: province 2 số, series 1–2 chữ hoặc chữ+số, serial 4/5 số. Đây là nhận dạng cấu trúc thông dụng, không phải kiểm tra pháp lý hay xác nhận biển số đúng. Family khác được giữ trong trace nhưng chưa trả như plate hợp lệ.',
      '- Chỉ xét tối đa một correction có đầy đủ các slot còn lại và chỉ chấp nhận một candidate duy nhất. Hai dòng có thể dùng ngữ cảnh top/serial rõ ràng; merged single-line chỉ sửa province nếu suffix đầy đủ. Không còn ép chữ tại index 2 trên mọi chuỗi.',
      '- `format_vn_plate_details` tách raw, normalized, corrected, corrections, formatted text, format_status và plate_number. `text_verified` luôn False. `read_text_details` giữ OCR boxes/confidences/inputs và split decision theo từng call, không dùng shared mutable last_result.',
      '- Text thiếu/không hỗ trợ: formatted_text giữ evidence normalized, plate_number trả rỗng. `read_text` giữ API string; log raw/final và corrections. API schema vẫn như cũ; detector confidence không phải OCR confidence.','',
      '## 3. Controlled formatter replay (không phải ground truth)','',
      '| Raw input | Before | After formatted evidence | After plate_number | Status |','|---|---|---|---|---|']
    for r in replays:
        a=r['after'];lines.append(f'| `{r["raw_lines"]}` | `{r["before"]}` | `{a["formatted_text"]}` | `{a["plate_number"]}` | {a["format_status"]} |')
    lines+=['','## 4. OCR Before/After trên ảnh','',
       'Giữ nguyên các detector crops và ảnh preprocessed diagnostic; chạy lại EasyOCR cùng weights/language/CPU/allowlist, download disabled. Không rerun detector cho bảng này. Ảnh mẫu 1.jpg dùng bbox/padding trong ketqua.md rồi preprocessing nguyên cấu hình. Before dùng module gốc đã snapshot; After dùng code patch. Mỗi crop/phiên bản có subprocess riêng.','',
       '| Crop / ảnh | Before OCR calls (raw) | Before plate | After OCR calls (raw) | After plate | After status/split |',
       '|---|---|---|---|---|---|']
    for r in comparisons:
        b,a=r['before'],r['after'];d=a.get('details',{})
        lines.append(f'| {r["case"]} / {r["image"]} | `{raw(b)}` | {final(b)} | `{raw(a)}` | {final(a)} | {d.get("format_status", "not completed")} / {d.get("split_status", "n/a")} |')
    lines+=['','Raw trong bảng là từng OCR call, trước sắp xếp theo geometry; raw_lines đã ghép đúng thứ tự có trong before_after.json. Không dùng các raw/final outputs làm GT. Empty sau patch có thể là OCR thật rỗng, hoặc cấu trúc bị từ chối; phải xem details để phân biệt.','',
       '## 5. Tests và integrity','',
       f'- Lần chạy cuối: **{test_results.get("tests", "pending")} tests, {test_results.get("failures", "pending")} failures, {test_results.get("errors", "pending")} errors, {test_results.get("skipped", "pending")} skipped**, gồm smoke test YOLO/EasyOCR thật. Log: pytest-final.log; JUnit: pytest-results.xml.',
       f'- SHA256 kiểm tra {checks.get("checked_files", "pending")} file có sẵn: chỉ hai production modules và hai test files được phép thay đổi; unexpected changes: {checks.get("unexpected_changes", "pending")}. Model SHA256: {checks.get("weights_sha256", "pending")}.',
       '- Lượt full suite trước đó gặp native AV khi chạy song song OCR; before-003-0 từng thiếu RAM lúc khởi tạo reader. Chạy nặng tuần tự sau đó: full suite cuối PASS, retry before-003-0 hoàn tất. Không quy các lỗi vận hành này thành kết quả OCR. before-005-0 vẫn có AV được lưu; after-027-0 AV cả lần đầu và một retry, chưa được khắc phục.',
       '- Control một dòng 001-0 giữ nguyên 51KM-164.95; control hai dòng 052-1 giữ nguyên 59CL-054.78. Control 014-0 đổi 59MA-098.52 → 59MA-098.57 khi dùng whole crop; chưa có GT nên không chứng nhận đổi này tốt hơn hay xấu hơn.',
       '- Regression unit tests bao phủ no-gap, blank/tiny/single-row, minimum còn ink, gap quá mỏng, connected glyph, img_0026 saved crop, whole-crop/missing-half fallback, reading order, raw/normalization/correction trace, serial dài/thiếu và không âm thầm loại ký tự.',
       '- Hai test cũ được cập nhật có chủ đích: không còn kỳ vọng cắt serial sáu chữ số; split fixture phải có hai vùng ink thật thay cho ảnh đen chỉ có ratio.',
       '- Kết quả pytest và các lần thử runtime được ghi riêng trong test_results.json / pytest logs. Không coi runtime access violation hay thiếu RAM là sai/đúng OCR.','',
       '## 6. Giới hạn và ngoài phạm vi','',
       '- Không có text GT độc lập cho batch này, nên không tính CER/Exact Match và không tuyên bố accuracy tăng. Các control trước đây nonempty chỉ kiểm tra tính ổn định output, không phải chứng nhận đúng.',
       '- Native EasyOCR/PyTorch CRAFT max_pool2d access violation và tài nguyên RAM vẫn là vấn đề runtime riêng. Không sửa engine/threading/dependencies trong patch này.',
       '- Detector bbox/crop coverage, blur/occlusion, deskew, CLAHE lặp lại và adaptive line grouping hiện có vẫn có thể gây lỗi; không sửa ngoài hai vấn đề được yêu cầu.',
       '- Chọn gap bảo thủ có thể OCR toàn crop thường xuyên hơn và đổi kết quả trước đây. Đây là tradeoff có chủ đích để không cắt qua chữ khi không chắc; không chứng minh mọi split được chấp nhận đều đúng.',
       '- Một correction có cấu trúc vẫn là heuristic, không phải xác minh con người; format_status=heuristic_corrected được ghi rõ. Confidence OCR được giữ để phân tích nhưng không đặt thêm threshold để che lỗi.','',
       '## 7. Artefacts và rollback','',
       '- `before/`: bản gốc nguyên byte của hai production modules và hai test files; manifest SHA256 trước sửa.',
       '- `changes.patch`, `function_locations.json`: diff và vị trí function trước/sau.',
       '- `runs/`, `run_status.json`, `before_after.json`, `formatter_replay.json`: bằng chứng từng lượt, bao gồm lỗi vận hành.',
       '- `integrity.json`: các file thay đổi có chủ đích và kiểm tra dataset/labels/weights/baseline giữ nguyên.',
       '- `split-0026-review.png`: ảnh processed thu nhỏ có đánh dấu đường split cũ; diagnostic gốc giữ nguyên.',
       '- Rollback chỉ bốn file đã sửa; không phục hồi toàn thư mục dataset/baseline. Từ project root:','',
       '```powershell',
       *[f"Copy-Item -LiteralPath 'experiments/targeted-ocr-fix/before/{n}' -Destination '{n}'" for n in EDITED],
       '```','']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Wrote targeted report, controlled replay and rollback diff')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['snapshot','analyze','worker','run','followup','integrity','report'])
    parser.add_argument('--case');parser.add_argument('--version',choices=['before','after']);parser.add_argument('--result',type=Path)
    args=parser.parse_args()
    {'snapshot':snapshot,'analyze':analyze,'worker':lambda:worker(args),'run':run,'followup':followup,'integrity':integrity,'report':report}[args.command]()

"""Bounded input ablation on fixed saved detector crops; production read-only.

Every OCR candidate is complete and independent. No cross-candidate text merging,
format-based selection, prediction-as-GT, engine replacement or detector rerun.
"""
from __future__ import annotations
import argparse
import csv
import faulthandler
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'experiments/ocr-recognition-study'
BASE=ROOT/'experiments/diagnostic-test'
CASES=['002-0','006-0','014-0','024-0','052-0','sample-1','001-0','052-1']


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,
        default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)),encoding='utf-8')
    temporary.replace(path)


def load(path): return json.loads(path.read_text(encoding='utf-8-sig'))


def csv_write(path,rows,fields):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def prepare():
    import cv2
    import numpy as np
    from src.inference import preprocessing as pre
    from src.utils import config
    if (OUT/'manifest.json').exists(): raise RuntimeError('Frozen study already exists; do not overwrite inputs')
    OUT.mkdir(parents=True,exist_ok=True)
    paths=[]
    for folder in ['src','be','tests','data/custom/images/test','data/custom/labels/test',
                   'experiments/diagnostic-test','experiments/targeted-ocr-fix']:
        paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths += [ROOT/'weights/best.pt',ROOT/'requirements.txt',ROOT/'assets/samples/1.jpg']
    dump(OUT/'protected_before.json',{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(paths))})
    with (BASE/'ocr_results.csv').open(encoding='utf-8-sig',newline='') as f: baseline=list(csv.DictReader(f))
    manifest={'case_ids':CASES,'detector_rerun':False,'production_patch_applied':False,
              'runtime_config':{k:getattr(config,k) for k in config.__all__ if k.startswith('OCR_') or k=='CROP_PADDING'},
              'packages':{p:importlib.metadata.version(p) for p in ['easyocr','torch','torchvision','opencv-python','numpy']},
              'comparison_policy':'Fixed detector crop, complete independent candidates. Raw OCR/confidences saved before formatting. No ranking by plate format or OCR confidence alone.',
              'confirmed_reference_policy':'No user-confirmed text for these cases found at preparation. Prior report visual readings and predictions are not user-confirmed GT.',
              'cases':[]}
    for case in CASES:
        folder=OUT/'cases'/case/'inputs';folder.mkdir(parents=True,exist_ok=True)
        if case=='sample-1':
            source=ROOT/'assets/samples/1.jpg'; original=cv2.imread(str(source))
            box=[109.10308837890625,0.,597.3599243164062,392.8330078125]
            crop=pre.crop_plate(original,box,padding=config.CROP_PADDING)
            source_info={'filename':source.relative_to(ROOT).as_posix(),'source_sha256':sha(source),
                         'xyxy':box,'padding':config.CROP_PADDING,'provenance':'Fixed box recorded in ketqua.md; no detector inference'}
        else:
            source=BASE/f'visualizations/crops/{case}.png'
            crop=cv2.imread(str(source))
            matches=[r for r in baseline if json.loads(r['artifacts']).get('crops')==f'visualizations/crops/{case}.png']
            if len(matches)!=1: raise RuntimeError(f'Need one baseline mapping: {case}')
            r=matches[0]
            source_info={'filename':r['filename'],'source_artifact':source.relative_to(ROOT).as_posix(),
                'source_sha256':sha(source),'xyxy':json.loads(r['xyxy']),'padded_xyxy':json.loads(r['padded_xyxy']),
                'provenance':'Exact saved PNG of padded detector crop; never substitute GT crop'}
        if crop is None or not crop.size: raise RuntimeError(f'Missing crop {case}')
        gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        upscale=config.OCR_UPSCALE
        gray_up=cv2.resize(gray,None,fx=upscale,fy=upscale,interpolation=cv2.INTER_CUBIC)
        images={'raw_whole':crop,'gray_up_whole':gray_up,
                'no_deskew_whole':pre.preprocess_plate(crop,upscale=upscale,use_deskew=False),
                'no_clahe_whole':pre.preprocess_plate(crop,upscale=upscale,use_clahe=False),
                'current_whole':pre.preprocess_plate(crop,upscale=upscale)}
        case_info={'case_id':case,'source':source_info,'fixed_crop_shape':list(crop.shape),
                   'control':case in {'001-0','052-1'},'variants':[],'split_decisions':{}}
        def save_variant(name,parts,split_row=None):
            inputs=[]
            for label,image in parts:
                path=folder/f'{name}-{label}.png'
                if not cv2.imwrite(str(path),image): raise RuntimeError('Failed PNG export')
                inputs.append({'label':label,'path':path.relative_to(OUT).as_posix(),'shape':list(image.shape),
                               'sha256':sha(path),'pixel_sha256':hashlib.sha256(image.tobytes()).hexdigest()})
            case_info['variants'].append({'name':name,'inputs':inputs,'split_row':split_row})
        for name,image in images.items(): save_variant(name,[('whole',image)])
        # Full factorial is unnecessary: compare split vs whole on the same pixels
        # for raw, simple grayscale and current preprocessing only when gate accepts.
        for name in ['raw_whole','gray_up_whole','current_whole']:
            image=images[name]
            candidate=pre.is_two_line(image,config.OCR_TWO_LINE_ASPECT)
            halves=pre.split_two_lines(image) if candidate else None
            row=halves[0].shape[0] if halves is not None else None
            case_info['split_decisions'][name]={'aspect_candidate':candidate,'accepted':row is not None,'split_row':row}
            if halves is not None:
                split_name=name.replace('_whole','_split')
                save_variant(split_name,list(zip(('top','bottom'),halves)),row)
                if name=='current_whole':
                    save_variant('current_split_clahe',[(label,pre.apply_clahe(part)) for label,part in zip(('top','bottom'),halves)],row)
        current_decision=case_info['split_decisions']['current_whole']
        case_info['current_production_first_candidate']='current_split_clahe' if config.OCR_SPLIT_TWO_LINE and current_decision['accepted'] else 'current_whole'
        if case!='sample-1':
            saved=cv2.imread(str(BASE/f'visualizations/preprocessed/{case}.png'),0)
            case_info['current_pixels_match_saved_diagnostic']=saved is not None and np.array_equal(saved,images['current_whole'])
        manifest['cases'].append(case_info)
        # Navigation previews; input PNGs above are exact pixels sent to OCR.
        panels=[]
        for variant in case_info['variants']:
            for item in variant['inputs']:
                im=cv2.imread(str(OUT/item['path']))
                h,w=im.shape[:2];factor=min(390/w,270/h)
                small=cv2.resize(im,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA if factor<1 else cv2.INTER_NEAREST)
                panel=np.full((330,420,3),245,np.uint8)
                cv2.putText(panel,variant['name']+'/'+item['label'],(10,23),cv2.FONT_HERSHEY_SIMPLEX,.44,(20,20,20),1)
                cv2.putText(panel,f'{w}x{h} | split={variant["split_row"]}',(10,45),cv2.FONT_HERSHEY_SIMPLEX,.4,(20,20,20),1)
                x=(420-small.shape[1])//2;panel[55:55+small.shape[0],x:x+small.shape[1]]=small;panels.append(panel)
        sheet=np.full((330*((len(panels)+2)//3),1260,3),245,np.uint8)
        for i,panel in enumerate(panels): sheet[(i//3)*330:(i//3+1)*330,(i%3)*420:(i%3+1)*420]=panel
        cv2.imwrite(str(OUT/'cases'/case/'inputs_overview.jpg'),sheet)
    dump(OUT/'manifest.json',manifest)
    jobs=sum(len(c['variants']) for c in manifest['cases'])
    print(json.dumps({'cases':len(manifest['cases']),'jobs':jobs,'split_eligible':{c['case_id']:c['split_decisions'] for c in manifest['cases']}},indent=2))


def worker(case_id,variant_name,result_path):
    faulthandler.enable()
    import cv2
    from src.inference.postprocessing import PlateOCR,format_vn_plate_details
    manifest=load(OUT/'manifest.json')
    case=next(c for c in manifest['cases'] if c['case_id']==case_id)
    variant=next(v for v in case['variants'] if v['name']==variant_name)
    config=manifest['runtime_config']
    result={'case_id':case_id,'variant':variant_name,'calls':[],'status':'initializing_reader','formatter_called':False}
    dump(result_path,result)
    import easyocr
    reader=easyocr.Reader(config['OCR_LANGUAGES'],gpu=config['OCR_USE_GPU'],download_enabled=False,verbose=False)
    detect,recognize=reader.detect,reader.recognize
    def traced_detect(*args,**kwargs):
        call=result['calls'][-1];call['stage']='text_detection';dump(result_path,result)
        value=detect(*args,**kwargs)
        call.update(detector_completed=True,horizontal_boxes=value[0],free_boxes=value[1])
        dump(result_path,result);return value
    def traced_recognize(*args,**kwargs):
        call=result['calls'][-1];call['stage']='text_recognition';dump(result_path,result)
        value=recognize(*args,**kwargs)
        call['recognizer_completed']=True;dump(result_path,result);return value
    reader.detect=traced_detect;reader.recognize=traced_recognize
    grouped=[]
    for item in variant['inputs']:
        if sha(OUT/item['path'])!=item['sha256']: raise RuntimeError('Frozen OCR input changed')
        image=cv2.imread(str(OUT/item['path']),cv2.IMREAD_UNCHANGED)
        call={'input':item,'stage':'readtext_start','readtext_completed':False}
        result['calls'].append(call);result['status']='running';dump(result_path,result)
        started=time.monotonic()
        values=reader.readtext(image,allowlist=config['OCR_ALLOWLIST'])
        call.update(stage='complete',readtext_completed=True,seconds=round(time.monotonic()-started,4),
                    raw=[{'bbox':box,'text':text,'confidence':float(conf)} for box,text,conf in values])
        # Persist raw results before invoking normalization, repair or formatter.
        dump(result_path,result)
        if item['label']=='whole': grouped=PlateOCR._group_lines(values)
        else: grouped.append(PlateOCR._join_fragments(values))
    result.update(raw_lines=grouped,status='ocr_complete',raw_complete_before_formatter=True)
    dump(result_path,result)
    result['format_trace']=format_vn_plate_details(grouped)
    result.update(status='complete',formatter_called=True)
    dump(result_path,result)


def run():
    manifest=load(OUT/'manifest.json')
    status_path=OUT/'run_status.json'
    statuses=load(status_path) if status_path.exists() else []
    attempted={(s['case_id'],s['variant']) for s in statuses}
    for case in manifest['cases']:
        for variant in case['variants']:
            identifier=(case['case_id'],variant['name'])
            if identifier in attempted: continue
            stem=f'{identifier[0]}--{identifier[1]}'
            result=OUT/'runs'/f'{stem}.json';log=result.with_suffix('.log');log.parent.mkdir(parents=True,exist_ok=True)
            start=time.monotonic()
            with log.open('w',encoding='utf-8') as f:
                try:
                    completed=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'worker',
                        '--case',identifier[0],'--variant',identifier[1],'--result',str(result)],cwd=ROOT,
                        stdout=f,stderr=subprocess.STDOUT,timeout=150)
                    code=completed.returncode
                except subprocess.TimeoutExpired: code='timeout'
            value=load(result) if result.exists() else {}
            if code==0 and value.get('status')=='complete': state='completed'
            elif code=='timeout': state='timeout'
            elif isinstance(code,int) and code & 0xffffffff==0xc0000005: state='native_crash'
            else: state='process_error'
            row={'case_id':identifier[0],'variant':identifier[1],'exit_code':code,'execution_status':state,
                 'result_file':result.relative_to(OUT).as_posix(),'log_file':log.relative_to(OUT).as_posix(),
                 'seconds':round(time.monotonic()-start,2)}
            statuses.append(row);dump(status_path,statuses)
            print(json.dumps(row),flush=True)


def integrity():
    protected=load(OUT/'protected_before.json')
    changed=[name for name,digest in protected.items() if not (ROOT/name).is_file() or sha(ROOT/name)!=digest]
    manifest=load(OUT/'manifest.json');input_errors=[]
    for case in manifest['cases']:
        for variant in case['variants']:
            for item in variant['inputs']:
                if sha(OUT/item['path'])!=item['sha256']: input_errors.append(item['path'])
    keys=[(c['case_id'],v['name']) for c in manifest['cases'] for v in c['variants']]
    statuses=load(OUT/'run_status.json') if (OUT/'run_status.json').exists() else []
    completed_errors=[]
    for status in statuses:
        if status['execution_status']!='completed': continue
        value=load(OUT/status['result_file'])
        if not value.get('raw_complete_before_formatter') or not all(c.get('readtext_completed') for c in value['calls']):
            completed_errors.append([status['case_id'],status['variant']])
    result={'checked_files':len(protected),'changed_or_missing':changed,'production_modified':bool([p for p in changed if p.startswith(('src/','be/'))]),
            'input_errors':input_errors,'duplicate_candidate_keys':len(keys)-len(set(keys)),
            'invalid_completed_traces':completed_errors,'planned_candidates':len(keys),'recorded_candidates':len(statuses),
            'passed':not changed and not input_errors and len(keys)==len(set(keys)) and not completed_errors and len(keys)==len(statuses)}
    dump(OUT/'integrity.json',result);print(json.dumps(result))
    if not result['passed']: raise SystemExit(1)


def run_tests():
    import xml.etree.ElementTree as ET
    manifest=load(OUT/'manifest.json');statuses=load(OUT/'run_status.json')
    if len(statuses)!=sum(len(c['variants']) for c in manifest['cases']):
        raise RuntimeError('Finish sequential OCR jobs before model smoke tests')
    attempts=[]
    def attempt(name,extra):
        log=OUT/f'{name}.log';xml=OUT/f'{name}-results.xml'
        with log.open('wb') as handle:
            result=subprocess.run([sys.executable,'-B','-m','pytest','tests','-q','-p','no:cacheprovider',
                                  f'--junitxml={xml}',*extra],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,timeout=240)
        data={'name':name,'exit_code':result.returncode,'log':log.name,'junit':xml.name}
        if xml.exists():
            suites=ET.parse(xml).getroot().findall('testsuite')
            data.update({field:sum(int(s.get(field,0)) for s in suites) for field in ['tests','failures','errors','skipped']})
        attempts.append(data);dump(OUT/'test_results.json',{'attempts':attempts})
        return result.returncode
    code=attempt('pytest',[])
    if code & 0xffffffff==0xc0000005:
        attempt('pytest-without-native-smoke',['--ignore=tests/test_inference_smoke.py'])
    print(json.dumps(attempts))


def report():
    manifest=load(OUT/'manifest.json');statuses=load(OUT/'run_status.json')
    rows=[]
    for case in manifest['cases']:
        original_path=case['source']['filename'] if case['case_id']=='sample-1' else 'data/custom/images/test/'+case['source']['filename']
        lines=[f'# {case["case_id"]} — {case["source"]["filename"]}','',
          'Crop detector cố định; không phải ground-truth crop. Không có đáp án được người dùng xác nhận cho ca này.','',
          f'[Ảnh gốc](../../../../{original_path}) · [Detector crop nguyên pixel](inputs/raw_whole-whole.png)','',
          '![Input overview](inputs_overview.jpg)','',
          '| Phương án | Trạng thái | Raw EasyOCR / confidence | Raw lines theo geometry | Sau formatter / correction |','|---|---|---|---|---|']
        for variant in case['variants']:
            status=next((s for s in statuses if s['case_id']==case['case_id'] and s['variant']==variant['name']),{})
            value=load(OUT/status['result_file']) if status and (OUT/status['result_file']).exists() else {}
            complete=status.get('execution_status')=='completed'
            raw=[{'input_label':call['input']['label'],'results':call.get('raw',[]),
                  'readtext_completed':call.get('readtext_completed',False),'stage':call.get('stage')}
                 for call in value.get('calls',[])]
            display='; '.join(f'{part["input_label"]}: '+(', '.join(f'{r["text"]!r} ({r["confidence"]:.6f})' for r in part['results']) or ('∅' if part['readtext_completed'] else 'INCOMPLETE')) for part in raw)
            trace=value.get('format_trace',{})
            row={'case_id':case['case_id'],'filename':case['source']['filename'],'variant':variant['name'],
                 'execution_status':status.get('execution_status','not_run'),'raw':raw,
                 'raw_lines':value.get('raw_lines') if complete else None,'format_trace':trace if complete else None,
                 'input_paths':[item['path'] for item in variant['inputs']],'result_file':status.get('result_file'),
                 'log_file':status.get('log_file')}
            row['detector_region_counts']=[
                {'input_label':call['input']['label'],
                 'horizontal':sum(len(group) for group in call.get('horizontal_boxes',[])),
                 'free':sum(len(group) for group in call.get('free_boxes',[])),
                 'detector_completed':call.get('detector_completed',False)}
                for call in value.get('calls',[])]
            # Draw OCR regions on a copy, solely to inspect missed/partial boxes.
            import cv2
            import numpy as np
            for call in value.get('calls',[]):
                if not call.get('readtext_completed'): continue
                canvas=cv2.imread(str(OUT/call['input']['path']))
                for region in call.get('raw',[]):
                    polygon=np.round(np.asarray(region['bbox'])).astype(np.int32)
                    cv2.polylines(canvas,[polygon],True,(20,20,240),max(1,canvas.shape[1]//600))
                    x,y=polygon.min(axis=0)
                    cv2.putText(canvas,f'{region["text"]} ({region["confidence"]:.3f})',
                        (max(0,int(x)),max(14,int(y)-3)),cv2.FONT_HERSHEY_SIMPLEX,.45,(20,20,240),1)
                overlay=OUT/'overlays'/f'{case["case_id"]}--{variant["name"]}--{call["input"]["label"]}.jpg'
                overlay.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(overlay),canvas)
                call['overlay']=overlay.relative_to(OUT).as_posix()
            row['ocr_overlay_paths']=[c['overlay'] for c in value.get('calls',[]) if c.get('overlay')]
            rows.append(row)
            lines.append(f'| {variant["name"]} | {row["execution_status"]} | {display} | `{row["raw_lines"]}` | `{trace.get("plate_number", "")}` / {trace.get("format_status", "NOT COMPLETED")} |')
        lines+=['','## Exact inputs','']
        for variant in case['variants']:
            for item in variant['inputs']:
                lines.append(f'- [{variant["name"]}/{item["label"]}]({Path(item["path"]).relative_to("cases/"+case["case_id"]).as_posix()}): shape={item["shape"]}, SHA256={item["sha256"]}')
        lines+=['','## OCR region overlays','']
        for row in rows:
            if row['case_id']==case['case_id']:
                lines.extend(f'- [{row["variant"]}: OCR boxes](../../{p})' for p in row['ocr_overlay_paths'])
        lines+=['','## Split eligibility','', '```json',json.dumps(case['split_decisions'],indent=2),'```','',
                 'Empty OCR chỉ ghi nhận khi readtext hoàn tất và trả []; crash/timeout là thiếu kết quả, không phải chuỗi rỗng. Không xếp hạng bằng confidence hoặc regex.']
        (OUT/'cases'/case['case_id']/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    dump(OUT/'results.json',rows)
    flat=[{'case_id':r['case_id'],'variant':r['variant'],'execution_status':r['execution_status'],
           'raw_text_and_confidence':json.dumps(r['raw'],ensure_ascii=False),'raw_lines':json.dumps(r['raw_lines'],ensure_ascii=False),
           'plate_number':(r['format_trace'] or {}).get('plate_number',''),
           'format_status':(r['format_trace'] or {}).get('format_status','not_completed'),
           'corrections':json.dumps((r['format_trace'] or {}).get('corrections',[]),ensure_ascii=False),
           'input_paths':';'.join(r['input_paths']),'result_file':r['result_file'],'log_file':r['log_file']} for r in rows]
    csv_write(OUT/'raw_results.csv',flat,list(flat[0]))
    # Whole-candidate comparison: never assemble a top line from one variant and
    # a bottom line from another. Confidence is displayed, not used to rank.
    counts={s:sum(r['execution_status']==s for r in rows) for s in ['completed','native_crash','process_error','timeout','not_run']}
    test_summary=load(OUT/'test_results.json') if (OUT/'test_results.json').exists() else {'attempts':[]}
    protection=load(OUT/'integrity.json') if (OUT/'integrity.json').exists() else {}
    controls=[]
    for case_id in ['001-0','052-1']:
        before_path=ROOT/f'experiments/targeted-ocr-fix/runs/after-{case_id}.json'
        current=next(r for r in rows if r['case_id']==case_id and r['variant']=='current_whole')
        previous=load(before_path) if before_path.exists() else {}
        after=(current['format_trace'] or {}).get('plate_number')
        controls.append({'case_id':case_id,'reference_kind':'previous_prediction_NOT_ground_truth',
                         'before':previous.get('final'),'after_current_route':after,
                         'output_unchanged':previous.get('completed') is True and after==previous.get('final')})
    dump(OUT/'regression_controls.json',controls)
    def short(row):
        if row['execution_status']!='completed': return row['execution_status'].upper()
        return '; '.join(', '.join(f'{v["text"]} [{v["confidence"]:.3f}]' for v in call['results']) or '∅' for call in row['raw'])
    def lookup(case_id,variant): return next(r for r in rows if r['case_id']==case_id and r['variant']==variant)
    main=['# OCR recognition input study — fixed detector crops','',
      '**Production chưa được sửa trong task này. Không có đáp án do người dùng xác nhận cho các crop đã chọn, nên chưa chọn winner hoặc tuyên bố accuracy tăng.**','',
      '## A. Recognition findings và cách diễn giải','',
      '- Mỗi phương án chạy EasyOCR trực tiếp trước normalization/correction/formatter. JSON ghi raw text và confidence trước lần gọi formatter; đủ phân biệt biến đổi ở engine với biến đổi do hậu xử lý.',
      '- `detector_region_counts` trong results.json tách text detection khỏi recognition. Nếu CRAFT không trả vùng chữ thì readtext có thể trả [] mà recognizer không nhận được vùng nào; đây không phải formatter xóa text. Khi có vùng chữ nhưng raw bất đồng, cần ảnh gốc và text xác nhận để kết luận ký tự nào sai.',
      '- Empty OCR chỉ dùng cho call hoàn tất trả []; native crash, timeout hoặc lỗi khởi tạo có trạng thái riêng và không được tính là chuỗi rỗng.',
      '- Các khác biệt giữa raw/grayscale/current không tự chứng minh bên nào đúng. So no_deskew/current và no_clahe/current để tách ảnh hưởng từng bước. Không dùng confidence cao hơn hoặc regex hợp lệ làm đáp án.',
      '- Không có thay đổi detector, detector confidence threshold, checkpoint, dataset, OCR decoder/threshold hay dependencies. Ngôn ngữ, GPU/CPU và allowlist lấy từ config hiện tại; weights EasyOCR dùng cache, download disabled.','',
      '## B. Exact input images','',
      '| Case | Filename / nguồn crop | Kích thước crop cố định | Ảnh các phương án + kết quả đầy đủ |','|---|---|---|---|']
    for case in manifest['cases']:
        main.append(f'| {case["case_id"]} | `{case["source"]["filename"]}` | {case["fixed_crop_shape"]} | [Artefacts](cases/{case["case_id"]}/report.md) · [Overview](cases/{case["case_id"]}/inputs_overview.jpg) |')
    main+=['','Exact PNG inputs + SHA256 nằm trong manifest.json và từng case report. Overview chỉ resize để điều hướng, không phải đầu vào OCR. Detector crop PNG diagnostic được giữ nguyên pixels; sample-1 dùng bbox/padding đã ghi ở audit, không rerun YOLO.','',
      'Tất cả 7 current_whole tái tạo từ crop diagnostic khớp nguyên pixel với preprocessed artefact cũ. Với 002-0, 024-0 và sample-1, deskew tự bỏ qua: no_deskew/current giống pixels, no_clahe/gray_up cũng giống pixels. Không xem những cặp này là hai bằng chứng chất lượng độc lập.','',
      'Các phương án: raw_whole = raw BGR; gray_up_whole = grayscale + cubic upscale; no_deskew_whole = current tắt riêng deskew; no_clahe_whole = current tắt riêng CLAHE; current_whole = preprocess hiện tại + OCR toàn crop. Split chỉ được xuất khi gate chấp nhận; current_split dùng đúng pixels preprocessing, current_split_clahe thêm CLAHE trên half như production.','',
      '## C. Raw EasyOCR / confidence trước formatter','',
      'Mỗi dấu [x.xxx] là confidence của fragment EasyOCR, không phải xác suất toàn biển đúng. Full precision, boxes và stage logs nằm trong raw_results.csv / results.json / runs/. Không lấy trung bình để chọn winner.','',
      '| Case | Raw whole | Gray + upscale whole | Current whole |','|---|---|---|---|']
    for case in manifest['cases']:
        main.append('| '+case['case_id']+' | '+' | '.join(short(lookup(case['case_id'],v)) for v in ['raw_whole','gray_up_whole','current_whole'])+' |')
    main+=['','### Isolated preprocessing changes','',
      '| Case | No deskew whole | No CLAHE whole |','|---|---|---|']
    for case in manifest['cases']:
        main.append('| '+case['case_id']+' | '+' | '.join(short(lookup(case['case_id'],v)) for v in ['no_deskew_whole','no_clahe_whole'])+' |')
    main+=['','### Split khi gate hiện tại chấp nhận','',
      '| Case / input | Split row | Raw OCR theo top/bottom |','|---|---:|---|']
    for case in manifest['cases']:
        for variant in case['variants']:
            if variant['split_row'] is not None:
                main.append(f'| {case["case_id"]} / {variant["name"]} | {variant["split_row"]} | {short(lookup(case["case_id"],variant["name"]))} |')
    main+=['','### Những vấn đề quan sát được trước formatter','',
      '- **002-0/raw:** CRAFT chỉ đặt region ở hàng dưới; recognizer trả `Z3` (0.342787) cho region đó. Hàng trên không được trả thành region. [Overlay](overlays/002-0--raw_whole--whole.jpg) cho thấy mất vùng chữ và output rất ngắn đã xảy ra trong EasyOCR; không phải normalization cắt chữ. Các variant crash không có kết quả OCR để kết luận.',
      '- **006-0/raw:** trả thêm `HONDA` và `P7` ở nền ngoài biển, có confidence cao hơn nhiều fragment trên biển. [Overlay](overlays/006-0--raw_whole--whole.jpg). Do đó maximum confidence có thể chọn sai đối tượng trong padded crop; không được dùng trực tiếp để chọn text biển.',
      '- **014-0:** raw/current whole trả `59-MA`, còn grayscale/upscale trả `59-NA` và nhận thêm chữ nền `TODA`; `098.57` xuất hiện từ EasyOCR, không phải formatter tạo ra. So sánh từng vùng [gray overlay](overlays/014-0--gray_up_whole--whole.jpg) và [current overlay](overlays/014-0--current_whole--whole.jpg). Chưa có text người dùng xác nhận để xếp hạng MA/NA.',
      '- **024-0:** raw `51K.922.52` (0.845181), gray `51K-922.52` (0.517886), current `51K-922.52` (0.489604): cùng alphanumeric, khác separator/confidence. Đây là ví dụ confidence thay đổi không chứng minh khả năng đọc ký tự tốt hơn.',
      '- **052-0:** grayscale/upscale và current cho các fragment như `63-84`, `L8957` / `L89.57`; raw/confidence trước formatter có đầy đủ trong case report. Không dùng mapping 8/B hay L/1 để tự tạo đáp án. Crop gốc chỉ 65×45 px; upscale không thêm thông tin nguồn.']
    main+=['',
      '- **sample-1 / same split row 537:** giữ nguyên half rồi OCR cho bottom `S699` (0.329331); thêm CLAHE lần nữa cho bottom `FAC` (0.183302). Đây là thay đổi trước formatter từ một biến đổi ảnh duy nhất. Cả hai chưa được chứng minh đúng; không coi việc bỏ CLAHE lần hai là đã sửa recognition.',
      '- **Whole vs split sample-1:** current whole bottom `S64-9` (0.061107), current split bottom `S699` (0.329331), raw split bottom `IG68PF` (0.029079). Chưa có phương án đọc đúng đã được xác nhận; không tuyên bố split hay whole thắng.']
    main+=['','## D. Functions cần cân nhắc','',
      '- `src/inference/preprocessing.py:137 preprocess_plate`: hiện trả một ảnh duy nhất, deskew + CLAHE mặc định. Nếu có GT chứng minh cải thiện, có thể cung cấp thêm grayscale/upscale làm ứng viên riêng mà giữ nguyên crop.',
      '- `src/inference/preprocessing.py:189 find_split_row`: giữ abstain khi không đủ khoảng cách; task này không nới gate để ép split.',
      '- `src/inference/postprocessing.py:283 PlateOCR.read_text_details`: hiện chỉ chạy một route, whole hoặc split. Nhánh split áp CLAHE thêm lần nữa ở khoảng dòng 300. Ứng viên whole và split phải giữ độc lập, không ghép fragment giữa các route.',
      '- `be/services/detection_service.py:106`: caller hiện chỉ chuyển ảnh đã preprocess vào OCR; muốn thử raw/simple gray trong production phải truyền crop gốc hoặc tạo các ứng viên trước khi mất nguồn pixel. Task này không sửa caller.',
      '- `format_vn_plate_details` chỉ đánh giá sau OCR, không được dùng làm scoring function để chọn biến thể ảnh.','',
      '## E. Small strategy proposal — chưa áp dụng','',
      'Đề xuất thử nghiệm tiếp theo sau khi có text xác nhận: giữ candidate hiện tại, tạo tối đa một alternative từ grayscale/upscale nếu candidate rỗng hoặc hai route bất đồng; split chỉ khi gate chấp nhận. Lưu trọn từng candidate cùng raw tuples, vùng chữ, confidence, source pixels và timing. Khi bất đồng, không lấy top của phương án A ghép bottom của B, không tự sửa ký tự để đạt regex, không chọn chỉ bằng confidence. Cần GT xác nhận và regression nhỏ để quyết định thứ tự fallback; thiếu chứng cứ thì giữ trạng thái chưa xác định.',
      'Patch nhỏ cần kiểm chứng riêng trước tiên: trong nhánh split, so sánh readtext(half) với readtext(apply_clahe(half)) khi half đã được CLAHE trong preprocess_plate. Artefact sample-1 đã tách đúng một yếu tố này; chưa áp dụng bỏ CLAHE lặp lại vì chưa có phương án nào được người dùng xác nhận là đúng.',
      'Không có production patch hoặc file .patch được áp dụng trong task này. Việc một phương án trả text dài hơn hoặc hợp cấu trúc hơn không đủ điều kiện sửa production.','',
      '## F. Before/After và regression','',
      f'- Phạm vi: 8 crop từ 7 ảnh, gồm sáu ca ưu tiên và hai controls. {len(rows)} candidates; trạng thái: {counts}. Mỗi job chạy tuần tự trong subprocess riêng, không chạy song song model nặng.',
      f'- Unit test execution: {test_summary["attempts"]}. Các cảnh báo deprecation có trong log; không thay dependencies.',
      f'- Integrity: kiểm tra {protection.get("checked_files", "pending")} file có sẵn; changed_or_missing={protection.get("changed_or_missing", "pending")}; input_errors={protection.get("input_errors", "pending")}; PASS={protection.get("passed", "pending")}.',
      f'- Controls so với lần chạy targeted trước (prediction, không phải GT): {controls}. Gray/upscale ở 052-1 đổi CL thành CZ dù cả hai qua regex; không thể chọn chỉ vì định dạng hợp lệ.',
      '- Before là route production hiện tại; After trong bảng dưới là các alternative độc lập, chưa phải thay đổi được triển khai. Không báo delta accuracy khi không có GT.',
      '- Unit tests và integrity ghi trong test_results.json, pytest-results.xml, pytest.log, integrity.json. Native failures ghi riêng trong native_failures.json.','',
      '| Case | Route production hiện tại | Raw hiện tại | Alternative raw gray/upscale |','|---|---|---|---|']
    for case in manifest['cases']:
        route=case['current_production_first_candidate']
        main.append(f'| {case["case_id"]} | {route} | {short(lookup(case["case_id"],route))} | {short(lookup(case["case_id"],"gray_up_whole"))} |')
    main+=['','Nếu split có half rỗng, production có thể fallback whole: dữ liệu candidate giữ độc lập để không gọi text của một route là output cuối của route khác.','',
      '## G. Chưa khắc phục / thông tin thiếu','',
      '- Cần người dùng xác nhận text từ ảnh gốc cho 002-0, 006-0, 014-0, 024-0, 052-0 và sample-1; nêu rõ ký tự mơ hồ/không đọc được. Hai control 001-0 và 052-1 cũng chỉ là previously nonempty, chưa phải GT được xác nhận.',
      '- Cần kết quả chạy hoàn tất cho những phương án native crash; không thể so sánh correctness hay confidence của một call chưa kết thúc. Không suy luận missing result thành OCR rỗng.',
      '- Blur/occlusion và thông tin pixel không còn trong crop không được khôi phục chắc chắn bằng upscale hoặc CLAHE. Nếu detector crop đã thiếu ký tự, preprocessing không thể tạo lại ký tự đúng từ không có dữ liệu.',
      '- Các chuỗi cấu trúc hợp lệ nhưng sai ký tự, đa nghĩa I/1, O/0 và biến dạng phối cảnh cần text xác nhận để chọn phương án. Không lấy prediction làm ground truth.',
      '- Native CRAFT/PyTorch crash là vấn đề runtime riêng; không đổi engine, thread settings hoặc dependencies trong task này.','',
      '## Reproduce','',
      '```powershell',
      'Set-Location D:/CCNLTHD',
      './.venv/Scripts/python.exe -B scripts/recognition_input_study.py run',
      './.venv/Scripts/python.exe -B scripts/recognition_input_study.py test',
      './.venv/Scripts/python.exe -B scripts/recognition_input_study.py report',
      './.venv/Scripts/python.exe -B scripts/recognition_input_study.py integrity',
      '```','',
      '`prepare` đã chạy và từ chối ghi đè manifest. `run` bỏ qua mọi job đã ghi trạng thái, kể cả crash, để giữ nguyên evidence; không tự động retry vô hạn. Không cần chạy lại toàn bộ diagnostic benchmark.']
    (OUT/'report.md').write_text('\n'.join(main),encoding='utf-8')
    failures=[]
    for status in statuses:
        if status['execution_status']=='completed': continue
        value=load(OUT/status['result_file']) if (OUT/status['result_file']).exists() else {}
        log=(OUT/status['log_file']).read_text(encoding='utf-8',errors='replace')
        failures.append(dict(status,last_stage=value.get('calls',[{}])[-1].get('stage') if value.get('calls') else value.get('status'),
                             stack_has_craft='craft.py' in log,stack_has_max_pool2d='_max_pool2d' in log,
                             counted_as_empty_ocr=False))
    dump(OUT/'native_failures.json',failures)
    print(json.dumps({'candidates':len(rows),'completed':sum(r['execution_status']=='completed' for r in rows),
                      'native_crash':sum(r['execution_status']=='native_crash' for r in rows)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','worker','run','integrity','test','report'])
    parser.add_argument('--case');parser.add_argument('--variant');parser.add_argument('--result',type=Path)
    args=parser.parse_args()
    {'prepare':prepare,'worker':lambda:worker(args.case,args.variant,args.result),'run':run,'integrity':integrity,'test':run_tests,'report':report}[args.command]()

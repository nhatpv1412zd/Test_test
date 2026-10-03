"""One real plate crop A and B control; one targeted grouping candidate only."""
import argparse
import faulthandler
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from compare_alpr_cases import CASES,dump,load,sha
OUT=ROOT/'experiments/plate-A-recognition-fix'


def prepare():
    if (OUT/'before_hashes.json').exists():raise RuntimeError('Already prepared')
    files=[]
    for folder in ['src','be','tests','data/custom/images/test','data/custom/labels/test']:
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files += [ROOT/'weights/best.pt',ROOT/'requirements.txt']
    dump(OUT/'before_hashes.json',{p.relative_to(ROOT).as_posix():sha(p) for p in files})
    for p in files:
        if p.suffix=='.py' and p.parts[len(ROOT.parts)] in ['src','be','tests']:
            target=OUT/'before'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
    print('Prepared snapshots; only A/0 and B/0 are selected')


def worker(case,tag):
    faulthandler.enable()
    import contextlib
    import cv2
    import numpy as np
    import easyocr
    import torch
    from src.inference.preprocessing import crop_plate,preprocess_plate
    from src.inference.postprocessing import PlateOCR
    from src.utils import config
    from unittest.mock import patch
    folder=OUT/tag;folder.mkdir(parents=True,exist_ok=True)
    # Frozen original production detection index0, never the billboard index1.
    source_trace=load(ROOT/'experiments/ocr-runtime-fix'/(case+'-after')/'trace.json')
    detection=source_trace['detections'][0]
    state={'case':case,'filename':CASES[case]['filename'],'detection_index':0,'detection':detection,
        'crop_provenance':str((ROOT/'experiments/ocr-runtime-fix'/(case+'-after')/'trace.json').relative_to(ROOT)),
        'config':{k:getattr(config,k) for k in config.__all__},'ocr_calls':[],'status':'starting'}
    def save():dump(folder/'trace.json',state)
    def image(name,pixels):
        target=folder/name;cv2.imwrite(str(target),pixels)
        return {'path':name,'shape':list(pixels.shape),'sha256':sha(target)}
    original=cv2.imread(str(ROOT/'data/custom/images/test'/state['filename']))
    state['source_sha256']=sha(ROOT/'data/custom/images/test'/state['filename'])
    state['original_detector_crop']=image('01-unpadded.png',crop_plate(original,detection['xyxy']))
    crop=crop_plate(original,detection['xyxy'],padding=config.CROP_PADDING)
    state['padded_crop']=image('02-padded.png',crop);del original
    pixels=preprocess_plate(crop,upscale=config.OCR_UPSCALE)
    state['easyocr_input']=image('03-easyocr.png',pixels)
    frozen=cv2.imread(str(ROOT/'experiments/ocr-runtime-fix'/(case+'-after')/'det-0/04-upscale.png'),0)
    state['same_pixels_as_production']=bool(np.array_equal(pixels,frozen));del frozen
    assert state['same_pixels_as_production']
    real_read=easyocr.Reader.readtext;real_detect=easyocr.Reader.detect;real_recognize=easyocr.Reader.recognize
    active=None
    def read(reader,im,**kwargs):
        nonlocal active
        active={'stage':'readtext','shape':list(im.shape),'kwargs':kwargs}
        state['ocr_calls'].append(active);save()
        values=real_read(reader,im,**kwargs)
        active['raw']=[{'bbox':b,'text':t,'confidence':float(c)} for b,t,c in values]
        active['stage']='completed';save();return values
    def detect(reader,*a,**kw):
        active['stage']='craft';save();result=real_detect(reader,*a,**kw)
        active['craft']={'horizontal':result[0],'free':result[1]};save();return result
    def recognize(reader,*a,**kw):
        active['stage']='recognition';save();return real_recognize(reader,*a,**kw)
    def tensor_hook(module,args):
        if module.__class__.__name__=='CRAFT' and active is not None:
            active['craft_tensor_shape']=list(args[0].shape);save()
    hook=torch.nn.modules.module.register_module_forward_pre_hook(tensor_hook)
    save()
    try:
        with patch.object(easyocr.Reader,'readtext',read),patch.object(easyocr.Reader,'detect',detect),patch.object(easyocr.Reader,'recognize',recognize):
            state['details']=PlateOCR().read_text_details(pixels)
        state['status']='completed';save()
    finally:hook.remove()
    for i,call in enumerate(state['ocr_calls']):
        canvas=cv2.cvtColor(pixels,cv2.COLOR_GRAY2BGR)
        for j,item in enumerate(call.get('raw',[])):
            pts=np.int32(item['bbox']);cv2.polylines(canvas,[pts],True,(0,0,255),3)
            x,y=pts.min(axis=0);cv2.putText(canvas,f"{j}: {item['text']} {item['confidence']:.5f}",(max(0,int(x)),max(30,int(y))),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
        image(f'04-boxes-{i}.jpg',canvas)
    print(json.dumps({'case':case,'status':state['details']['ocr_status'],'plate_number':state['details']['plate_number']}),flush=True)


def run(phase):
    statuses=load(OUT/'runs.json') if (OUT/'runs.json').exists() else []
    for case in ['A','B']:
        tag=phase+'-'+case;folder=OUT/tag;folder.mkdir(parents=True,exist_ok=True)
        with (folder/'runtime.log').open('wb') as log:
            try:code=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'worker','--case',case,'--tag',tag],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=180).returncode
            except subprocess.TimeoutExpired:code='timeout'
        row={'case':case,'tag':tag,'exit_code':code};statuses.append(row);dump(OUT/'runs.json',statuses);print(json.dumps(row),flush=True)
        trace_path=folder/'trace.json'
        if code!=0 or not trace_path.exists():return
        details=load(trace_path).get('details',{})
        if case=='A' and (details.get('ocr_status')=='ocr_runtime_error' or not details.get('ocr_calls') or any(c['status']!='completed' or not c.get('memory_admission',{}).get('allowed',False) for c in details['ocr_calls'])):
            print('A did not complete an admitted OCR call; stop recognition conclusions',flush=True);return


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','worker','run']);parser.add_argument('--phase',default='before')
    parser.add_argument('--case',choices=['A','B']);parser.add_argument('--tag');args=parser.parse_args()
    {'prepare':prepare,'worker':lambda:worker(args.case,args.tag),'run':lambda:run(args.phase)}[args.command]()

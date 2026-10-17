"""Two-image production trace. Wrappers observe and return original values unchanged."""
import argparse
import faulthandler
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'experiments/compare-alpr-A-B'
CASES={'A':{'filename':'img_0167_png.rf.f50701c886b2d70b2ef6d4bcb798e019.jpg','user_ground_truth':'29-B1 542.70',
            'reported_confidences':[0.7889413237571716,0.5282004475593567]},
       'B':{'filename':'img_0197_png.rf.22034f6ed7ab143c0ff180392b3dd4b2.jpg','user_ground_truth':'17-AK 044.70',
            'reported_confidences':[0.7126506567001343]}}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,
        default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)),encoding='utf-8');temp.replace(path)
def load(path): return json.loads(path.read_text(encoding='utf-8-sig'))


def prepare():
    if (OUT/'protected_before.json').exists(): raise ValueError('Study already initialized')
    files=[]
    for folder in ['src','be','tests','data/custom/images/test','data/custom/labels/test']:
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files += [ROOT/'weights/best.pt',ROOT/'requirements.txt']
    dump(OUT/'protected_before.json',{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))})
    dump(OUT/'user_references.json',{'provenance':'User supplied in this task, never passed to OCR/formatter', 'cases':CASES})
    for name in ['src/inference/preprocessing.py','src/inference/postprocessing.py','be/services/detection_service.py']:
        target=OUT/'before'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    print('Prepared exactly two source images and protected hashes')


def worker(case_id,tag=None):
    faulthandler.enable()
    import contextlib
    import math
    import cv2
    import numpy as np
    import easyocr
    from fastapi.testclient import TestClient
    from src.utils import config
    from src.inference.predictor import PlateDetector
    from src.inference.postprocessing import PlateOCR
    import src.inference.postprocessing as post
    import src.inference.preprocessing as pre
    import be.services.detection_service as sm
    import be.routes.detection as route
    from be.main import app
    folder=OUT/(tag or case_id);folder.mkdir(parents=True,exist_ok=True)
    source=ROOT/'data/custom/images/test'/CASES[case_id]['filename']
    state={'case_id':case_id,'filename':source.name,'source_sha256':sha(source),'status':'initializing',
           'config':{k:getattr(config,k) for k in config.__all__},'detections':[],'traces':[]}
    active=None;active_call=None
    def checkpoint(): dump(folder/'trace.json',state)
    def image(name,im):
        path=folder/name;path.parent.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(path),im): raise RuntimeError(f'Export failed {path}')
        return {'path':path.relative_to(OUT).as_posix(),'shape':list(im.shape),'sha256':sha(path)}
    checkpoint()
    detector=PlateDetector();ocr=PlateOCR()
    reader=None
    service=sm.DetectionService(detector=detector,ocr=ocr)
    real_detect=detector.detect;real_crop=sm.crop_plate;real_preprocess=sm.preprocess_plate
    real_deskew=pre.deskew_plate;real_quad=pre._find_plate_quad;real_clahe=pre.apply_clahe
    real_split=post.split_two_lines;real_details=ocr.read_text_details
    real_readtext=None;real_craft=None;real_recognize=None
    def detect(im):
        state['status']='yolo_running';checkpoint()
        values=real_detect(im)
        state['detections']=[{'index':i,'xyxy':d.xyxy,'confidence':d.confidence,'class_id':d.class_id} for i,d in enumerate(values)]
        state['yolo_effective_args']={k:getattr(detector.model.predictor.args,k,None) for k in ['conf','iou','imgsz','max_det','agnostic_nms','classes','device','half','augment','rect']}
        canvas=im.copy()
        for i,d in enumerate(values):
            x1,y1,x2,y2=map(round,d.xyxy);cv2.rectangle(canvas,(x1,y1),(x2,y2),(0,0,255),5)
            cv2.putText(canvas,f'{i} {d.confidence:.6f}',(x1,max(30,y1-10)),cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,0,255),3)
        state['detection_overlay']=image('detections.jpg',canvas);state['status']='yolo_completed';checkpoint();return values
    def crop(im,xyxy,padding=0.):
        nonlocal active
        index=next(d['index'] for d in state['detections'] if np.allclose(d['xyxy'],xyxy))
        h,w=im.shape[:2];x1,y1,x2,y2=xyxy;px=(x2-x1)*padding;py=(y2-y1)*padding
        active={'index':index,'xyxy':list(xyxy),'padding':padding,
                'padded_xyxy':[max(0,round(x1-px)),max(0,round(y1-py)),min(w,round(x2+px)),min(h,round(y2+py))],
                'stages':{},'ocr_calls':[]}
        state['traces'].append(active)
        result=real_crop(im,xyxy,padding);active['stages']['crop']=image(f'det-{index}/00-crop.png',result)
        checkpoint();return result
    def quad(gray):
        result=real_quad(gray)
        active['deskew_quad']=result
        if result is not None:
            p=np.asarray(result);active['quad_top_edge_angle_degrees']=math.degrees(math.atan2(p[1,1]-p[0,1],p[1,0]-p[0,0]))
            canvas=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR);cv2.polylines(canvas,[np.round(p).astype(np.int32)],True,(0,0,255),3)
            active['quad_overlay']=image(f'det-{active["index"]}/quad.jpg',canvas)
        checkpoint();return result
    def deskew(im):
        result=real_deskew(im)
        active['deskew_applied']=result is not im
        active['stages']['deskew']=image(f'det-{active["index"]}/01-deskew.png',result)
        checkpoint();return result
    def clahe(gray):
        active['stages']['grayscale']=image(f'det-{active["index"]}/02-grayscale.png',gray)
        result=real_clahe(gray)
        active['stages']['clahe']=image(f'det-{active["index"]}/03-clahe.png',result)
        checkpoint();return result
    def preprocess(im,**kwargs):
        result=real_preprocess(im,**kwargs)
        active['stages']['upscale']=image(f'det-{active["index"]}/04-upscale.png',result)
        active['split_aspect_candidate']=pre.is_two_line(result)
        checkpoint();return result
    def split(im):
        halves=real_split(im)
        active['split_row']=halves[0].shape[0] if halves is not None else None
        active['split_accepted']=halves is not None
        if halves is not None:
            active['split_images']={label:image(f'det-{active["index"]}/split-{label}.png',half) for label,half in zip(['top','bottom'],halves)}
        checkpoint();return halves
    def craft(*a,**kw):
        active_call['stage']='craft_detection';checkpoint()
        result=real_craft(*a,**kw)
        active_call.update(craft_completed=True,horizontal_boxes=result[0],free_boxes=result[1]);checkpoint();return result
    def recognize(*a,**kw):
        active_call['stage']='text_recognition';checkpoint()
        result=real_recognize(*a,**kw);active_call['recognition_completed']=True;checkpoint();return result
    def readtext(im,**kwargs):
        nonlocal active_call,reader,real_readtext,real_craft,real_recognize
        index=len(active['ocr_calls'])
        active_call={'input':image(f'det-{active["index"]}/ocr-{index}.png',im),'kwargs':kwargs,'stage':'started','completed':False}
        active['ocr_calls'].append(active_call);checkpoint()
        if reader is None:
            active_call['stage']='reader_initializing_in_request_thread';checkpoint()
            reader=easyocr.Reader(ocr.languages,gpu=ocr.use_gpu,download_enabled=False,verbose=False)
            real_readtext=reader.readtext;real_craft=reader.detect;real_recognize=reader.recognize
            reader.detect=craft;reader.recognize=recognize
        result=real_readtext(im,**kwargs)
        active_call.update(stage='completed',completed=True,raw=[{'bbox':b,'text':t,'confidence':float(c)} for b,t,c in result])
        canvas=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR) if im.ndim==2 else im.copy()
        for box,text,confidence in result:
            pts=np.round(box).astype(np.int32);cv2.polylines(canvas,[pts],True,(0,0,255),2)
            x,y=pts.min(axis=0);cv2.putText(canvas,f'{text} ({confidence:.3f})',(max(0,int(x)),max(20,int(y)-5)),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,255),2)
        active_call['overlay']=image(f'det-{active["index"]}/ocr-{index}-boxes.jpg',canvas)
        checkpoint();return result
    def details(im):
        result=real_details(im);active['details']=result;checkpoint();return result
    class LazyReader:
        # Production initializes Reader at first OCR call in the request thread.
        def readtext(self,im,**kwargs): return readtext(im,**kwargs)
    ocr._reader=LazyReader()
    with contextlib.ExitStack() as stack:
        for obj,name,value in [(route,'_service',service),(detector,'detect',detect),(sm,'crop_plate',crop),
            (sm,'preprocess_plate',preprocess),(pre,'_find_plate_quad',quad),(pre,'deskew_plate',deskew),
            (pre,'apply_clahe',clahe),(post,'split_two_lines',split),(ocr,'read_text_details',details)]:
            stack.enter_context(patch.object(obj,name,value))
        with TestClient(app) as client:
            response=client.post('/detect',files={'file':(source.name,source.read_bytes(),'image/jpeg')})
        state.update(status='completed',http_status=response.status_code,api_response=response.json())
        checkpoint()
    print(json.dumps({'case':case_id,'response':state['api_response']}),flush=True)


def run(lazy_retry=False):
    statuses=load(OUT/'run_status.json') if (OUT/'run_status.json').exists() else []
    for case in ['A','B']:
        tag=case+'-lazy' if lazy_retry else case
        folder=OUT/tag;folder.mkdir(parents=True,exist_ok=True)
        with (folder/'runtime.log').open('wb') as stream:
            try:
                code=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'worker','--case',case,'--tag',tag],cwd=ROOT,
                                    stdout=stream,stderr=subprocess.STDOUT,timeout=240).returncode
            except subprocess.TimeoutExpired: code='timeout'
        row={'case':case,'tag':tag,'exit_code':code,'status':'completed' if code==0 else 'native_crash' if isinstance(code,int) and code & 0xffffffff==0xc0000005 else 'error'}
        statuses.append(row);dump(OUT/'run_status.json',statuses);print(json.dumps(row),flush=True)


def replay(case,index):
    """Exact frozen OCR pixels, separate process without resident YOLO model."""
    faulthandler.enable()
    import cv2
    from src.inference.postprocessing import PlateOCR
    source=OUT/(case+'-lazy')/f'det-{index}/04-upscale.png'
    folder=OUT/f'{case}-replay-{index}';folder.mkdir(parents=True,exist_ok=True)
    state={'input':str(source.relative_to(OUT)),'input_sha256':sha(source),'mode':'isolated exact-pixel replay; no detector resident','calls':[]}
    def save():dump(folder/'trace.json',state)
    save();ocr=PlateOCR();reader=ocr.reader
    real_detect=reader.detect;real_read=reader.readtext
    def detect(*a,**kw):
        state['stage']='craft';save();value=real_detect(*a,**kw)
        state['craft']={'horizontal':value[0],'free':value[1]};save();return value
    def read(im,**kw):
        value=real_read(im,**kw);state['calls'].append([{'bbox':b,'text':t,'confidence':float(c)} for b,t,c in value]);save();return value
    reader.detect=detect;reader.readtext=read
    state['details']=ocr.read_text_details(cv2.imread(str(source),cv2.IMREAD_GRAYSCALE));state['stage']='completed';save()
    print(json.dumps(state['details']),flush=True)


def geometry(case,index):
    """Diagnostic only: bounded-size baseline vs rotation vs projective warp."""
    faulthandler.enable()
    import cv2
    import numpy as np
    from src.inference.preprocessing import _order_quad,preprocess_plate
    from src.inference.postprocessing import PlateOCR
    source=OUT/(case+'-lazy')/f'det-{index}/00-crop.png';im=cv2.imread(str(source))
    folder=OUT/f'{case}-geometry-{index}';folder.mkdir(parents=True,exist_ok=True)
    state={'diagnostic_only':True,'production_changed':False,'max_side':1444,'source':str(source.relative_to(OUT)),'variants':{}}
    def save():dump(folder/'trace.json',state)
    variants={'bounded_whole':im}
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    thr=cv2.threshold(cv2.GaussianBlur(gray,(5,5),0),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    thr=cv2.morphologyEx(thr,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8),iterations=2)
    contour=max(cv2.findContours(thr,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],key=cv2.contourArea)
    rect=cv2.minAreaRect(contour);quad=_order_quad(cv2.boxPoints(rect));angle=float(np.degrees(np.arctan2(*(quad[1]-quad[0])[::-1])))
    state.update(bright_contour_area_fraction=cv2.contourArea(contour)/gray.size,rotation_degrees=angle,min_area_quad=quad)
    if index==0:
        h,w=gray.shape;matrix=cv2.getRotationMatrix2D((w/2,h/2),angle,1)
        nw=int(np.ceil(h*abs(matrix[0,1])+w*abs(matrix[0,0])));nh=int(np.ceil(h*abs(matrix[0,0])+w*abs(matrix[0,1])))
        matrix[0,2]+=nw/2-w/2;matrix[1,2]+=nh/2-h/2
        variants['rotation_only']=cv2.warpAffine(im,matrix,(nw,nh),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
        approx=cv2.approxPolyDP(contour,.02*cv2.arcLength(contour,True),True)
        state['polygon_vertices']=approx
        if len(approx)==4 and cv2.isContourConvex(approx):
            q=_order_quad(approx);tl,tr,br,bl=q
            ww=round(max(np.linalg.norm(tr-tl),np.linalg.norm(br-bl)));hh=round(max(np.linalg.norm(bl-tl),np.linalg.norm(br-tr)))
            target=np.float32([[0,0],[ww-1,0],[ww-1,hh-1],[0,hh-1]])
            variants['panel_projective']=cv2.warpPerspective(im,cv2.getPerspectiveTransform(q,target),(ww,hh),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
            state['projective_quad']=q
    save();ocr=PlateOCR();reader=ocr.reader;real_detect=reader.detect;active=None
    def detect(*a,**kw):
        active['stage']='craft';save();value=real_detect(*a,**kw)
        active['craft']={'horizontal':value[0],'free':value[1]};save();return value
    reader.detect=detect
    for name,pixels in variants.items():
        pixels=preprocess_plate(pixels,upscale=1,use_deskew=False)
        scale=1444/max(pixels.shape);pixels=cv2.resize(pixels,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC if scale>1 else cv2.INTER_AREA)
        cv2.imwrite(str(folder/(name+'.png')),pixels)
        active={'input':name+'.png','shape':pixels.shape,'stage':'starting'};state['variants'][name]=active;save()
        active['details']=ocr.read_text_details(pixels);active['stage']='completed'
        canvas=cv2.cvtColor(pixels,cv2.COLOR_GRAY2BGR)
        for call in active['details']['ocr_calls']:
            if call['label']=='whole':
                for item in call['results']:cv2.polylines(canvas,[np.int32(item['bbox'])],True,(0,0,255),2)
        cv2.imwrite(str(folder/(name+'-boxes.jpg')),canvas);save()
    print(json.dumps({'case':case,'completed_variants':list(state['variants'])}),flush=True)


def panel(case):
    """Manually located FRAME corners for diagnosis, never production or text GT."""
    import cv2
    import numpy as np
    from src.inference.preprocessing import preprocess_plate
    from src.inference.postprocessing import PlateOCR
    q=np.float32({'A':[[521,288],[1614,750],[1277,1618],[252,1106]],
                  'B':[[130,84],[620,149],[563,496],[105,425]]}[case])
    folder=OUT/f'{case}-panel';folder.mkdir(parents=True,exist_ok=True)
    im=cv2.imread(str(OUT/(case+'-lazy')/'det-0/00-crop.png'))
    canvas=im.copy();cv2.polylines(canvas,[np.int32(q)],True,(0,0,255),3);cv2.imwrite(str(folder/'manual-frame.jpg'),canvas)
    tl,tr,br,bl=q;w=round(max(np.linalg.norm(tr-tl),np.linalg.norm(br-bl)));h=round(max(np.linalg.norm(bl-tl),np.linalg.norm(br-tr)))
    dst=np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    lo=np.floor(q.min(axis=0)).astype(int);hi=np.ceil(q.max(axis=0)).astype(int)
    variants={'panel_axis_aligned':im[lo[1]:hi[1],lo[0]:hi[0]],
              'panel_projective':cv2.warpPerspective(im,cv2.getPerspectiveTransform(q,dst),(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)}
    state={'manual_geometry_only':True,'quad':q,'not_a_production_candidate':True,'variants':{}}
    ocr=PlateOCR();reader=ocr.reader;real_detect=reader.detect;active=None
    def save():dump(folder/'trace.json',state)
    def detect(*a,**kw):
        active['stage']='craft';save();value=real_detect(*a,**kw);active['craft']={'horizontal':value[0],'free':value[1]};save();return value
    reader.detect=detect
    for name,pixels in variants.items():
        pixels=preprocess_plate(pixels,upscale=1,use_deskew=False)
        s=1444/max(pixels.shape);pixels=cv2.resize(pixels,None,fx=s,fy=s,interpolation=cv2.INTER_CUBIC if s>1 else cv2.INTER_AREA)
        cv2.imwrite(str(folder/(name+'.png')),pixels);active={'input':name+'.png','shape':pixels.shape,'stage':'started'};state['variants'][name]=active;save()
        active['details']=ocr.read_text_details(pixels);active['stage']='completed';save()
        row=active['details']['split_row']
        if row is not None:
            cv2.imwrite(str(folder/(name+'-top.png')),pixels[:row]);cv2.imwrite(str(folder/(name+'-bottom.png')),pixels[row:])
    print('Panel diagnostic completed',case)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','worker','run','run-lazy','replay','geometry','panel'])
    parser.add_argument('--case',choices=['A','B']);parser.add_argument('--tag');parser.add_argument('--index',type=int,default=0);args=parser.parse_args()
    {'prepare':prepare,'worker':lambda:worker(args.case,args.tag),'run':run,'run-lazy':lambda:run(True),'replay':lambda:replay(args.case,args.index),'geometry':lambda:geometry(args.case,args.index),'panel':lambda:panel(args.case)}[args.command]()

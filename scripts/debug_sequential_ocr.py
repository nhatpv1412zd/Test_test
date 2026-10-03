"""Controlled two-crop runtime isolation; all heavy workers run serially."""
import argparse
import ctypes
import difflib
import faulthandler
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from compare_alpr_cases import dump,load,sha,CASES
from check_ocr_runtime import memory
OUT=ROOT/'experiments/sequential-ocr-fix'


def process_memory(handle=None):
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[
            (name,ctypes.c_size_t) for name in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage',
                'QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage',
                'PagefileUsage','PeakPagefileUsage','PrivateUsage']]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    psapi=ctypes.WinDLL('psapi',use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD]
    value=Counters();value.cb=ctypes.sizeof(value)
    if not psapi.GetProcessMemoryInfo(handle or kernel.GetCurrentProcess(),ctypes.byref(value),value.cb):
        return {'error':ctypes.get_last_error()}
    return {name:getattr(value,name) for name in ['PeakWorkingSetSize','WorkingSetSize','PrivateUsage','PeakPagefileUsage']}


def snapshot():return {'system':memory(),'process':process_memory(),'monotonic':time.monotonic()}


def prepare():
    if (OUT/'before_hashes.json').exists():raise RuntimeError('Already initialized')
    files=[]
    for name in ['src','be','tests','data/custom/images/test','data/custom/labels/test']:
        files += [p for p in (ROOT/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files += [ROOT/'weights/best.pt',ROOT/'requirements.txt']
    dump(OUT/'before_hashes.json',{p.relative_to(ROOT).as_posix():sha(p) for p in files})
    for p in files:
        if p.suffix=='.py' and p.parts[len(ROOT.parts)] in ['src','be','tests']:
            target=OUT/'before'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
    dump(OUT/'memory_before.json',snapshot());print('Snapshots saved')


def worker(tag,order,reader_mode,backend):
    faulthandler.enable()
    import os
    folder=OUT/tag;folder.mkdir(parents=True,exist_ok=True)
    dump(folder/'worker_pid.json',{'pid':os.getpid()})
    import contextlib
    import threading
    import cv2
    import torch
    import easyocr
    from src.inference.predictor import PlateDetector
    from src.inference.postprocessing import PlateOCR
    from unittest.mock import patch
    state={'order':order,'reader_mode':reader_mode,'backend':backend,'resident_yolo':True,
        'versions':{'torch':torch.__version__,'easyocr':easyocr.__version__},
        'num_threads':torch.get_num_threads(),'mkldnn_default':torch.backends.mkldnn.enabled,
        'calls':[],'initial':snapshot(),'status':'started'}
    (folder/'torch-config.txt').write_text(torch.__config__.show(),encoding='utf-8')
    current=None;names={}
    def save():dump(folder/'trace.json',state)
    def pre_hook(module,args):
        nonlocal names
        if current is None:return
        if module.__class__.__name__=='CRAFT':
            names={id(m):n for n,m in module.named_modules()}
            current.update(stage='craft_forward',craft_input_shape=list(args[0].shape),craft_dtype=str(args[0].dtype),craft_before=snapshot());save()
        elif id(module) in names and isinstance(module,torch.nn.Conv2d):
            current['last_conv']={'name':names[id(module)],'input_shape':list(args[0].shape),
                'out_channels':module.out_channels,'kernel':module.kernel_size,'before':snapshot()};save()
    def post_hook(module,args,value):
        if current is not None and module.__class__.__name__=='CRAFT':
            current.update(stage='craft_completed',craft_after=snapshot());save()
    real_detect=easyocr.Reader.detect;real_recognize=easyocr.Reader.recognize
    def detect(reader,*args,**kwargs):
        current['stage']='craft_detection';save()
        context=torch.backends.mkldnn.flags(enabled=False) if backend=='no-mkldnn' else contextlib.nullcontext()
        with context:result=real_detect(reader,*args,**kwargs)
        current['craft_boxes']={'horizontal':result[0],'free':result[1]};save();return result
    def recognize(reader,*args,**kwargs):
        current['stage']='recognition';save();return real_recognize(reader,*args,**kwargs)
    def execute():
        nonlocal current
        # Same model residency/order/thread as service. Original image stays live.
        detector=PlateDetector();original=cv2.imread(str(ROOT/'data/custom/images/test'/CASES['A']['filename']))
        state['detections']=[d._asdict() for d in detector.detect(original)]
        state['after_yolo']=snapshot();save()
        if backend=='one-thread':torch.set_num_threads(1)
        state['ocr_num_threads']=torch.get_num_threads();save()
        ocr=None
        for i,index in enumerate(order):
            path=ROOT/'experiments/ocr-runtime-fix/A-after'/f'det-{index}/04-upscale.png'
            pixels=cv2.imread(str(path),0)
            current={'detection':index,'input_shape':list(pixels.shape),'input_sha256':sha(path),
                     'stage':'before_reader','before':snapshot()}
            state['calls'].append(current);save()
            if ocr is None:ocr=PlateOCR()
            current['details']=ocr.read_text_details(pixels)
            current.update(stage='completed',after=snapshot());save()
            del pixels
            if reader_mode=='fresh':ocr=None;gc.collect();current['after_release']=snapshot();save()
        state.update(status='completed',final=snapshot());save()
    save()
    hooks=[torch.nn.modules.module.register_module_forward_pre_hook(pre_hook),torch.nn.modules.module.register_module_forward_hook(post_hook)]
    with patch.object(easyocr.Reader,'detect',detect),patch.object(easyocr.Reader,'recognize',recognize):
        errors=[]
        def run_thread():
            try:execute()
            except BaseException as exc:
                import traceback
                state.update(status='python_error',error=repr(exc),traceback=traceback.format_exc());save();errors.append(exc)
        thread=threading.Thread(target=run_thread,name='production-request-thread');thread.start();thread.join()
        for hook in hooks:hook.remove()
        if errors:raise errors[0]
    print(json.dumps({'tag':tag,'status':state['status']}),flush=True)


def launch(tag,order,reader_mode='shared',backend='default',case=None):
    from ctypes import wintypes
    folder=OUT/tag;folder.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();peak={};samples=0
    with (folder/'runtime.log').open('wb') as log:
        command=[sys.executable,'-B',str(Path(__file__).resolve())]
        command+=['api-worker','--case',case,'--tag',tag] if case else ['worker','--tag',tag,
            '--order',order,'--reader-mode',reader_mode,'--backend',backend]
        proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=kernel.OpenProcess(0x411,False,proc.pid)
        actual_pid=proc.pid;timed_out=False
        while proc.poll() is None:
            samples+=1
            pid_path=folder/'worker_pid.json'
            if pid_path.exists():
                child_pid=load(pid_path)['pid']
                if child_pid!=actual_pid:
                    if handle:kernel.CloseHandle(handle)
                    actual_pid=child_pid;handle=kernel.OpenProcess(0x411,False,actual_pid)
            if handle:
                values=process_memory(handle)
                for key,val in values.items():peak[key]=max(peak.get(key,0),val)
            if time.monotonic()-started>180:
                timed_out=True
                if handle:
                    kernel.TerminateProcess.argtypes=[wintypes.HANDLE,wintypes.UINT]
                    kernel.TerminateProcess(handle,124)
                proc.kill();proc.wait();break
            time.sleep(.25)
        if handle:kernel.CloseHandle(handle)
    code=proc.returncode
    row={'tag':tag,'order':order,'reader_mode':reader_mode,'backend':backend,'exit_code':code,
        'native_crash':code & 0xffffffff==0xc0000005,'elapsed_seconds':time.monotonic()-started,'memory_peaks':peak,'samples':samples,
        'measured_pid':actual_pid,'timed_out':timed_out,'api_case':case}
    dump(folder/'exit.json',row)
    existing=load(OUT/'runs.json') if (OUT/'runs.json').exists() else []
    existing.append(row);dump(OUT/'runs.json',existing);print(json.dumps(row),flush=True)


def matrix(backend='default'):
    for tag,order,mode in [('plate-alone','0','shared'),('sign-alone','1','shared'),
        ('plate-sign-shared','01','shared'),('sign-plate-shared','10','shared'),
        ('plate-sign-fresh','01','fresh'),('sign-plate-fresh','10','fresh')]:
        launch(('ocr1-' if backend=='one-thread' else '')+tag,order,mode,backend)


def package():
    before=load(OUT/'before_hashes.json');changed=[];diff=[]
    current=dict(before)
    for directory in ['src','be','tests']:
        for path in (ROOT/directory).rglob('*.py'):
            current.setdefault(path.relative_to(ROOT).as_posix(),None)
    for name,digest in current.items():
        path=ROOT/name
        if path.exists() and sha(path)==digest:continue
        changed.append(name);old=OUT/'before'/name
        diff.extend(difflib.unified_diff(old.read_text(encoding='utf-8').splitlines(True) if old.exists() else [],path.read_text(encoding='utf-8').splitlines(True),fromfile='a/'+name if old.exists() else '/dev/null',tofile='b/'+name))
    (OUT/'patch.diff').write_text(''.join(diff),encoding='utf-8')
    dump(OUT/'integrity.json',{'changed':changed,'weights_sha256':sha(ROOT/'weights/best.pt')})
    print(json.dumps({'changed':changed}))


def api_worker(case,tag):
    import os
    folder=OUT/tag;folder.mkdir(parents=True,exist_ok=True)
    dump(folder/'worker_pid.json',{'pid':os.getpid()})
    import check_ocr_runtime
    check_ocr_runtime.OUT=OUT
    check_ocr_runtime.worker(case,tag)


def production_sequence(tag):
    """Unwrapped production Reader initialization; one service for A -> B -> A."""
    import os
    folder=OUT/tag;folder.mkdir(parents=True,exist_ok=True)
    dump(folder/'worker_pid.json',{'pid':os.getpid()});faulthandler.enable()
    import hashlib
    import cv2
    import numpy as np
    import torch
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    import be.routes.detection as route
    from be.main import app
    from src.utils import config
    state={'mode':'actual production Reader/property; persistent service A-B-A',
           'config':{k:getattr(config,k) for k in config.__all__},'requests':[],'before':snapshot()}
    current=None;active=None;names={}
    def save():dump(folder/'trace.json',state)
    real_details=route._service._ocr.read_text_details
    real_detect=route._service._detector.detect
    def detect(image):
        result=real_detect(image);current['detections']=[x._asdict() for x in result];save();return result
    def details(image):
        nonlocal active
        index=len(current['ocr'])
        expected_path=ROOT/'experiments/ocr-runtime-fix'/(current['case']+'-after')/f'det-{index}/04-upscale.png'
        expected=cv2.imread(str(expected_path),0)
        matches=bool(np.array_equal(expected,image));del expected
        active={'index':index,'input_shape':list(image.shape),'same_pixels_as_frozen_input':matches,
                'stage':'ocr_entry','before':snapshot()}
        current['ocr'].append(active);save()
        active['details']=real_details(image);active.update(stage='completed',after=snapshot());save()
        return active['details']
    def pre(module,args):
        nonlocal names
        if active is None:return
        if module.__class__.__name__=='CRAFT':
            names={id(m):n for n,m in module.named_modules()}
            active.update(stage='craft',craft_tensor_shape=list(args[0].shape),craft_before=snapshot());save()
        elif id(module) in names and isinstance(module,torch.nn.Conv2d):
            active['last_conv']={'name':names[id(module)],'input_shape':list(args[0].shape),'before':snapshot()};save()
    hook=torch.nn.modules.module.register_module_forward_pre_hook(pre)
    save()
    try:
        with patch.object(route._service._ocr,'read_text_details',details),patch.object(route._service._detector,'detect',detect),TestClient(app) as client:
            for case in ['A','B','A']:
                current={'case':case,'ocr':[],'before':snapshot()};state['requests'].append(current);save()
                source=ROOT/'data/custom/images/test'/CASES[case]['filename'];started=time.monotonic()
                response=client.post('/detect',files={'file':(source.name,source.read_bytes(),'image/jpeg')})
                current.update(http_status=response.status_code,response=response.json(),elapsed_seconds=time.monotonic()-started,after=snapshot());save()
        state.update(status='completed',after=snapshot());save()
    finally:hook.remove()
    print(json.dumps({'responses':[x['response'] for x in state['requests']]}),flush=True)


def api_runs():
    for case,tag in [('A','api-A-after-1'),('B','api-B-after'),('A','api-A-after-2')]:
        launch(tag,'',case=case)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','matrix','worker','launch','package','api-worker','api'])
    parser.add_argument('--tag');parser.add_argument('--order',default='01')
    parser.add_argument('--case',choices=['A','B','sequence'])
    parser.add_argument('--reader-mode',choices=['shared','fresh'],default='shared')
    parser.add_argument('--backend',choices=['default','no-mkldnn','one-thread'],default='default')
    args=parser.parse_args()
    {'prepare':prepare,'matrix':lambda:matrix(args.backend),'worker':lambda:worker(args.tag,[int(x) for x in args.order],args.reader_mode,args.backend),
        'launch':lambda:launch(args.tag,args.order,args.reader_mode,args.backend,args.case),'package':package,
        'api-worker':lambda:production_sequence(args.tag) if args.case=='sequence' else api_worker(args.case,args.tag),'api':api_runs}[args.command]()

"""Targeted runtime evidence for the same two A/B images; no recognition tuning."""
import argparse
import ctypes
import difflib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import compare_alpr_cases as comparison

OUT = ROOT / 'experiments/ocr-runtime-fix'
CHANGED = ['src/utils/config.py', 'src/inference/postprocessing.py',
           'be/services/detection_service.py', 'tests/test_postprocessing.py',
           'tests/test_service.py', 'tests/test_api.py']


def memory():
    """Windows snapshots distinguish physical availability from commit headroom."""
    class Status(ctypes.Structure):
        _fields_ = [('length',ctypes.c_ulong), ('load',ctypes.c_ulong)] + [
            (name,ctypes.c_ulonglong) for name in ['total_phys','avail_phys','total_pagefile',
                'avail_pagefile','total_virtual','avail_virtual','avail_extended_virtual']]
    state = Status(); state.length = ctypes.sizeof(state)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
        return {'snapshot_error':ctypes.get_last_error()}
    return {name:getattr(state,name) for name,_ in state._fields_}


def prepare():
    if (OUT/'protected_before.json').exists(): raise RuntimeError('Already initialized')
    files=[]
    for name in ['src','be','tests','data/custom/images/test','data/custom/labels/test']:
        files.extend(p for p in (ROOT/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files += [ROOT/'weights/best.pt', ROOT/'requirements.txt']
    comparison.dump(OUT/'protected_before.json',{p.relative_to(ROOT).as_posix():comparison.sha(p) for p in files})
    for name in CHANGED:
        path=OUT/'before'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/name).read_bytes())
    comparison.dump(OUT/'memory_before.json',memory())
    print('Prepared snapshots and memory evidence')


def worker(case,tag,index=None):
    import faulthandler
    faulthandler.enable()
    import torch
    folder=OUT/tag;folder.mkdir(parents=True,exist_ok=True)
    state={'case':case,'before':memory(),'craft_forwards':[]}
    active=None; names={}
    def save(): comparison.dump(folder/'tensors.json',state)
    def before(module,args):
        nonlocal active,names
        if module.__class__.__name__=='CRAFT':
            tensor=args[0]
            active={'shape':list(tensor.shape),'dtype':str(tensor.dtype),'device':str(tensor.device),
                    'memory_before':memory(),'status':'started'}
            state['craft_forwards'].append(active);save()
            names={id(m):name for name,m in module.named_modules()}
        elif active is not None and isinstance(module,torch.nn.Conv2d) and 'first_conv' not in active:
            tensor=args[0];n,c,h,w=tensor.shape
            active['first_conv']={'input_shape':list(tensor.shape),'out_channels':module.out_channels,
                'float32_full_resolution_activation_bytes':n*module.out_channels*h*w*tensor.element_size()}
            save()
        if active is not None and isinstance(module,torch.nn.Conv2d):
            active['last_conv']={'name':names.get(id(module)), 'input_shape':list(args[0].shape),
                'out_channels':module.out_channels,'kernel':module.kernel_size,'memory_before':memory()}
            save()
    def after(module,args,value):
        nonlocal active
        if module.__class__.__name__=='CRAFT' and active is not None:
            active.update(status='completed',memory_after=memory());save();active=None
    save()
    pre=torch.nn.modules.module.register_module_forward_pre_hook(before)
    post=torch.nn.modules.module.register_module_forward_hook(after)
    comparison.OUT=OUT
    try:
        if index is None:
            comparison.worker(case,tag)
        else:
            import cv2
            from src.inference.postprocessing import PlateOCR
            source=OUT/(case+'-after')/f'det-{index}/04-upscale.png'
            result={'mode':'exact-pixel isolated replay; no resident YOLO, main thread',
                    'input_sha256':comparison.sha(source),'input':str(source.relative_to(OUT)),
                    'details':PlateOCR().read_text_details(cv2.imread(str(source),0))}
            comparison.dump(folder/'trace.json',result)
    finally:
        pre.remove();post.remove();state['after']=memory();save()


def run():
    statuses=[]
    for case in ['A','B']:
        tag=case+'-after';folder=OUT/tag;folder.mkdir(parents=True,exist_ok=True)
        with (folder/'runtime.log').open('wb') as log:
            try:
                code=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'worker','--case',case,'--tag',tag],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=240).returncode
            except subprocess.TimeoutExpired:code='timeout'
        status={'case':case,'tag':tag,'exit_code':code,
            'status':'completed' if code==0 else 'native_crash' if isinstance(code,int) and code & 0xffffffff==0xc0000005 else 'error'}
        statuses.append(status);comparison.dump(OUT/'run_status.json',statuses);print(json.dumps(status),flush=True)


def package():
    before=comparison.load(OUT/'protected_before.json')
    changes=[p for p,digest in before.items() if not (ROOT/p).exists() or comparison.sha(ROOT/p)!=digest]
    patch=[]
    for name in CHANGED:
        patch.extend(difflib.unified_diff((OUT/'before'/name).read_text(encoding='utf-8').splitlines(True),
            (ROOT/name).read_text(encoding='utf-8').splitlines(True),fromfile='a/'+name,tofile='b/'+name))
    (OUT/'production-and-tests.patch').write_text(''.join(patch),encoding='utf-8')
    comparison.dump(OUT/'integrity.json',{'changed':changes,'unexpected_changes':[p for p in changes if p not in CHANGED],
        'protected_count':len(before),'weights_sha256':comparison.sha(ROOT/'weights/best.pt'),'memory_after':memory()})
    print(json.dumps({'changed':changes}))


def isolate(case,index):
    tag=f'{case}-isolated-{index}';folder=OUT/tag;folder.mkdir(parents=True,exist_ok=True)
    with (folder/'runtime.log').open('wb') as log:
        try:
            code=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'worker',
                '--case',case,'--tag',tag,'--index',str(index)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=180).returncode
        except subprocess.TimeoutExpired:code='timeout'
    comparison.dump(folder/'exit.json',{'exit_code':code});print(json.dumps({'tag':tag,'exit_code':code}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run','worker','package','isolate'])
    parser.add_argument('--case',choices=['A','B']);parser.add_argument('--tag');parser.add_argument('--index',type=int)
    args=parser.parse_args()
    {'prepare':prepare,'run':run,'worker':lambda:worker(args.case,args.tag,args.index),'package':package,
     'isolate':lambda:isolate(args.case,args.index)}[args.command]()

"""Reproduce one cached OCR input in isolation, without changing OCR settings.

Run with -X faulthandler. This records stage boundaries before a native crash.
It never substitutes its output for benchmark results.
"""
import argparse
import faulthandler
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
faulthandler.enable()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('image', type=Path)
    ap.add_argument('--output', type=Path, default=ROOT/'experiments/diagnostic-test/native_probe.jsonl')
    args = ap.parse_args()
    import cv2
    import easyocr
    import numpy as np
    import torch
    from src.utils.config import OCR_LANGUAGES, OCR_USE_GPU, OCR_ALLOWLIST
    def event(stage, **values):
        data = {'stage':stage,'seconds':time.perf_counter()-start,**values}
        with args.output.open('a',encoding='utf-8') as f:
            f.write(json.dumps(data,ensure_ascii=False,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n')
        print(data,flush=True)
    start = time.perf_counter()
    baseline_env = args.output.parent/'environment.json'
    if baseline_env.exists():
        baseline = json.loads(baseline_env.read_text(encoding='utf-8'))
        torch.set_num_threads(baseline.get('runtime_torch_threads_after_inference',8))
        cv2.setNumThreads(baseline.get('opencv_threads',1))
    im = cv2.imdecode(np.frombuffer(args.image.read_bytes(),np.uint8),cv2.IMREAD_GRAYSCALE)
    event('input',path=str(args.image),shape=list(im.shape),torch_threads=torch.get_num_threads(),opencv_threads=cv2.getNumThreads())
    reader = easyocr.Reader(OCR_LANGUAGES,gpu=OCR_USE_GPU,download_enabled=False)
    event('reader_ready',device=str(reader.device))
    real_detect,real_recognize = reader.detect,reader.recognize
    def detect(image,*a,**kw):
        event('text_detect_start',shape=list(image.shape))
        result = real_detect(image,*a,**kw)
        event('text_detect_end',boxes=result)
        return result
    def recognize(image,*a,**kw):
        event('text_recognize_start',shape=list(image.shape))
        result = real_recognize(image,*a,**kw)
        event('text_recognize_end',result=result)
        return result
    reader.detect,reader.recognize = detect,recognize
    result = reader.readtext(im,allowlist=OCR_ALLOWLIST)
    event('completed',result=result)


if __name__=='__main__':
    main()
